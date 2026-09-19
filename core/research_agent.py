"""
research_agent.py — LangGraph Agentic Quant Research Workflow
"""
import os
import sys

# Ensure UTF-8 output for emojis in terminal
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import logging
import argparse
from typing import TypedDict
from dotenv import load_dotenv

# --------------- Env setup ---------------
load_dotenv()
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# --------------- Logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# --------------- LangGraph / LLM ---------------
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --------------- Core modules ---------------
from data_loader import DataLoader
from features import FeatureEngineer
from labels import LabelGenerator
from optimizer import QuantOptimizer
from visualizer import Visualizer


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

class AgentState(TypedDict):
    iteration: int
    feedback: str
    tp_range: list
    sl_range: list
    top_results: list
    best_precision: float
    best_config: dict
    messages: list

def make_nodes(optimizer, llm):
    def researcher_node(state: AgentState):
        log.info(f"[Researcher] Iteration {state['iteration']}")
        if state['iteration'] == 0:
            state['tp_range'] = [0.03, 0.05, 0.08]
            state['sl_range'] = [0.01, 0.02]
            state['messages'].append(SystemMessage(content="Started research."))
        else:
            clean_results = [{'tp_pct': r['tp_pct'], 'sl_pct': r['sl_pct'], 'precision': r['precision']} for r in state['top_results']]
            prompt = (
                "You are a Quant Researcher. Our last hyperparameter sweep results:\n"
                f"{clean_results}\n"
                f"Evaluator feedback: {state['feedback']}\n\n"
                "Propose new TP and SL values to test. "
                "Respond ONLY in this exact format (no markdown):\n"
                "TP: 0.04, 0.06, 0.09\n"
                "SL: 0.015, 0.025"
            )
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                text = _extract_text(response.content)
                lines = text.strip().split('\n')
                tp_line = [l for l in lines if 'TP:' in l][0]
                sl_line = [l for l in lines if 'SL:' in l][0]
                tps = [float(x.strip()) for x in tp_line.split('TP:')[1].split(',')]
                sls = [float(x.strip()) for x in sl_line.split('SL:')[1].split(',')]
                state['tp_range'] = tps
                state['sl_range'] = sls
                log.info(f"  LLM proposed: TP={tps}, SL={sls}")
            except Exception as e:
                log.warning(f"  LLM parse failed ({e}), using fallback grid")
                state['tp_range'] = [0.04, 0.06, 0.10]
                state['sl_range'] = [0.015, 0.025]
        return state

    def executor_node(state: AgentState):
        log.info("[Executor] Running parameter sweep ...")
        results = optimizer.run_sweep(state['tp_range'], state['sl_range'])
        state['top_results'] = results
        if results:
            state['best_precision'] = results[0]['precision']
            state['best_config'] = results[0]
        else:
            state['best_precision'] = 0.0
        return state

    def evaluator_node(state: AgentState):
        prec = state['best_precision']
        log.info(f"[Evaluator] Best precision = {prec*100:.2f}%")
        if prec >= 0.80:
            log.info("  -> PASS! High-precision model found.")
            state['feedback'] = "PASS"
        elif state['iteration'] >= 3:
            log.info("  -> Max iterations reached. Stopping.")
            state['feedback'] = "FAIL_MAX_ITER"
        else:
            state['feedback'] = f"Precision {prec:.4f} too low. Try wider TP or tighter SL."
            state['iteration'] += 1
        return state

    return researcher_node, executor_node, evaluator_node

def routing_logic(state: AgentState):
    if state['feedback'] in ("PASS", "FAIL_MAX_ITER"):
        return END
    return "researcher"

def build_graph(optimizer, llm):
    researcher, executor, evaluator = make_nodes(optimizer, llm)
    g = StateGraph(AgentState)
    g.add_node("researcher", researcher)
    g.add_node("executor", executor)
    g.add_node("evaluator", evaluator)
    g.set_entry_point("researcher")
    g.add_edge("researcher", "executor")
    g.add_edge("executor", "evaluator")
    g.add_conditional_edges("evaluator", routing_logic)
    return g.compile()

def main(symbol):
    print("=" * 50)
    print(f" Agentic Quant Research System - {symbol}")
    print("=" * 50)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", symbol.lower(), f"{symbol}_4h_full.csv")

    if not os.path.exists(data_file):
        log.error(f"Data file not found: {data_file}")
        log.info(f"Please run: python core/data_downloader.py --symbol {symbol}")
        sys.exit(1)

    loader = DataLoader(data_file, symbol)
    base_df = loader.get_full_data()

    fe = FeatureEngineer(base_df)
    feature_df = fe.generate_all_features()

    optimizer = QuantOptimizer(feature_df, LabelGenerator)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    app = build_graph(optimizer, llm)

    initial_state = {
        "iteration": 0, "feedback": "", "tp_range": [], "sl_range": [],
        "top_results": [], "best_precision": 0.0, "best_config": {}, "messages": []
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print(" FINAL RESULTS")
    print("=" * 50)
    if final_state['best_config']:
        cfg = final_state['best_config']
        tp_str = f"{cfg['tp_pct']*100:.1f}%"
        sl_str = f"{cfg['sl_pct']*100:.1f}%"
        prec_str = f"{cfg['precision']*100:.2f}%"
        
        print(f"  เป้าหมายทำกำไร (Take Profit): {tp_str}")
        print(f"  จุดตัดขาดทุน (Stop Loss): {sl_str}")
        print(f"  ความแม่นยำ (Precision): {prec_str} (โมเดลทายว่าจะชน TP แล้วชนจริงกี่เปอร์เซ็นต์)")
        
        print("\n  [Top 10 Feature Importances]")
        top_feats = cfg.get('top_features', {})
        for i, (feat, score) in enumerate(top_feats.items(), 1):
            print(f"    {i}. {feat}: {score:.4f}")

        import matplotlib.pyplot as plt
        import shap
        
        res_long = cfg.get('res_long', {})
        res_short = cfg.get('res_short', {})

        # 1. SHAP Values Plot
        def save_shap_plot(model, X, filename):
            shap_img_path = os.path.join(base_dir, "data", symbol.lower(), filename)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X, show=False)
                plt.savefig(shap_img_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"  [OK] SHAP Plot saved to {shap_img_path}")
            except Exception as e:
                print(f"  [ERROR] Failed to generate SHAP plot {filename}: {e}")

        if res_long and 'last_model' in res_long:
            save_shap_plot(res_long['last_model'], res_long['last_X_test'], "shap_long.png")
        if res_short and 'last_model' in res_short:
            save_shap_plot(res_short['last_model'], res_short['last_X_test'], "shap_short.png")

        # 2. Confusion Matrix & Trade Stats
        wins_long = sum(1 for yt, yp in zip(res_long.get('all_y_true', []), res_long.get('all_y_pred', [])) if yp == 1 and yt == 1) if res_long else 0
        losses_long = sum(1 for yt, yp in zip(res_long.get('all_y_true', []), res_long.get('all_y_pred', [])) if yp == 1 and yt == 0) if res_long else 0
        
        wins_short = sum(1 for yt, yp in zip(res_short.get('all_y_true', []), res_short.get('all_y_pred', [])) if yp == 1 and yt == 1) if res_short else 0
        losses_short = sum(1 for yt, yp in zip(res_short.get('all_y_true', []), res_short.get('all_y_pred', [])) if yp == 1 and yt == 0) if res_short else 0

        total_trades = wins_long + losses_long + wins_short + losses_short
        wins = wins_long + wins_short
        losses = losses_long + losses_short
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # Save Summary Report to markdown file
        report_path = os.path.join(base_dir, "data", symbol.lower(), "summary_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 Quant Research Report: {symbol} (Long/Short Dual Model)\n\n")
            f.write("## 1. ⚙️ การตั้งค่าที่ให้ผลลัพธ์ดีที่สุด (Best Configuration)\n")
            f.write(f"- **Take Profit (TP):** {tp_str}\n")
            f.write(f"- **Stop Loss (SL):** {sl_str}\n")
            f.write(f"- **ความแม่นยำรวม (Precision):** {prec_str}\n\n")
            f.write("> **💡 ความหมาย:** เมื่อโมเดลสั่งเทรด ราคาจะมีโอกาสวิ่งไปชน TP ก่อน SL ด้วยความแม่นยำประมาณ " + prec_str + "\n\n")
            
            f.write("## 2. 📉 ผลการจำลองเทรด (Trade Simulation & Confusion Matrix)\n")
            f.write(f"- **จำนวนไม้ทั้งหมดที่โมเดลบอกให้เทรด (Total Trades):** {total_trades} (Long: {wins_long+losses_long}, Short: {wins_short+losses_short})\n")
            f.write(f"- **ชนะ (Wins):** {wins}\n")
            f.write(f"- **แพ้ (Losses):** {losses}\n")
            f.write(f"- **Win Rate รวม:** {win_rate:.2f}%\n\n")
            
            f.write("## 3. 🧠 SHAP Values (Explainable AI)\n")
            f.write("วิเคราะห์ว่า Feature แต่ละตัวส่งผลอย่างไรต่อการตัดสินใจของโมเดล (จุดสีแดง = ค่าสูง, จุดสีน้ำเงิน = ค่าต่ำ)\n\n")
            f.write("### ฝั่ง Long\n")
            f.write(f"![SHAP Long](./shap_long.png)\n\n")
            f.write("### ฝั่ง Short\n")
            f.write(f"![SHAP Short](./shap_short.png)\n\n")
            
            f.write("## 4. 🔍 ปัจจัยที่มีผลต่อการตัดสินใจมากที่สุด (Top Feature Importances)\n")
            for i, (feat, score) in enumerate(top_feats.items(), 1):
                f.write(f"{i}. **{feat}** (Score: {score:.4f})\n")
        
        print(f"\n  📝 บันทึกรายงานสรุปผลไว้ที่: {report_path}")

        # Visualization calls
        labeler = LabelGenerator(feature_df, window=20, tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'])
        labeled_df = labeler.generate_labels()
        
        # Add ML Predictions to labeled_df for visual backtesting
        if res_long and 'last_model' in res_long:
            try:
                model_long = res_long['last_model']
                expected_cols = res_long['last_X_test'].columns
                X_full = labeled_df[expected_cols]
                labeled_df['ML_Prob_Long'] = model_long.predict_proba(X_full)[:, 1]
            except Exception as e:
                print(f"  [ERROR] Failed to generate ML Long predictions: {e}")
                
        if res_short and 'last_model' in res_short:
            try:
                model_short = res_short['last_model']
                expected_cols = res_short['last_X_test'].columns
                X_full = labeled_df[expected_cols]
                labeled_df['ML_Prob_Short'] = model_short.predict_proba(X_full)[:, 1]
            except Exception as e:
                print(f"  [ERROR] Failed to generate ML Short predictions: {e}")

        vis = Visualizer(labeled_df)
        
        vis.plot_feature_distributions(list(top_feats.keys()))
        vis.plot_evaluation_metrics(res_long, res_short, cfg['tp_pct'], cfg['sl_pct'])
        vis.plot_correlation_heatmap(list(top_feats.keys()))
        vis.plot_results(feature_importances=top_feats, tail_bars=500)
    else:
        print("  ไม่มีการตั้งค่าใดที่ให้ผลลัพธ์ผ่านเกณฑ์ (No valid configuration found).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Quant Research Agent')
    parser.add_argument('--symbol', type=str, default='ZECUSDT', help='Trading pair symbol (e.g., ZECUSDT)')
    args = parser.parse_args()
    
    main(args.symbol.upper())
