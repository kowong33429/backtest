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
            state['tp_range'] = [0.05, 0.10, 0.15, 0.20]
            state['sl_range'] = [0.05, 0.10, 0.15, 0.20]
            state['messages'].append(SystemMessage(content="Started research."))
        else:
            clean_results = [{'tp_pct': r['tp_pct'], 'sl_pct': r['sl_pct'], 'precision': r['precision']} for r in state['top_results']]
            prompt = (
                "You are a Quant Researcher. Our last hyperparameter sweep results:\n"
                f"{clean_results}\n"
                f"Evaluator feedback: {state['feedback']}\n\n"
                "Propose new TP and SL values to test. Aim for trend following with wide TP, but KEEP SL <= 0.30 to avoid bad entry points. "
                "Respond ONLY in this exact format (no markdown):\n"
                "TP: 0.15, 0.20, 0.30\n"
                "SL: 0.10, 0.20, 0.30"
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
                state['tp_range'] = [0.08, 0.12, 0.20]
                state['sl_range'] = [0.08, 0.12, 0.20]
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
            state['feedback'] = f"Precision {prec:.4f} too low. Try wider SL to give the trade more room to breathe."
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

# Known big-move windows the strategy should ideally catch (OOS sanity check).
# NOTE: this is a READ-ONLY diagnostic. We do NOT retrain to fit these windows —
# that would be curve-fitting to known outcomes. It only reports whether the
# walk-forward backtest happened to trade inside them.
TARGET_WINDOWS = [
    ("2024-07-09", "2024-08-24"),
    ("2024-10-14", "2024-12-08"),
    ("2025-09-06", "2025-11-16"),
    ("2026-03-31", "2026-05-24"),
    ("2026-08-17", None),  # open / ongoing
]


def _net_sharpe(trade_log):
    """Annualized Sharpe on NET per-trade returns (Rule #9: Sharpe on net profit
    only). Uses Net_Return when present, else gross PnL_Pct. Scales the per-trade
    Sharpe by sqrt(trades per year) inferred from the actual date span.
    Returns 0.0 if there are too few trades or zero dispersion."""
    import numpy as np
    import pandas as pd
    if trade_log is None or trade_log.empty:
        return 0.0
    col = 'Net_Return' if 'Net_Return' in trade_log.columns else 'PnL_Pct'
    r = np.asarray(trade_log[col], dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    entry = pd.to_datetime(trade_log['Entry_Date'])
    exit_ = pd.to_datetime(trade_log['Exit_Date'])
    span_days = max((exit_.max() - entry.min()).days, 1)
    trades_per_year = len(r) / (span_days / 365.25)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(max(trades_per_year, 1e-9)))


def run_exit_sweep(labeled_df, thr_long, thr_short, stagnation_bars,
                   stagnation_min_profit_pct, cfg,
                   trail_grid=(0.10, 0.15, 0.20, 0.25, 0.30),
                   atr_grid=(1.5, 2.0, 2.5, 3.0)):
    """Sweep the two exit knobs (trailing-stop % × ATR SL multiplier) and record
    the NET Sharpe of each combo (AGENTS.md Rule #10 Step 5 stability heatmap).

    This is a robustness diagnostic ONLY — the main run keeps its default exit
    params. We do NOT auto-adopt the best cell: a single spiky cell that beats a
    smooth neighborhood is over-fit, and the heatmap is there to show the human
    whether the good region is a plateau or a lucky pixel."""
    import pandas as pd
    from backtester import RealisticBacktester
    rows = []
    print(f"  [Sweep] Exit-parameter stability: {len(trail_grid)}×{len(atr_grid)} combos ...")
    for trail in trail_grid:
        for atr_mult in atr_grid:
            bt = RealisticBacktester(
                labeled_df, tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'], max_bars=300,
                trail_pct=trail, atr_sl_mult=atr_mult, use_trailing=True,
                long_threshold=thr_long, short_threshold=thr_short,
                stagnation_bars=stagnation_bars,
                stagnation_min_profit_pct=stagnation_min_profit_pct,
            )
            tl = bt.run()
            sharpe = _net_sharpe(tl)
            net_ret = float(tl['Net_Return'].sum()) if (not tl.empty and 'Net_Return' in tl.columns) else 0.0
            rows.append({'trail_pct': trail, 'atr_sl_mult': atr_mult,
                         'sharpe': sharpe, 'net_return': net_ret, 'n_trades': len(tl)})
    df = pd.DataFrame(rows)
    if not df.empty:
        best = df.loc[df['sharpe'].idxmax()]
        print(f"  [Sweep] Best cell (diagnostic only): trail={best['trail_pct']}, "
              f"atr={best['atr_sl_mult']}, Sharpe={best['sharpe']:.2f} | "
              f"default trail=0.15/atr=2.0 kept for the main run.")
    return df


def build_validation_report(trade_log):
    """Return a markdown string with (a) per-year PnL/win-rate (regime view) and
    (b) whether trades overlap the known TARGET_WINDOWS. Prints a summary too."""
    import pandas as pd
    lines = ["## 5. 🔬 Walk-Forward Validation (Regime & Target Windows)\n"]

    if trade_log is None or trade_log.empty:
        lines.append("_No trades to validate._\n")
        print("  [Validation] No trades.")
        return "\n".join(lines)

    tl = trade_log.copy()
    tl['Entry_Date'] = pd.to_datetime(tl['Entry_Date'])
    tl['Exit_Date'] = pd.to_datetime(tl['Exit_Date'])

    # Prefer the NET return (Kelly-sized, after fees/slippage — Rule #7 & #9)
    # when the backtester provides it; otherwise fall back to raw gross PnL_Pct.
    ret_col = 'Net_Return' if 'Net_Return' in tl.columns else 'PnL_Pct'
    ret_label = "Net Return" if ret_col == 'Net_Return' else "Gross PnL"
    print(f"  [Validation] Using '{ret_col}' as the return column.")

    # (a) Per-year breakdown — exposes regime dependence (e.g. strong 2021-22,
    #     weak 2023-25) without cropping the training data.
    lines.append("### Per-Year Performance\n")
    lines.append(f"| Year | Trades | Win Rate | {ret_label} % (sum) |")
    lines.append("|------|--------|----------|-----------------|")
    tl['Year'] = tl['Entry_Date'].dt.year
    print("  [Validation] Per-year performance:")
    for year, grp in tl.groupby('Year'):
        n = len(grp)
        wr = (grp[ret_col] > 0).mean() * 100
        net = grp[ret_col].sum() * 100
        lines.append(f"| {year} | {n} | {wr:.1f}% | {net:+.1f}% |")
        print(f"    {year}: {n} trades, win {wr:.1f}%, {ret_label.lower()} {net:+.1f}%")

    # (b) Target-window overlap — did the walk-forward model catch the big moves?
    lines.append("\n### Target-Window Coverage (did we catch the big moves?)\n")
    lines.append("| Window | Trades inside | Net PnL % |")
    lines.append("|--------|---------------|-----------|")
    print("  [Validation] Target-window coverage:")
    for start, end in TARGET_WINDOWS:
        w_start = pd.Timestamp(start)
        w_end = pd.Timestamp(end) if end else tl['Exit_Date'].max()
        # A trade overlaps the window if entry<=w_end and exit>=w_start.
        inside = tl[(tl['Entry_Date'] <= w_end) & (tl['Exit_Date'] >= w_start)]
        net = inside[ret_col].sum() * 100
        label = f"{start} → {end or 'open'}"
        hit = "✅" if len(inside) > 0 else "❌"
        lines.append(f"| {label} | {hit} {len(inside)} | {net:+.1f}% |")
        print(f"    {label}: {hit} {len(inside)} trades, net {net:+.1f}%")

    return "\n".join(lines)


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

        # 2. Add ML Predictions to feature_df for backtesting.
        # Use WALK-FORWARD OUT-OF-FOLD probabilities (AGENTS.md Rule #8): each bar
        # is scored only by a fold-model that never trained on it, so the backtest
        # contains no in-sample leakage. The OOF array is aligned to feature_df's
        # row positions (labels.generate_labels adds columns without reindexing).
        # The initial training block has no OOF prediction (NaN -> 0 -> no trade),
        # which is the expected walk-forward warm-up.
        import numpy as np
        labeler = LabelGenerator(feature_df, tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'], max_bars=700)
        labeled_df = labeler.generate_labels()

        def _attach_oof(res, col):
            if res and res.get('oof_prob') is not None:
                oof = np.asarray(res['oof_prob'], dtype=float)
                if len(oof) == len(labeled_df):
                    labeled_df[col] = np.nan_to_num(oof, nan=0.0)
                    covered = int(np.isfinite(oof).sum())
                    print(f"  [OOF] {col}: {covered}/{len(oof)} bars scored out-of-fold "
                          f"({covered/len(oof)*100:.0f}% coverage, rest = warm-up).")
                    return
                print(f"  [WARN] OOF length {len(oof)} != df {len(labeled_df)} for {col}; using 0.")
            labeled_df[col] = 0.0

        _attach_oof(res_long, 'ML_Prob_Long')
        _attach_oof(res_short, 'ML_Prob_Short')

        # 3. Run Realistic Backtester (ATR-based SL + Trailing Stop)
        from backtester import RealisticBacktester
        # Probability thresholds come from the optimizer's out-of-fold selection
        # (per model), not a hardcoded cutoff — so the backtest trades on the
        # same rule the precision was measured at.
        thr_long = res_long.get('prob_threshold', 0.8) if res_long else 0.8
        thr_short = res_short.get('prob_threshold', 0.8) if res_short else 0.8
        print(f"  Entry thresholds -> Long: {thr_long:.2f}, Short: {thr_short:.2f}")

        # --- Exit config (AGENTS.md Rule #6, two layers) ---
        #   Layer 1: ATR trailing stop (always on via use_trailing).
        #   Layer 2: "price not moving" — close if the trade has not reached
        #            profit within STAGNATION_BARS (7 days on 4H = 42 bars).
        STAGNATION_BARS = 42            # 7 days on the 4H timeframe
        STAGNATION_MIN_PROFIT_PCT = 0.0  # any profit by then keeps it open
        backtester = RealisticBacktester(
            labeled_df, tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'], max_bars=300,
            trail_pct=0.15, atr_sl_mult=2.0, use_trailing=True,
            long_threshold=thr_long, short_threshold=thr_short,
            stagnation_bars=STAGNATION_BARS,
            stagnation_min_profit_pct=STAGNATION_MIN_PROFIT_PCT
        )
        trade_log = backtester.run()
        
        if not trade_log.empty:
            trade_log_path = os.path.join(base_dir, "data", symbol.lower(), "trade_log.csv")
            trade_log.to_csv(trade_log_path, index=False)
            print(f"  📝 บันทึกประวัติการเทรดแบบสมจริง (Trade Log) ไว้ที่: {trade_log_path}")
            
            total_trades = len(trade_log)
            wins = len(trade_log[trade_log['PnL_Pct'] > 0])
            losses = len(trade_log[trade_log['PnL_Pct'] <= 0])
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            wins_long = len(trade_log[(trade_log['Type'] == 'Long') & (trade_log['PnL_Pct'] > 0)])
            losses_long = len(trade_log[(trade_log['Type'] == 'Long') & (trade_log['PnL_Pct'] <= 0)])
            wins_short = len(trade_log[(trade_log['Type'] == 'Short') & (trade_log['PnL_Pct'] > 0)])
            losses_short = len(trade_log[(trade_log['Type'] == 'Short') & (trade_log['PnL_Pct'] <= 0)])

            # Net performance (Kelly-sized, after fees/slippage — Rule #7 & #9)
            if 'Equity' in trade_log.columns and 'Net_Return' in trade_log.columns:
                final_equity = float(trade_log['Equity'].iloc[-1])
                net_total_return = (final_equity / 10000.0 - 1.0) * 100
            else:
                final_equity, net_total_return = 10000.0, 0.0
            net_sharpe = _net_sharpe(trade_log)
        else:
            total_trades, wins, losses, win_rate = 0, 0, 0, 0
            wins_long, losses_long, wins_short, losses_short = 0, 0, 0, 0
            final_equity, net_total_return, net_sharpe = 10000.0, 0.0, 0.0

        # 4. Save Summary Report to markdown file
        report_path = os.path.join(base_dir, "data", symbol.lower(), "summary_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 Quant Research Report: {symbol} (Long/Short Dual Model)\n\n")
            f.write("## 1. ⚙️ การตั้งค่าที่ให้ผลลัพธ์ดีที่สุด (Best Configuration)\n")
            f.write(f"- **Take Profit (TP):** {tp_str}\n")
            f.write(f"- **Stop Loss (SL):** {sl_str}\n")
            f.write(f"- **ความแม่นยำรวม (Precision ในรอบเทรน):** {prec_str}\n\n")
            
            f.write("## 2. 📉 ผลการจำลองเทรดจริง (Realistic Trade Simulation)\n")
            f.write("จำลองแบบเปิด 1 ไม้ เดินหน้าหาจุด TP/SL จริงๆ ไม่เปิดซ้อนทับกัน (No Overlapping)\n")
            f.write(f"- **จำนวนไม้ทั้งหมด (Total Trades):** {total_trades} (Long: {wins_long+losses_long}, Short: {wins_short+losses_short})\n")
            f.write(f"- **ชนะ (Wins):** {wins}\n")
            f.write(f"- **แพ้ (Losses):** {losses}\n")
            f.write(f"- **Win Rate รวม:** {win_rate:.2f}%\n")
            f.write("\n### 💰 ผลตอบแทนสุทธิ (Net — Kelly-sized, หักค่าธรรมเนียม/Slippage แล้ว | Rule #7 & #9)\n")
            f.write(f"- **เงินทุนเริ่มต้น (Initial Equity):** ${10000:,.0f}\n")
            f.write(f"- **เงินทุนสุดท้าย (Final Equity):** ${final_equity:,.0f}\n")
            f.write(f"- **ผลตอบแทนสุทธิรวม (Net Total Return):** {net_total_return:+.1f}%\n")
            f.write(f"- **Sharpe Ratio (annualized, net):** {net_sharpe:.2f}\n\n")
            
            f.write("## 3. 🧠 SHAP Values (Explainable AI)\n")
            f.write("วิเคราะห์ว่า Feature แต่ละตัวส่งผลอย่างไรต่อการตัดสินใจของโมเดล (จุดสีแดง = ค่าสูง, จุดสีน้ำเงิน = ค่าต่ำ)\n\n")
            f.write("### ฝั่ง Long\n")
            f.write(f"![SHAP Long](./shap_long.png)\n\n")
            f.write("### ฝั่ง Short\n")
            f.write(f"![SHAP Short](./shap_short.png)\n\n")
            
            f.write("## 4. 🔍 ปัจจัยที่มีผลต่อการตัดสินใจมากที่สุด (Top Feature Importances)\n")
            for i, (feat, score) in enumerate(top_feats.items(), 1):
                f.write(f"{i}. **{feat}** (Score: {score:.4f})\n")

            # Walk-forward validation: per-year regime view + target-window coverage
            f.write("\n" + build_validation_report(trade_log) + "\n")

        print(f"\n  📝 บันทึกรายงานสรุปผลไว้ที่: {report_path}")

        # 5. Visualization calls
        vis = Visualizer(labeled_df, trade_log=trade_log if not trade_log.empty else None, symbol=symbol)
        
        vis.plot_results(feature_importances=top_feats, tail_bars=None)
        vis.plot_feature_distributions(list(top_feats.keys()))
        vis.plot_evaluation_metrics(res_long, res_short, cfg['tp_pct'], cfg['sl_pct'])
        vis.plot_correlation_heatmap(list(top_feats.keys()))

        # 6. Exit-parameter stability sweep + 2D heatmap (Rule #10 Step 5).
        #    Diagnostic only — the main backtest above keeps the default exit
        #    params; this shows whether good performance is a robust plateau.
        if not trade_log.empty:
            sweep_df = run_exit_sweep(
                labeled_df, thr_long, thr_short,
                STAGNATION_BARS, STAGNATION_MIN_PROFIT_PCT, cfg)
            sweep_path = os.path.join(base_dir, "data", symbol.lower(), "exit_sweep.csv")
            sweep_df.to_csv(sweep_path, index=False)
            print(f"  📝 บันทึกผล Exit-Parameter Sweep ไว้ที่: {sweep_path}")
            vis.plot_param_stability(sweep_df)
    else:
        print("  ไม่มีการตั้งค่าใดที่ให้ผลลัพธ์ผ่านเกณฑ์ (No valid configuration found).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Quant Research Agent')
    parser.add_argument('--symbol', type=str, default='ZECUSDT', help='Trading pair symbol (e.g., ZECUSDT)')
    args = parser.parse_args()
    
    main(args.symbol.upper())
