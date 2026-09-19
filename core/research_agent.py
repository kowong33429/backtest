"""
research_agent.py — LangGraph Agentic Quant Research Workflow
"""
import os
import sys
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
            prompt = (
                "You are a Quant Researcher. Our last hyperparameter sweep results:\n"
                f"{state['top_results']}\n"
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
        print(f"  TP = {cfg['tp_pct']*100:.1f}%")
        print(f"  SL = {cfg['sl_pct']*100:.1f}%")
        print(f"  Precision = {cfg['precision']*100:.2f}%")
        print(f"  Top Features = {cfg.get('top_features', {})}")

        labeler = LabelGenerator(feature_df, window=20, tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'])
        labeled_df = labeler.generate_labels()
        vis = Visualizer(labeled_df)
        vis.plot_results(tail_bars=500)
    else:
        print("  No valid configuration found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Quant Research Agent')
    parser.add_argument('--symbol', type=str, default='ZECUSDT', help='Trading pair symbol (e.g., ZECUSDT)')
    args = parser.parse_args()
    
    main(args.symbol.upper())
