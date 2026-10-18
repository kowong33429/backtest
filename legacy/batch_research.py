"""
batch_research.py — Run Quant Research Agent on Multiple Coins Automatically
"""
import os
import sys
import json
import argparse
import subprocess

def get_already_downloaded(base_dir):
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        return []
    downloaded = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, f"{folder.upper()}_4h_full.csv")
            if os.path.exists(csv_file):
                downloaded.append(folder.upper() + 'USDT' if not folder.upper().endswith('USDT') else folder.upper())
    return sorted(downloaded)

def main():
    parser = argparse.ArgumentParser(description='Batch Run Quant Research Agent')
    parser.add_argument('--priority', action='store_true', help='Run only on priority coins')
    args = parser.parse_args()

    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(core_dir)
    agent_script = os.path.join(core_dir, 'research_agent.py')

    # Determine symbols
    if args.priority:
        priority_file = os.path.join(base_dir, 'data', 'scan_results', 'scanned_priority_coins.json')
        if os.path.exists(priority_file):
            with open(priority_file, 'r', encoding='utf-8') as f:
                symbols = json.load(f)
            print(f"Loaded {len(symbols)} priority coins from trend scanner.")
        else:
            print("Priority file not found! Please run 'python core/trend_scanner.py' first.")
            return
    else:
        symbols = get_already_downloaded(base_dir)
        print(f"Loaded {len(symbols)} already downloaded coins.")

    print(f"{'=' * 60}")
    print(f"  Batch Research: {len(symbols)} coins")
    print(f"  Note: Each coin will trigger LLM API calls. This may take time.")
    print(f"{'=' * 60}\n")

    success = 0
    failed = 0
    skipped = 0

    for i, sym in enumerate(symbols, 1):
        # Check if report already exists to avoid re-running expensive LLM tasks
        folder_name = sym.replace('USDT', '').lower() + 'usdt'
        report_path = os.path.join(base_dir, "data", folder_name, "summary_report.md")
        
        if os.path.exists(report_path):
            print(f"[{i}/{len(symbols)}] Skipping {sym} (Report already exists. Delete summary_report.md to re-run)")
            skipped += 1
            continue

        print(f"\n[{i}/{len(symbols)}] Running Research Agent for {sym}...")
        result = subprocess.run(
            [sys.executable, agent_script, '--symbol', sym]
        )
        if result.returncode == 0:
            success += 1
        else:
            print(f"  -> ERROR running {sym}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Batch Research complete! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
