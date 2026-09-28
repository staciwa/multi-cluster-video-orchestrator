#!/usr/bin/env python3
"""Monitor batch solving progress."""

import json
import time
from pathlib import Path

CHECKPOINT = Path("dataset_10k/batch_checkpoint.json")
STATS = Path("dataset_10k/stats.json")
JSONL = Path("dataset_10k/features_labels.jsonl")

def main():
    print("Monitoring batch solving progress...\n")
    last_count = 0
    while True:
        if CHECKPOINT.exists():
            with open(CHECKPOINT) as f:
                ckpt = json.load(f)
            solved = ckpt["solved"]
            total = ckpt["total"]
            pct = solved / total * 100
            eta_min = ckpt.get("eta_s", 0) / 60

            jsonl_lines = 0
            if JSONL.exists():
                with open(JSONL) as f:
                    jsonl_lines = sum(1 for _ in f)

            speed = solved - last_count
            last_count = solved

            print(f"\r  {solved}/{total} ({pct:.1f}%) | JSONL: {jsonl_lines} | ETA: {eta_min:.0f} min | +{speed}/10s", end="", flush=True)

            if solved >= total:
                print(f"\n\nAll {total} cases solved!")
                if STATS.exists():
                    with open(STATS) as f:
                        stats = json.load(f)
                    print(f"\nStats:")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                break
        else:
            print("  No checkpoint yet...")

        time.sleep(10)

if __name__ == "__main__":
    main()
