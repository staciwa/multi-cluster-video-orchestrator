#!/usr/bin/env python3
"""Run MiniZinc solves in batches with resume support — single process."""

import argparse
import json
import sys
import time
from pathlib import Path

MAGISTERKA_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MAGISTERKA_ROOT / "minizinc"))

from build_supervision_dataset_parallel import (
    find_cases,
    load_existing_results,
    build_dataset_parallel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="minizinc/generated_10k")
    parser.add_argument("--output-dir", default="dataset_10k")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--model", default="minizinc/model_erlang.mzn")
    args = parser.parse_args()

    input_dir = MAGISTERKA_ROOT / args.input_dir
    output_dir = MAGISTERKA_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_cases = len(list(input_dir.glob("reference_case_*.json")))
    solved = load_existing_results(output_dir)
    solved_count = len(solved)
    remaining = total_cases - solved_count
    total_batches = (remaining + args.batch_size - 1) // args.batch_size

    print(f"Total cases: {total_cases}")
    print(f"Already solved: {solved_count}")
    print(f"Remaining: {remaining}")
    print(f"Batch size: {args.batch_size}, workers: {args.workers}")
    print(f"Batches to run: {total_batches}")
    print()

    batch_num = 0
    start_time = time.time()

    while True:
        current_solved = len(load_existing_results(output_dir))
        remaining_now = total_cases - current_solved
        if remaining_now <= 0:
            print("All cases solved!")
            break

        batch_num += 1
        this_batch = min(args.batch_size, remaining_now)

        print(f"\n{'='*60}")
        print(f" Batch {batch_num}: solving next {this_batch} cases")
        print(f"{'='*60}")

        build_dataset_parallel(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=MAGISTERKA_ROOT / args.model,
            timeout_s=args.timeout,
            solver_name=args.solver,
            workers=args.workers,
            chunk_size=this_batch,
            limit=this_batch,
        )

        elapsed = time.time() - start_time
        current_solved = len(load_existing_results(output_dir))
        rate = current_solved / elapsed if elapsed > 0 else 0
        eta = (total_cases - current_solved) / rate if rate > 0 else 0

        print(f"\nProgress: {current_solved}/{total_cases} ({current_solved/total_cases*100:.1f}%)")
        print(f"Rate: {rate:.1f} cases/s, ETA: {eta/60:.0f} min")

        checkpoint = output_dir / "batch_checkpoint.json"
        with checkpoint.open("w") as f:
            json.dump({
                "batch": batch_num,
                "solved": current_solved,
                "total": total_cases,
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
            }, f, indent=2)

    final_solved = len(load_existing_results(output_dir))
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" DONE: {final_solved}/{total_cases} cases in {total_elapsed/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
