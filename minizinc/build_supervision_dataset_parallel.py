"""Build supervision dataset by solving REFERENCE cases with MiniZinc — parallel version."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from erlang_utils import generate_erlang_table

# Cache key: (traffic_per_demand, blockage_probability)
# Each worker pre-generates a large table once and slices per case.
ERLANG_TABLE_MAX = 10000
_erlang_cache: Dict[tuple, List[int]] = {}


def get_erlang_table_cached(max_demands: int, traffic: float, blocking: float) -> List[int]:
    key = (traffic, blocking)
    if key not in _erlang_cache:
        _erlang_cache[key] = generate_erlang_table(ERLANG_TABLE_MAX, traffic, blocking)
    full = _erlang_cache[key]
    if max_demands >= len(full):
        full = generate_erlang_table(max_demands, traffic, blocking)
        _erlang_cache[key] = full
    return full[: max_demands + 1]


def find_cases(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("reference_case_*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_instance_data(meta: Dict[str, Any]) -> Dict[str, Any]:
    pop_distances = meta["popDistances"]
    pop_demands = meta["popDemands"]
    total_demands = sum(pop_demands)

    config = meta["config"]
    clusters = meta["clusters"]
    data = {
        "Clusters_": len(clusters),
        "Pops_": len(pop_demands),
        "TotalDemands": total_demands,
        "PopDemands": pop_demands,
        "Resources": meta["clusterResources"],
        "DeploymentCost": meta["clusterDeploymentCosts"],
        "SRSResources": meta["srsResources"],
        "Distances": pop_distances,
        "InitialPlacement": [[0] * len(clusters) for _ in range(len(pop_demands))],
        "MaxLatency": meta["maxLatencyUnits"],
        "LatencyPer10km": meta["latencyPer10kmUnits"],
        "TransmissionCost": config["transmission_cost"],
        "MigrationCost": config["migration_cost"],
        "UnservedPenalty": config["unserved_penalty"],
        "Alpha_Cost": config["alpha_cost"],
        "Beta_Latency": config["beta_latency"],
    }

    erlang = meta["erlang"]
    erlang_table = get_erlang_table_cached(
        max_demands=total_demands,
        traffic=erlang["trafficPerDemand"],
        blocking=erlang["blockageProbability"],
    )

    data["BlockageProbability"] = erlang["blockageProbability"]
    data["ErlangTable"] = erlang_table
    data["SRSCapacity"] = meta["srsCapacity"]
    data["OriginDistances"] = meta["originDistances"]
    return data


def solve_case(model_path: str, meta: Dict[str, Any], timeout_s: int, solver_name: str) -> Dict[str, Any]:
    """Solve a single case. Designed to run in a subprocess."""
    from minizinc import Model, Solver, Instance

    try:
        solver = Solver.lookup(solver_name)
        model = Model([model_path])
        instance = Instance(solver, model)
        data = build_instance_data(meta)
        instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
        instance["SRSResources"] = data["SRSResources"]
        instance["MaxLatency"] = data["MaxLatency"]
        instance["TransmissionCost"] = data["TransmissionCost"]
        instance["LatencyPer10km"] = data["LatencyPer10km"]
        instance["MigrationCost"] = data["MigrationCost"]
        instance["UnservedPenalty"] = data["UnservedPenalty"]
        instance["Clusters_"] = data["Clusters_"]
        instance["Pops_"] = data["Pops_"]
        instance["TotalDemands"] = data["TotalDemands"]
        instance["PopDemands"] = data["PopDemands"]
        instance["Resources"] = data["Resources"]
        instance["Distances"] = data["Distances"]
        instance["InitialPlacement"] = data["InitialPlacement"]
        instance["DeploymentCost"] = data["DeploymentCost"]
        instance["Alpha_Cost"] = data["Alpha_Cost"]
        instance["Beta_Latency"] = data["Beta_Latency"]
        instance["ErlangTable"] = data["ErlangTable"]
        instance["BlockageProbability"] = data["BlockageProbability"]
        instance["SRSCapacity"] = data["SRSCapacity"]
        instance["OriginDistances"] = data["OriginDistances"]

        if timeout_s > 0:
            result = instance.solve(timeout=timedelta(seconds=timeout_s))
        else:
            result = instance.solve()
        if result.solution is None:
            return {"status": "unsat", "solve_time_s": None, "labels": None, "error": None}

        solve_time = result.statistics.get("solveTime", timedelta(0))
        if isinstance(solve_time, timedelta):
            solve_time_s = solve_time.total_seconds()
        else:
            solve_time_s = float(solve_time)

        sol = result.solution
        labels = {
            "instances": getattr(sol, "Instances", []),
            "demandsPerCluster": getattr(sol, "DemandsPerCluster", []),
            "unserved": getattr(sol, "UnservedDemands", 0),
            "migrations": getattr(sol, "Migrations", 0),
            "totalCost": getattr(sol, "TotalCost", 0),
            "totalLatency": getattr(sol, "TotalLatency", 0),
            "placement": getattr(sol, "AssignedDemands", []),
            "usedResources": getattr(sol, "UsedResources", []),
            "scores": {
                "globalScore": getattr(sol, "GlobalScore", 0),
                "normalizedCost": getattr(sol, "NormCost", 0),
                "normalizedLatency": getattr(sol, "NormLatency", 0),
            },
            "costBreakdown": {
                "deployment": getattr(sol, "TotalDeploymentCost", 0),
                "transmission": getattr(sol, "TotalTransmissionCost", 0),
                "migration": getattr(sol, "TotalMigrationCost", 0),
                "unservedPenalty": getattr(sol, "TotalPenalty", 0),
            }
        }
        return {"status": "ok", "solve_time_s": solve_time_s, "labels": labels, "error": None}
    except Exception as exc:
        return {"status": "error", "solve_time_s": None, "labels": None, "error": str(exc)}


def solve_single(args: tuple) -> Dict[str, Any]:
    """Wrapper for ProcessPoolExecutor — unpacks (model_path, json_path, timeout_s, solver_name)."""
    model_path, json_path, timeout_s, solver_name = args
    try:
        meta = load_json(json_path)
        result = solve_case(model_path, meta, timeout_s, solver_name)
        result["caseId"] = json_path.stem.replace("reference_case_", "")
        result["jsonPath"] = str(json_path)
        return result
    except Exception as exc:
        return {
            "caseId": json_path.stem.replace("reference_case_", ""),
            "status": "error",
            "solve_time_s": None,
            "labels": None,
            "error": str(exc),
            "jsonPath": str(json_path),
        }


def load_existing_results(output_dir: Path) -> set:
    """Load already-solved caseIds from existing features_labels.jsonl and failed_cases.jsonl."""
    solved = set()
    fl = output_dir / "features_labels.jsonl"
    if fl.exists():
        with fl.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        solved.add(row["caseId"])
                    except json.JSONDecodeError:
                        pass
    fc = output_dir / "failed_cases.jsonl"
    if fc.exists():
        with fc.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if row.get("status") == "unsat":
                            solved.add(row["caseId"])
                    except json.JSONDecodeError:
                        pass
    return solved


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_dataset_parallel(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    timeout_s: int,
    solver_name: str,
    workers: int,
    chunk_size: int,
    limit: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(input_dir)
    if not cases:
        print(f"No cases found in {input_dir}")
        return

    already_solved = load_existing_results(output_dir)
    pending = [c for c in cases if c.stem.replace("reference_case_", "") not in already_solved]
    if limit is not None:
        pending = pending[:limit]
    print(f"Total cases: {len(cases)}, already solved: {len(already_solved)}, pending: {len(pending)}")

    if not pending:
        print("All cases already solved. Nothing to do.")
        return

    model_path_str = str(model_path.resolve())
    tasks = [(model_path_str, json_path, timeout_s, solver_name) for json_path in pending]

    features_labels_path = output_dir / "features_labels.jsonl"
    failed_cases_path = output_dir / "failed_cases.jsonl"

    total_ok = 0
    total_fail = 0
    total_unsat = 0
    total_time = 0.0
    start_time = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(solve_single, task): task for task in tasks}

            chunk_results_ok: List[Dict[str, Any]] = []
            chunk_results_fail: List[Dict[str, Any]] = []

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                case_id = result["caseId"]

                if result["status"] == "ok" and result["labels"] is not None:
                    total_ok += 1
                    if result["solve_time_s"] is not None:
                        total_time += result["solve_time_s"]
                    chunk_results_ok.append({
                        "caseId": case_id,
                        "input": {
                            "meta": load_json(Path(result["jsonPath"])),
                            "dznPath": result["jsonPath"],
                        },
                        "labels": result["labels"],
                        "solveTimeSec": result["solve_time_s"],
                        "status": result["status"],
                    })
                elif result["status"] == "unsat":
                    total_unsat += 1
                    chunk_results_fail.append({
                        "caseId": case_id,
                        "status": "unsat",
                        "jsonPath": result["jsonPath"],
                    })
                else:
                    total_fail += 1
                    chunk_results_fail.append({
                        "caseId": case_id,
                        "status": result["status"],
                        "error": result["error"],
                        "jsonPath": result["jsonPath"],
                    })

                if i % 10 == 0 or i == len(pending):
                    if chunk_results_ok:
                        append_jsonl(features_labels_path, chunk_results_ok)
                    if chunk_results_fail:
                        append_jsonl(failed_cases_path, chunk_results_fail)
                    chunk_results_ok.clear()
                    chunk_results_fail.clear()

                elapsed = time.time() - start_time
                avg_time = total_time / total_ok if total_ok > 0 else 0
                eta = (len(pending) - i) * avg_time / workers if avg_time > 0 else 0
                print(
                    f"\r  [{i}/{len(pending)}] ok={total_ok} unsat={total_unsat} err={total_fail} "
                    f"avg={avg_time:.2f}s eta={eta:.0f}s",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] Execution interrupted. Cleaning up orphaned HiGHS/MiniZinc processes...")
        import subprocess
        subprocess.run(["pkill", "-9", "-f", "highs"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "minizinc"], stderr=subprocess.DEVNULL)
        raise

    total_elapsed = time.time() - start_time
    print(f"\nDone in {total_elapsed:.0f}s")
    print(f"  OK: {total_ok}, UNSAT: {total_unsat}, Errors: {total_fail}")
    if total_ok > 0:
        print(f"  Avg solve time: {total_time / total_ok:.2f}s")

    stats = {
        "total_cases": len(cases),
        "already_solved": len(already_solved),
        "new_ok": total_ok,
        "new_unsat": total_unsat,
        "new_errors": total_fail,
        "total_ok": total_ok + len(already_solved),
        "total_elapsed_s": round(total_elapsed, 1),
        "avg_solve_time_s": round(total_time / total_ok, 2) if total_ok > 0 else 0,
    }
    stats_path = output_dir / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats written to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="minizinc/generated")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--model", default="minizinc/model_erlang.mzn")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0 = no limit)")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--workers", type=int, default=7, help="Number of parallel workers (default: 7)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Flush results every N cases")
    parser.add_argument("--limit", type=int, help="Limit number of cases to process")
    args = parser.parse_args()

    build_dataset_parallel(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_path=Path(args.model),
        timeout_s=args.timeout,
        solver_name=args.solver,
        workers=args.workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
