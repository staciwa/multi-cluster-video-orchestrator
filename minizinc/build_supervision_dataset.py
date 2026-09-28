"""Build supervision dataset by solving REFERENCE cases with MiniZinc."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from minizinc import Model, Solver, Instance
from enum import Enum

from erlang_utils import generate_erlang_table


@dataclass
class BuildResult:
    status: str
    solve_time_s: Optional[float]
    labels: Optional[Dict[str, Any]]
    error: Optional[str]


def find_cases(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("reference_case_*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_instance_data(meta: Dict[str, Any]) -> Dict[str, Any]:
    demand_to_pop = meta["demandToPop"]
    pop_distances = meta["popDistances"]
    distances = [pop_distances[pop_idx] for pop_idx in demand_to_pop]

    config = meta["config"]
    clusters = meta["clusters"]
    data = {
        "Clusters_": len(clusters),
        "Demands_": len(demand_to_pop),
        "Resources": meta["clusterResources"],
        "DeploymentCost": meta["clusterDeploymentCosts"],
        "SRSResources": meta["srsResources"],
        "Distances": distances,
        "InitialPlacement": [[0] * len(clusters) for _ in range(len(demand_to_pop))],
        "MaxLatency": meta["maxLatencyUnits"],
        "LatencyPer10km": meta["latencyPer10kmUnits"],
        "TransmissionCost": config["transmission_cost"],
        "MigrationCost": config["migration_cost"],
        "UnservedPenalty": config["unserved_penalty"],
        "Alpha_Cost": config["alpha_cost"],
        "Beta_Latency": config["beta_latency"],
    }

    erlang = meta["erlang"]
    erlang_table = generate_erlang_table(
        max_demands=data["Demands_"],
        traffic_per_demand=erlang["trafficPerDemand"],
        target_blocking=erlang["blockageProbability"],
    )

    data["BlockageProbability"] = erlang["blockageProbability"]
    data["ErlangTable"] = erlang_table
    data["SRSCapacity"] = meta["srsCapacity"]
    data["OriginDistances"] = meta["originDistances"]
    return data


def solve_case(model_path: Path, meta: Dict[str, Any], timeout_s: int, solver_name: str) -> BuildResult:
    try:
        solver = Solver.lookup(solver_name)
        model = Model([str(model_path)])
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
        instance["Demands_"] = data["Demands_"]
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
        result = instance.solve(timeout=timedelta(seconds=timeout_s))
        if result.solution is None:
            return BuildResult(status="unsat", solve_time_s=None, labels=None, error=None)

        solve_time = result.statistics.get("solveTime", timedelta(0))
        if isinstance(solve_time, timedelta):
            solve_time_s = solve_time.total_seconds()
        else:
            solve_time_s = float(solve_time)

        output_str = str(result.solution)
        start = output_str.find("{")
        end = output_str.rfind("}") + 1
        if start == -1 or end <= 0:
            return BuildResult(status="error", solve_time_s=solve_time_s, labels=None, error="cannot_parse_output")

        labels = json.loads(output_str[start:end])
        return BuildResult(status="ok", solve_time_s=solve_time_s, labels=labels, error=None)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return BuildResult(status="error", solve_time_s=None, labels=None, error=str(exc))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    timeout_s: int,
    solver_name: str,
    limit: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(input_dir)
    if limit is not None:
        cases = cases[:limit]

    features_labels: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for json_path in cases:
        case_id = json_path.stem.replace("reference_case_", "")
        dzn_path = json_path.with_name(f"reference_case_{case_id}_erlang.dzn")
        if not dzn_path.exists():
            failed.append({
                "caseId": case_id,
                "status": "missing_dzn",
                "jsonPath": str(json_path),
                "dznPath": str(dzn_path),
            })
            continue

        meta = load_json(json_path)
        result = solve_case(model_path, meta, timeout_s, solver_name)
        if result.status != "ok" or result.labels is None:
            failed.append({
                "caseId": case_id,
                "status": result.status,
                "error": result.error,
                "jsonPath": str(json_path),
                "dznPath": str(dzn_path),
            })
            continue

        features_labels.append({
            "caseId": case_id,
            "input": {
                "meta": meta,
                "dznPath": str(dzn_path),
            },
            "labels": result.labels,
            "solveTimeSec": result.solve_time_s,
            "status": result.status,
        })

    write_jsonl(output_dir / "features_labels.jsonl", features_labels)
    write_jsonl(output_dir / "failed_cases.jsonl", failed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="minizinc/generated")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--model", default="minizinc/model_erlang.mzn")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    build_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_path=Path(args.model),
        timeout_s=args.timeout,
        solver_name=args.solver,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
