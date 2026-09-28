"""Generate REFERENCE-like datasets for MiniZinc (DZN) and ML (JSON)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from erlang_utils import generate_erlang_table
from reference_topology import (
    DEFAULT_CENTRAL,
    DEFAULT_EDGE,
    CITY_COORDS,
    ClusterSpec,
    Node,
    build_clusters,
    build_pops,
    distance_matrix,
)


@dataclass
class GeneratorConfig:
    seed: int = 42
    num_pops: int = 50
    central_names: List[str] = None
    edge_names: List[str] = None
    srs_vcpu: int = 1
    srs_ram_mb: int = 256
    srs_capacity_mbps: float = 512.0
    demand_bitrate_mbps: float = 25.0
    blockage_probability: float = 0.01
    traffic_per_demand: float = 1.0
    transmission_cost: int = 1
    migration_cost: int = 20
    unserved_penalty: int = 10000
    alpha_cost: float = 0.7
    beta_latency: float = 0.3
    latency_unit_ms: float = 0.1
    wan_ms_per_100km: float = 1.0
    wan_budget_ms: float = 30.0
    lambda_min: float = 60.0
    lambda_max: float = 2400.0
    mu_min: float = 1.0
    mu_max: float = 10.0
    poisson_cap: int = 8000


def poisson_sample(mean: float, rng: random.Random) -> int:
    if mean <= 0:
        return 0
    if mean < 30:
        l = math.exp(-mean)
        k = 0
        p = 1.0
        while p > l:
            k += 1
            p *= rng.random()
        return k - 1
    # Normal approximation for large means
    value = rng.gauss(mean, math.sqrt(mean))
    return max(0, int(round(value)))


def compute_latency_per_10km_units(cfg: GeneratorConfig) -> int:
    ms_per_10km = cfg.wan_ms_per_100km / 10.0
    return max(1, int(round(ms_per_10km / cfg.latency_unit_ms)))


def compute_max_latency_units(cfg: GeneratorConfig) -> int:
    return max(1, int(round(cfg.wan_budget_ms / cfg.latency_unit_ms)))


def build_cluster_resources(clusters: List[ClusterSpec], rng: random.Random) -> Tuple[List[List[int]], List[int]]:
    resources: List[List[int]] = []
    deployment_costs: List[int] = []
    for cluster in clusters:
        if cluster.kind == "central":
            vcpu = rng.randint(64, 128)
            ram_mb = rng.randint(131072, 262144)
            cost = rng.randint(10, 30)
        else:
            vcpu = rng.randint(8, 16)
            ram_mb = rng.randint(8192, 32768)
            cost = rng.randint(40, 80)
        resources.append([vcpu, ram_mb])
        deployment_costs.append(cost)
    return resources, deployment_costs


def expand_demands(pop_demands: List[int]) -> List[int]:
    demand_to_pop: List[int] = []
    for pop_idx, count in enumerate(pop_demands):
        demand_to_pop.extend([pop_idx] * count)
    return demand_to_pop


def write_dzn(path: str, data: Dict[str, object]) -> None:
    def fmt_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return f"{value:.1f}"
            return str(value)
        if isinstance(value, list):
            if not value:
                return "[]"
            if isinstance(value[0], list):
                rows = [" , ".join(fmt_value(v) for v in row) for row in value]
                return "[| " + " | ".join(rows) + " |]"
            return "[ " + ", ".join(fmt_value(v) for v in value) + " ]"
        if isinstance(value, set):
            return "{ " + ", ".join(str(v) for v in value) + " }"
        return str(value)

    lines: List[str] = []
    for key, value in data.items():
        lines.append(f"{key} = {fmt_value(value)};")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def generate_case(cfg: GeneratorConfig, case_id: int, erlang_table_full: List[int]) -> Dict[str, object]:
    rng = random.Random(cfg.seed + case_id)

    central = cfg.central_names or DEFAULT_CENTRAL
    edge = cfg.edge_names or DEFAULT_EDGE
    clusters = build_clusters(central, edge)
    # We use only cities that are NOT clusters as PoPs (30 additional cities)
    cluster_names = set(central + edge)
    pop_names = [name for name in CITY_COORDS.keys() if name not in cluster_names]
    # Respect requested num_pops if smaller than available
    if cfg.num_pops < len(pop_names):
        pop_names = rng.sample(pop_names, cfg.num_pops)
    pops = [Node(name, CITY_COORDS[name][0], CITY_COORDS[name][1]) for name in pop_names]

    lat_per_10km = compute_latency_per_10km_units(cfg)
    max_latency = compute_max_latency_units(cfg)

    resources, deployment_costs = build_cluster_resources(clusters, rng)

    pop_lambda: List[float] = []
    pop_mu: List[float] = []
    pop_demands: List[int] = []
    for _ in pops:
        lam = rng.uniform(cfg.lambda_min, cfg.lambda_max)
        mu = rng.uniform(cfg.mu_min, cfg.mu_max)
        offered = lam / mu
        n = poisson_sample(offered, rng)
        if cfg.poisson_cap:
            n = min(n, cfg.poisson_cap)
        pop_lambda.append(lam)
        pop_mu.append(mu)
        pop_demands.append(n)

    demand_to_pop = expand_demands(pop_demands)
    distances_pop = distance_matrix(pops, clusters)

    from reference_topology import haversine_km
    # Draw one cluster as Origin for a given case
    origin_cluster = rng.choice(clusters)
    origin_node = Node("Origin", origin_cluster.lat, origin_cluster.lon)
    
    origin_distances = []
    for c in clusters:
        dist = haversine_km(origin_node, Node(c.name, c.lat, c.lon))
        origin_distances.append(int(round(dist)))

    total_demands = sum(pop_demands)

    srs_capacity = int(cfg.srs_capacity_mbps / cfg.demand_bitrate_mbps)

    if len(erlang_table_full) < total_demands + 1:
        from erlang_utils import generate_erlang_table
        erlang_table_full = generate_erlang_table(
            total_demands + 1,
            cfg.traffic_per_demand,
            cfg.blockage_probability,
            srs_capacity=srs_capacity
        )
    srs_resources = [cfg.srs_vcpu, cfg.srs_ram_mb]

    data = {
        "ResourceTypes": {"vCPU", "RAM"},
        "SRSCapacity": srs_capacity,
        "MaxLatency": max_latency,
        "TransmissionCost": cfg.transmission_cost,
        "LatencyPer10km": lat_per_10km,
        "MigrationCost": cfg.migration_cost,
        "UnservedPenalty": cfg.unserved_penalty,
        "Clusters_": len(clusters),
        "Pops_": len(pops),
        "TotalDemands": total_demands,
        "PopDemands": pop_demands,
        "OriginDistances": origin_distances,
        "Resources": resources,
        "SRSResources": srs_resources,
        "Distances": distances_pop,
        "InitialPlacement": [[0] * len(clusters) for _ in range(len(pops))],
        "DeploymentCost": deployment_costs,
        "Alpha_Cost": cfg.alpha_cost,
        "Beta_Latency": cfg.beta_latency,
    }

    erlang_table = erlang_table_full[: total_demands + 1]

    meta = {
        "caseId": case_id,
        "config": asdict(cfg),
        "clusters": [asdict(c) for c in clusters],
        "clusterResources": resources,
        "clusterDeploymentCosts": deployment_costs,
        "pops": [asdict(p) for p in pops],
        "popLambda": pop_lambda,
        "popMu": pop_mu,
        "popDemands": pop_demands,
        "demandToPop": demand_to_pop,
        "popDistances": distances_pop,
        "originDistances": origin_distances,
        "originNode": {"name": origin_cluster.name, "lat": origin_cluster.lat, "lon": origin_cluster.lon},
        "latencyUnitMs": cfg.latency_unit_ms,
        "wanMsPer100km": cfg.wan_ms_per_100km,
        "wanBudgetMs": cfg.wan_budget_ms,
        "latencyPer10kmUnits": lat_per_10km,
        "maxLatencyUnits": max_latency,
        "srsCapacity": srs_capacity,
        "srsResources": srs_resources,
        "erlang": {
            "blockageProbability": cfg.blockage_probability,
            "trafficPerDemand": cfg.traffic_per_demand,
        },
    }

    return {
        "dzn_generic": data,
        "dzn_erlang": {
            **data,
            "BlockageProbability": cfg.blockage_probability,
            "TrafficPerDemand": cfg.traffic_per_demand,
            "ErlangTable": erlang_table,
        },
        "json": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="minizinc/generated")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start-id", type=int, default=0, help="Starting case ID offset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pops", type=int, default=50, help="Number of PoPs")
    parser.add_argument("--max-demands-estimate", type=int, default=10000,
                        help="Upper bound for pre-computed Erlang table (default: 10000)")
    parser.add_argument("--lambda-min", type=float, default=60.0)
    parser.add_argument("--lambda-max", type=float, default=2400.0)
    args = parser.parse_args()

    cfg = GeneratorConfig(seed=args.seed, num_pops=args.num_pops, lambda_min=args.lambda_min, lambda_max=args.lambda_max)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Pre-computing Erlang table (max_demands={args.max_demands_estimate})...")
    srs_capacity = int(cfg.srs_capacity_mbps / cfg.demand_bitrate_mbps)
    erlang_table_full = generate_erlang_table(
        max_demands=args.max_demands_estimate,
        traffic_per_demand=cfg.traffic_per_demand,
        target_blocking=cfg.blockage_probability,
        srs_capacity=srs_capacity
    )
    print(f"  Erlang table size: {len(erlang_table_full)} entries")

    for i in range(args.count):
        actual_id = args.start_id + i
        case = generate_case(cfg, actual_id, erlang_table_full)
        base = os.path.join(args.output_dir, f"reference_case_{actual_id+1:07d}")
        write_dzn(base + "_generic.dzn", case["dzn_generic"])
        write_dzn(base + "_erlang.dzn", case["dzn_erlang"])
        with open(base + ".json", "w", encoding="utf-8") as handle:
            json.dump(case["json"], handle, indent=2)
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{args.count} cases...")
    print(f"Done: {args.count} cases in {args.output_dir}")


if __name__ == "__main__":
    main()
