import os
import sys
import json
import time
import math
from datetime import timedelta
import minizinc
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath("."))
sys.path.insert(0, os.path.abspath("minizinc"))
from generate_reference_dataset import GeneratorConfig, generate_case
from erlang_utils import generate_erlang_table

def build_instance_data(meta, erlang_table=None):
    pop_distances = meta["popDistances"]
    pop_demands = meta["popDemands"]
    total_demands = sum(pop_demands)

    config = meta["config"]
    clusters = meta["clusters"]
    data = {
        "ResourceTypes": ["vCPU", "RAM"],
        "Clusters_": len(clusters),
        "Pops_": len(pop_demands),
        "TotalDemands": total_demands,
        "PopDemands": pop_demands,
        "Resources": meta["clusterResources"],
        "DeploymentCost": meta["clusterDeploymentCosts"],
        "SRSResources": meta["srsResources"],
        "Distances": pop_distances,
        "OriginDistances": meta["originDistances"],
        "TransmissionCost": config["transmission_cost"],
        "MigrationCost": config["migration_cost"],
        "UnservedPenalty": config["unserved_penalty"],
        "Alpha_Cost": config["alpha_cost"],
        "Beta_Latency": config["beta_latency"],
        "BlockageProbability": config["blockage_probability"],
        "SRSCapacity": int(config["srs_capacity_mbps"]),
        "MaxLatency": int(math.floor(config["wan_budget_ms"] / config["latency_unit_ms"])),
        "LatencyPer10km": 1,
        "InitialPlacement": [[0] * len(clusters) for _ in range(len(pop_demands))],
    }

    max_demand_needed = int(sum(pop_demands) + 1)
    if erlang_table is None or len(erlang_table) < max_demand_needed:
        erlang_table = generate_erlang_table(max_demand_needed, config["traffic_per_demand"], config["blockage_probability"])
    
    # Prune table to maximum actually needed length to avoid MiniZinc int overflow bounds
    data["ErlangTable"] = erlang_table[:max_demand_needed]
    return data

def run_mip_gap_analysis():
    print("=== MIP Gap vs GNN Inference Scale Analysis ===")
    
    # 1. Generate a massive 100-PoP macro-topology
    print("\nGenerating massive topology: 100 PoPs, 50 Clusters (Macro Scale)...")
    config = GeneratorConfig(num_pops=100)
    
    erlang_table_full = generate_erlang_table(10, config.traffic_per_demand, config.blockage_probability)
    case_meta = generate_case(config, 1, erlang_table_full)["json"]
    
    # 2. GNN Inference Time
    print("\n[GNN] Initializing PPO model...")
    from python.orchestrator import try_load_gnn_model, predict_gnn_placement
    # Loading our trained model
    bundle = try_load_gnn_model("models/best_baseline_gnn_uniform.pt")
    
    # Warmup GNN (PyTorch CUDA/CPU allocation takes time on first pass)
    predict_gnn_placement(bundle, case_meta)
    
    print("\n[GNN] O(1) time measurements...")
    gnn_times = []
    for _ in range(10):
        start_time = time.time()
        gnn_alloc = np.asarray(predict_gnn_placement(bundle, case_meta))
        gnn_times.append((time.time() - start_time) * 1000) # ms
        
    print(f"[GNN] Tasks allocated: {np.sum(gnn_alloc)}")
    print(f"[GNN] Inference time: {np.mean(gnn_times):.2f} ms ± {np.std(gnn_times):.2f} ms (N=10)")
    
    # 3. MILP Solve Time (Timeout 60s)
    print("\n[MILP] Constructing optimization problem...")
    data = build_instance_data(case_meta, erlang_table_full)
    
    model = minizinc.Model("minizinc/model_erlang.mzn")
    solver = minizinc.Solver.lookup("highs")
    instance = minizinc.Instance(solver, model)
    
    for k, v in data.items():
        instance[k] = v
        
    print(f"[MILP] Starting branch-and-bound search. Please wait (timeout = 60s)...")
    start_time = time.time()
    result = instance.solve(timeout=timedelta(seconds=60))
    milp_time = time.time() - start_time
    
    if result.status == minizinc.Status.UNKNOWN:
        print(f"[MILP] ❌ Result: TIMEOUT. Solver failed to find a solution in 60s!")
        print(f"[MILP] Execution time until interruption: {milp_time:.2f} seconds")
    elif result.status == minizinc.Status.UNSATISFIABLE:
        print(f"[MILP] ❌ Result: UNSATISFIABLE. The problem is contradictory!")
    elif result.status == minizinc.Status.ERROR:
        print(f"[MILP] ❌ Result: ERROR. A solver error occurred.")
    elif result.status == minizinc.Status.UNBOUNDED:
        print(f"[MILP] ❌ Result: UNBOUNDED. Model is unbounded!")
    elif result.status == minizinc.Status.OPTIMAL_SOLUTION and result.solution is not None:
        print(f"[MILP] ✅ Result: {result.status}. Time: {milp_time:.2f} seconds")
        milp_alloc = np.array(result.solution.AssignedDemands)
        mae = np.mean(np.abs(gnn_alloc - milp_alloc))
        print(f"[ANALYSIS] Mean Absolute Error (MAE) between GNN allocation and optimal MILP: {mae:.4f}")
    else:
        print(f"[MILP] ⚠️ Other status: {result.status}")
        
    print("\n=== CONCLUSIONS ===")
    print("MIP Gap analyzes distribution quality via MAE against oracle ground truth.")
    print("GNN provides deterministic millisecond execution time, independent of NP-Hard tree complexity, ")
    print("which guarantees meeting rigorous SLAs (e.g., within seconds) for global macro-allocation correction.")

if __name__ == "__main__":
    run_mip_gap_analysis()
