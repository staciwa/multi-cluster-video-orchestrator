import os
import sys
import json
import time
import argparse
import torch
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from python.orchestrator import try_load_gnn_model, predict_gnn_placement

def compute_objective(meta, placement):
    """
    Calculates the simplified objective function value (cost) for a given placement.
    Includes: UnservedPenalty and TransmissionCost.
    We assume starting from zero (no migration costs at the start of the window).
    """
    if "popDemands" not in meta:
        raise KeyError("meta must contain 'popDemands'")
    if "popDistances" not in meta:
        raise KeyError("meta must contain 'popDistances'")
    if "clusters" not in meta:
        raise KeyError("meta must contain 'clusters'")
    if "config" not in meta:
        raise KeyError("meta must contain 'config'")

    pop_demands = meta.get("popDemands", [meta.get("demandToPop", []).count(i) for i in range(len(meta.get("pops", [])))])
    distances = meta["popDistances"]
    config = meta["config"]
    unserved_penalty = config["unserved_penalty"]
    transmission_cost = config["transmission_cost"]
    
    unserved_penalty_cost = 0
    transmission_cost_total = 0
    
    for p in range(len(pop_demands)):
        assigned = sum(placement[p])
        unserved = max(0, pop_demands[p] - assigned)
        unserved_penalty_cost += unserved * unserved_penalty
        
        for c in range(len(meta["clusters"])):
            transmission_cost_total += placement[p][c] * distances[p][c] * transmission_cost
            
    total_cost = unserved_penalty_cost + transmission_cost_total
    return total_cost, unserved_penalty_cost, transmission_cost_total

def evaluate_scalability(gnn_model_path: str, datasets: list[str], output_json: str):
    gnn_bundle = try_load_gnn_model(gnn_model_path)
    if not gnn_bundle:
        print("Failed to load GNN model.")
        return

    results = {}

    for dataset_name in datasets:
        print(f"\n--- Analyzing dataset: {dataset_name} ---")
        dataset_dir = Path(dataset_name)
        jsonl_path = dataset_dir / "features_labels.jsonl"
        
        if not jsonl_path.exists():
            print(f"Missing file {jsonl_path}. Skip.")
            continue
            
        milp_costs = []
        gnn_costs = []
        milp_times = []
        gnn_times = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                meta = data["input"]["meta"]
                
                # 1. MILP results (read from jsonl, previously generated with timeout=60)
                milp_placement = data["labels"]["placement"]
                milp_time = data["labels"].get("solve_time_s", 60.0)
                milp_cost, _, _ = compute_objective(meta, milp_placement)
                
                # 2. GNN results (on-the-fly inference)
                t0 = time.time()
                gnn_placement = predict_gnn_placement(gnn_bundle, meta)
                gnn_time = time.time() - t0
                gnn_cost, _, _ = compute_objective(meta, gnn_placement)
                
                milp_costs.append(milp_cost)
                gnn_costs.append(gnn_cost)
                milp_times.append(milp_time)
                gnn_times.append(gnn_time)
                
        if len(milp_costs) > 0:
            avg_milp_cost = sum(milp_costs) / len(milp_costs)
            avg_gnn_cost = sum(gnn_costs) / len(gnn_costs)
            avg_milp_time = sum(milp_times) / len(milp_times)
            avg_gnn_time = sum(gnn_times) / len(gnn_times)
            
            print(f"Average MILP solving time: {avg_milp_time:.3f} s")
            print(f"Average GNN inference time : {avg_gnn_time:.4f} s")
            print(f"Average MILP cost (60s limit): {avg_milp_cost:.1f}")
            print(f"Average GNN cost (O(1) time) : {avg_gnn_cost:.1f}")
            print(f"Objective function gain (GNN)   : {((avg_milp_cost - avg_gnn_cost) / max(1, avg_milp_cost)) * 100:.2f} %")
            
            results[dataset_name] = {
                "avg_milp_time_s": avg_milp_time,
                "avg_gnn_time_s": avg_gnn_time,
                "avg_milp_cost": avg_milp_cost,
                "avg_gnn_cost": avg_gnn_cost,
                "cost_improvement_percent": ((avg_milp_cost - avg_gnn_cost) / max(1, avg_milp_cost)) * 100
            }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nScalability results saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn_model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--datasets", nargs="+", default=["dataset_mip_gap_60", "dataset_mip_gap_120", "dataset_mip_gap_200"])
    parser.add_argument("--output", type=str, default="results/mip_gap_scalability.json")
    args = parser.parse_args()
    
    evaluate_scalability(args.gnn_model, args.datasets, args.output)
