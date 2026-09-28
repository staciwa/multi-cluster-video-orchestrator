import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from minizinc import Model, Solver, Instance
from build_supervision_dataset_parallel import build_instance_data, load_json
from enum import Enum
import concurrent.futures
from datetime import timedelta

def solve_for_weights(alpha, beta, base_meta):
    solver = Solver.lookup("highs")
    model = Model(["model_erlang.mzn"])
    
    meta = json.loads(json.dumps(base_meta))
    meta["config"]["alpha_cost"] = alpha
    meta["config"]["beta_latency"] = beta
    
    instance = Instance(solver, model)
    data = build_instance_data(meta)
    instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
    for k, v in data.items():
        if k in ["SRSResources", "MaxLatency", "TransmissionCost", "LatencyPer10km", "MigrationCost", "UnservedPenalty", "Clusters_", "Pops_", "TotalDemands", "PopDemands", "Resources", "Distances", "InitialPlacement", "DeploymentCost", "Alpha_Cost", "Beta_Latency", "ErlangTable", "BlockageProbability", "SRSCapacity", "OriginDistances"]:
            instance[k] = v
            
    # Use 3 seconds timeout to make N=100 feasible
    res = instance.solve(timeout=timedelta(seconds=3))
    
    if res.solution is not None:
        try:
            labels = json.loads(res.solution._output_item)
            return {
                "alpha": alpha,
                "beta": beta,
                "cost": labels["totalCost"],
                "latency": labels["totalLatency"],
                "norm_cost": labels["scores"]["normalizedCost"],
                "norm_latency": labels["scores"]["normalizedLatency"]
            }
        except Exception:
            return None
    return None

def run_dense():
    case_path = "generated_test/reference_case_006.json"
    if not os.path.exists(case_path):
        print(f"Error: {case_path} not found")
        return

    base_meta = load_json(Path(case_path))
    
    # Generate 100 points
    alphas = np.linspace(0.0, 1.0, 100)
    
    results = []
    print(f"Running dense Pareto front with N=100 points, timeout=3s per point...")
    
    # We can parallelize this slightly to make it even faster, e.g. 4 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        for a in alphas:
            futures.append(executor.submit(solve_for_weights, float(a), float(1.0 - a), base_meta))
            
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            print(f"Completed {i+1}/100")
            
    # Sort by alpha for clean plotting
    results.sort(key=lambda x: x["alpha"])
    
    plt.figure(figsize=(9, 6))
    costs = [r["norm_cost"] for r in results]
    lats = [r["norm_latency"] for r in results]
    
    plt.scatter(lats, costs, c=[r["alpha"] for r in results], cmap='coolwarm', alpha=0.8, edgecolors='k')
    plt.colorbar(label='Waga $\\alpha$ (Koszt)')
    
    plt.title("Densified Pareto Front (N=100) for weights $\\alpha$ and $\\beta$")
    plt.xlabel("Normalized Latency (Normalized Latency)")
    plt.ylabel("Znormalizowany Koszt (Normalized Cost)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_pdf = "../latex/tex/img/wrazliwosc_pareto_gesta.pdf"
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved dense plot to {out_pdf}")

if __name__ == "__main__":
    run_dense()
