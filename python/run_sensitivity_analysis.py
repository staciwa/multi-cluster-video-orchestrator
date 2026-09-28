import sys
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Fix python import path
sys.path.append(os.path.abspath("."))
import importlib.util
spec = importlib.util.spec_from_file_location("bsdp", "minizinc/build_supervision_dataset_parallel.py")
bsdp = importlib.util.module_from_spec(spec)
sys.modules["bsdp"] = bsdp
spec.loader.exec_module(bsdp)
solve_case = bsdp.solve_case
load_json = bsdp.load_json

def run_sensitivity():
    case_path = "minizinc/generated_nominal/reference_case_0000001.json"
    if not os.path.exists(case_path):
        import glob
        cases = glob.glob("minizinc/generated_*/reference_case_*.json")
        if not cases:
            cases = glob.glob("dataset_*/raw/reference_case_*.json")
        if cases:
            case_path = cases[0]
        else:
            return

    base_meta = load_json(Path(case_path))
    
    weights = [(round(i*0.02, 2), round(1.0 - i*0.02, 2)) for i in range(51)]
    
    results = []
    
    for alpha, beta in weights:
        meta = json.loads(json.dumps(base_meta))
        meta["config"]["alpha_cost"] = alpha
        meta["config"]["beta_latency"] = beta
        
        res = solve_case("minizinc/model_erlang.mzn", meta, timeout_s=30, solver_name="highs")
        if res["status"] == "ok":
            labels = res["labels"]
            results.append({
                "alpha": alpha,
                "beta": beta,
                "cost": labels["totalCost"],
                "latency": labels["totalLatency"]
            })
            print(f"Alpha={alpha}, Beta={beta} -> Cost: {labels['totalCost']}, Latency: {labels['totalLatency']}")

    plt.figure(figsize=(8, 5))
    costs = [r["cost"] for r in results]
    lats = [r["latency"] for r in results]
    
    plt.plot(lats, costs, marker='o', linestyle='-', color='b', markersize=4, alpha=0.6)
    
    plt.title("Sensitivity analysis of weights $\\alpha$ (Cost) and $\\beta$ (Latency) (N=51)")
    plt.xlabel("Total Latency")
    plt.ylabel("Total Cost")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_pdf = "latex/tex/img/wrazliwosc_pareto.pdf"
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot to {out_pdf} with {len(results)} points")

if __name__ == "__main__":
    run_sensitivity()
