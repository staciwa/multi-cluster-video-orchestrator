import os
import sys
import time
import numpy as np
from datetime import timedelta
import minizinc

sys.path.append(os.path.abspath("."))
sys.path.insert(0, os.path.abspath("minizinc"))
from generate_reference_dataset import GeneratorConfig, generate_case
from erlang_utils import generate_erlang_table
from python.orchestrator import try_load_gnn_model, predict_gnn_placement
from python.experiments.mip_gap_analysis import build_instance_data

def run_scalability_analysis():
    print("=== Scalability Analysis: GNN vs MILP (O(1) vs NP-Hard) ===")
    
    bundle = try_load_gnn_model("models/best_baseline_gnn.pt")
    model = minizinc.Model("minizinc/model_erlang.mzn")
    solver = minizinc.Solver.lookup("highs")
    
    # Scales to test: number of PoPs
    scales = [10, 30, 50, 80, 120, 200]
    
    results = {"scale": [], "gnn_time_ms": [], "milp_time_ms": []}
    
    for scale in scales:
        print(f"\n[Scale: {scale} PoPs]")
        config = GeneratorConfig(num_pops=scale)
        # Low demand to make solver task easier
        erlang_table_full = generate_erlang_table(20000, 10.0, 0.01)
        case_meta = generate_case(config, 42, erlang_table_full)["json"]
        
        # 1. Warmup GNN - hides PyTorch memory allocation overhead
        for _ in range(5):
            _ = predict_gnn_placement(bundle, case_meta)
            
        # 1. Measure actual GNN inference (100 repetitions)
        gnn_times = []
        for _ in range(100): 
            start_ns = time.perf_counter_ns()
            _ = predict_gnn_placement(bundle, case_meta)
            end_ns = time.perf_counter_ns()
            gnn_times.append((end_ns - start_ns) / 1e6)
            
        gnn_avg_ms = np.mean(gnn_times)
        print(f"GNN Inference: {gnn_avg_ms:.3f} ms")
        
        # 2. Measure MILP solving time
        data = build_instance_data(case_meta, erlang_table_full)
        instance = minizinc.Instance(solver, model)
        for k, v in data.items():
            instance[k] = v
            
        # Safeguard against infinite waiting for very large instances
        milp_timeout_sec = 10 if scale <= 50 else 30 
        start_ns = time.perf_counter_ns()
        result = instance.solve(timeout=timedelta(seconds=milp_timeout_sec))
        end_ns = time.perf_counter_ns()
        
        if result.status in [minizinc.Status.UNKNOWN, minizinc.Status.ERROR]:
            milp_time_ms = float(milp_timeout_sec * 1000)
            print(f"MILP Solve: Timeout/Error (> {milp_time_ms} ms)")
            results["milp_time_ms"].append(None) # Timeout marker for plot
        else:
            milp_time_ms = (end_ns - start_ns) / 1e6
            print(f"MILP Solve: {milp_time_ms:.3f} ms")
            results["milp_time_ms"].append(milp_time_ms)
            
        results["scale"].append(scale)
        results["gnn_time_ms"].append(gnn_avg_ms)
        
    # Export results for generating plot in LaTeX (pgfplots)
    import json
    with open("results/scalability_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nDone. Results saved in results/scalability_results.json")

if __name__ == "__main__":
    run_scalability_analysis()
