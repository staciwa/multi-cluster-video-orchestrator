import os
import argparse
import json
import numpy as np
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluation of the Greedy Heuristic (Greedy Latency)")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=300)
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        sys.exit(1)

    maes = []
    print(f"Starting evaluation of the Greedy Heuristic (Greedy Latency, N={args.seeds})...")
    
    for i in range(args.seeds):
        current_seed = args.base_seed + i
        np.random.seed(current_seed)
        
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0)
        try:
            obs, info = env.reset(seed=current_seed)
            done = False
            
            while not done:
                # In the greedy heuristic (Greedy Latency), we ignore advanced masking (Erlang capacity).
                # We select purely based on the lowest latency (ping), as long as it meets QoS (maxLatencyUnits).
                if env.current_arrival_event is None:
                    break
                    
                pop_id = env.current_arrival_event.pop_id
                best_cluster = env.num_clusters # default rejection (if none meet QoS)
                min_latency = float('inf')
                
                for c_id in range(env.num_clusters):
                    dist = env.state.pop_distances[pop_id, c_id] + env.state.origin_distances[c_id]
                    latency = dist * env.state.latency_per_10km_units
                    if latency <= env.state.max_latency_units:
                        if latency < min_latency:
                            min_latency = latency
                            best_cluster = c_id
                            
                action = best_cluster
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            edge_load = env.edge_load
            anchor_alloc = env.state.anchor_allocation
            
            mae = float(np.mean(np.abs(edge_load - anchor_alloc)))
            maes.append(mae)
            print(f"Seed {current_seed}: MAE = {mae:.4f}")
        finally:
            env.close()

    mean_mae = np.mean(maes)
    print("\n" + "="*50)
    print(" GREEDY HEURISTIC EVALUATION SUMMARY")
    print("="*50)
    print(f"Number of samples (N)      : {args.seeds}")
    print(f"Mean MAE error             : {mean_mae:.6f}")
    print("="*50)

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "greedy_heuristic_results.json")
    
    output_data = {
        "N": args.seeds,
        "mean_mae_greedy": float(mean_mae),
        "raw_maes": maes
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
