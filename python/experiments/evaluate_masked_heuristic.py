import os
import argparse
import json
import numpy as np
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluation of the Masked Heuristic (If-Else Mask)")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=300)
    parser.add_argument("--heuristic", type=str, choices=["random", "greedy_latency"], default="greedy_latency")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        sys.exit(1)

    maes = []
    uncond_maes = []
    drop_rates = []
    print(f"Starting evaluation of the {args.heuristic} heuristic using PPO Mask (N={args.seeds})...")
    
    for i in range(args.seeds):
        current_seed = args.base_seed + i
        np.random.seed(current_seed)
        
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0)
        try:
            obs, info = env.reset(seed=current_seed)
            done = False
            
            while not done:
                if env.current_arrival_event is None:
                    break
                
                mask = env.valid_action_mask()
                valid_actions = [a for a, m in enumerate(mask) if m == 1]
                
                if len(valid_actions) == 0:
                    action = env.num_clusters 
                elif len(valid_actions) == 1:
                    action = valid_actions[0]
                else:
                    if args.heuristic == "random":
                        action = int(np.random.choice(valid_actions))
                    elif args.heuristic == "greedy_latency":
                        pop_id = env.current_arrival_event.pop_id
                        best_cluster = valid_actions[0]
                        min_latency = float('inf')
                        for c_id in valid_actions:
                            if c_id == env.num_clusters: 
                                continue
                            dist = env.state.pop_distances[pop_id, c_id] + env.state.origin_distances[c_id]
                            latency = dist * env.state.latency_per_10km_units
                            if latency < min_latency:
                                min_latency = latency
                                best_cluster = c_id
                        action = best_cluster
                        
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            edge_load = env.edge_load
            anchor_alloc = env.state.anchor_allocation
            
            mae = float(np.mean(np.abs(edge_load - anchor_alloc)))
            
            ideal_unserved = env.current_case["labels"]["unserved"]
            actual_unserved = env.state.total_unserved
            
            unserved_diff = abs(actual_unserved - ideal_unserved)
            total_elements = env.num_pops * env.num_clusters
            uncond_mae = (np.sum(np.abs(edge_load - anchor_alloc)) + unserved_diff) / total_elements
            
            total_demands = sum(env.base_pop_demands)
            drop_rate = actual_unserved / max(1, total_demands)
            
            maes.append(mae)
            uncond_maes.append(uncond_mae)
            drop_rates.append(drop_rate)
            
            print(f"Seed {current_seed}: MAE = {mae:.4f} | Uncond = {uncond_mae:.4f} | Drop = {drop_rate:.4f} ({actual_unserved}/{total_demands})")
        finally:
            env.close()

    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    mean_uncond_mae = np.mean(uncond_maes)
    std_uncond_mae = np.std(uncond_maes)
    mean_drop = np.mean(drop_rates)
    std_drop = np.std(drop_rates)

    print("\n" + "="*50)
    print(f" EVALUATION SUMMARY ({args.heuristic.upper()} WITH MASK)")
    print("="*50)
    print(f"Number of samples (N)      : {args.seeds}")
    print(f"MAE (conditional)        : {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"Uncond MAE             : {mean_uncond_mae:.4f} ± {std_uncond_mae:.4f}")
    print(f"DropRate               : {mean_drop:.4f} ± {std_drop:.4f}")
    print("="*50)

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"masked_{args.heuristic}_results.json")
    
    output_data = {
        "N": args.seeds,
        "heuristic": args.heuristic,
        "mean_mae": float(mean_mae),
        "raw_maes": maes
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
