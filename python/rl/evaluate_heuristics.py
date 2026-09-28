import os
import argparse
import json
import numpy as np
import gymnasium as gym
import sys
import random

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def heuristic_random(env: SrsOrchestrationEnv) -> int:
    mask = env.valid_action_mask()
    valid_clusters = [i for i in range(env.num_clusters) if mask[i] == 1]
    if not valid_clusters:
        return env.num_clusters # Reject action
    return random.choice(valid_clusters)

def heuristic_least_loaded(env: SrsOrchestrationEnv) -> int:
    # Assign to the cluster with the largest available budget according to the reference model
    mask = env.valid_action_mask()
    valid_clusters = [i for i in range(env.num_clusters) if mask[i] == 1]
    if not valid_clusters:
        return env.num_clusters
    
    pop_id = env.current_arrival_event.pop_id
    best_cluster = -1
    max_free_budget = -1
    
    for c_id in valid_clusters:
        free_budget = env.state.anchor_allocation[pop_id, c_id] - env.edge_load[pop_id, c_id]
        if free_budget > max_free_budget:
            max_free_budget = free_budget
            best_cluster = c_id
            
    if best_cluster != -1:
        return best_cluster
    return env.num_clusters

def heuristic_nearest(env: SrsOrchestrationEnv) -> int:
    # Assign to the nearest cluster with available resources
    mask = env.valid_action_mask()
    valid_clusters = [i for i in range(env.num_clusters) if mask[i] == 1]
    if not valid_clusters:
        return env.num_clusters
        
    pop_id = env.current_arrival_event.pop_id
    best_cluster = -1
    min_latency = float('inf')
    
    for c_id in valid_clusters:
        dist = env.state.pop_distances[pop_id, c_id] + env.state.origin_distances[c_id]
        latency = dist * env.state.latency_per_10km_units
        if latency < min_latency:
            min_latency = latency
            best_cluster = c_id
            
    if best_cluster != -1:
        return best_cluster
    return env.num_clusters

def evaluate_heuristic(name: str, heuristic_fn, dataset_file: str, episodes: int = 5, gnn_model_path: str = None):
    print(f"Starting evaluation of heuristic: {name}")
    env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
    
    evaluation_results = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action = heuristic_fn(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        unwrapped_env = env.unwrapped
        mae = np.mean(np.abs(unwrapped_env.edge_load - unwrapped_env.state.anchor_allocation))
        
        ideal_unserved = unwrapped_env.current_case["labels"]["unserved"]
        actual_unserved = unwrapped_env.state.total_unserved
        
        print(f"[{name}] Episode {ep + 1}/{episodes} - Steps: {steps}, Reward: {total_reward:.2f}, MAE: {mae:.4f}, Dropped: {actual_unserved} (Solver: {ideal_unserved})")
        
        evaluation_results.append({
            "heuristic": name,
            "episode": ep + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "mae": float(mae),
            "actual_unserved": int(actual_unserved),
            "ideal_unserved": int(ideal_unserved)
        })
        
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluation of baseline heuristics")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with evaluation data")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes per heuristic")
    parser.add_argument("--gnn_model", type=str, default=None,
                        help="Path to the trained GNN (e.g., models/baseline_gnn.pth)")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        return

    print(f"Loading dataset index: {dataset_file}")
    
    heuristics = [
        ("Random", heuristic_random),
        ("LeastRequestedPriority (LeastLoaded)", heuristic_least_loaded),
        ("NearestCluster (GreedyLatency)", heuristic_nearest)
    ]
    
    all_results = []
    
    for name, func in heuristics:
        results = evaluate_heuristic(name, func, dataset_file, args.episodes, args.gnn_model)
        all_results.extend(results)
        print("-" * 40)

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "heuristics_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Full evaluation results have been saved in: {results_path}")

if __name__ == "__main__":
    main()
