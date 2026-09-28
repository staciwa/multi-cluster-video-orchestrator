import os
import sys
import numpy as np

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def evaluate_greedy_latency(dataset_dir: str):
    dataset_file = os.path.join(dataset_dir, "features_labels.jsonl")
    env = SrsOrchestrationEnv(dataset_file)
    
    total_drop_rate = []
    
    print("Starting evaluation of the greedy heuristic (Greedy Latency)...")
    for ep in range(10):
        obs, info = env.reset()
        done = False
        
        while not done:
            current_event = env.current_arrival_event
            if current_event is None:
                obs, reward, terminated, truncated, info = env.step(env.num_clusters)
                done = terminated or truncated
                continue
                
            pop_id = current_event.pop_id
            
            best_cluster = env.num_clusters # Default reject
            min_latency = float('inf')
            
            for c_id in range(env.num_clusters):
                dist = env.state.pop_distances[pop_id, c_id] + env.state.origin_distances[c_id]
                latency = dist * env.state.latency_per_10km_units
                
                # Select the absolutely closest cluster, ignoring capacity
                if latency < min_latency:
                    min_latency = latency
                    best_cluster = c_id
            
            obs, reward, terminated, truncated, info = env.step(best_cluster)
            done = terminated or truncated
            
        drop_rate = env.state.total_unserved / max(1, env.state.total_demands)
        total_drop_rate.append(drop_rate)
        print(f"Episode {ep+1} - Greedy Latency Drop Rate: {drop_rate*100:.2f}%")
        
    print(f"Average Drop Rate (Greedy Latency): {np.mean(total_drop_rate)*100:.2f}%")

if __name__ == "__main__":
    evaluate_greedy_latency("dataset_100k_uniform")
