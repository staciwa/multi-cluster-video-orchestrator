import os
import sys
import numpy as np
import json
from sb3_contrib import MaskablePPO

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def run_evaluation(env, model=None, heuristic=None, outage=False, noise_std=0.0, seed=42):
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    
    if outage:
        original_mask_fn = env.valid_action_mask
        def outage_action_mask():
            mask = np.array(original_mask_fn(), copy=True)
            mask[:5] = 0 # 0-4 are central clusters
            return mask
        env.valid_action_mask = outage_action_mask

    done = False
    
    while not done:
        if noise_std > 0.0:
            noise = rng.normal(0, noise_std, size=obs.shape)
            obs = np.clip(obs + noise, env.observation_space.low, env.observation_space.high)
            
        if heuristic == 'nearest_neighbor':
            current_event = env.current_arrival_event
            if current_event is None:
                action = env.num_clusters
            else:
                pop_id = current_event.pop_id
                action = env.num_clusters
                min_dist = float('inf')
                
                # Only active clusters from the mask (e.g., considering outage)
                mask = env.valid_action_mask()
                for c_id in range(env.num_clusters):
                    if mask[c_id]:
                        dist = env.state.pop_distances[pop_id, c_id] + env.state.origin_distances[c_id]
                        if dist < min_dist:
                            min_dist = dist
                            action = c_id
        else:
            action, _ = model.predict(obs, action_masks=env.valid_action_mask(), deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    total_reqs = max(1, env.state.total_served + env.state.total_unserved)
    drop_rate = env.state.total_unserved / total_reqs
    
    return drop_rate

def main():
    dataset_file = "dataset_100k_uniform/features_labels.jsonl"
    model_path = "models/ppo_orchestrator_uniform.zip"
    
    print("Initializing environment and loading model...")
    env = SrsOrchestrationEnv(dataset_file)
    model = MaskablePPO.load(model_path)
    
    seeds = [42, 123, 999]
    results = {}
    
    print("\n--- 1. Nearest-Neighbor Heuristic (Baseline) ---")
    results['NN_Baseline'] = [run_evaluation(env, heuristic='nearest_neighbor', seed=s) for s in seeds]
    print(f"Drop Rate: {np.mean(results['NN_Baseline'])*100:.2f}% ± {np.std(results['NN_Baseline'])*100:.2f}%")
    
    print("\n--- 2. PPO (Baseline) ---")
    results['PPO_Baseline'] = [run_evaluation(env, model=model, seed=s) for s in seeds]
    print(f"Drop Rate: {np.mean(results['PPO_Baseline'])*100:.2f}% ± {np.std(results['PPO_Baseline'])*100:.2f}%")

    print("\n--- 3. Regional Outage ---")
    results['NN_Outage'] = [run_evaluation(env, heuristic='nearest_neighbor', outage=True, seed=s) for s in seeds]
    results['PPO_Outage'] = [run_evaluation(env, model=model, outage=True, seed=s) for s in seeds]
    print(f"NN Drop Rate (Outage): {np.mean(results['NN_Outage'])*100:.2f}% ± {np.std(results['NN_Outage'])*100:.2f}%")
    print(f"PPO Drop Rate (Outage): {np.mean(results['PPO_Outage'])*100:.2f}% ± {np.std(results['PPO_Outage'])*100:.2f}%")

    print("\n--- 4. Noisy Observations (PPO) ---")
    for noise in [0.05, 0.15, 0.30]:
        res = [run_evaluation(env, model=model, noise_std=noise, seed=s) for s in seeds]
        results[f'PPO_Noise_{noise}'] = res
        print(f"PPO Drop Rate (Noise std={noise}): {np.mean(res)*100:.2f}% ± {np.std(res)*100:.2f}%")

    with open("results/eval_runner_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved results to results/eval_runner_results.json")

if __name__ == "__main__":
    main()
