import os
import sys
import numpy as np
import json
import argparse
import multiprocessing
from sb3_contrib import MaskablePPO
import concurrent.futures

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def evaluate_noise_seed(seed, dataset_path, model_path, noise_std):
    env = SrsOrchestrationEnv(dataset_file=dataset_path, sim_hours=24.0)
    model = MaskablePPO.load(model_path)
    
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=obs.shape)
            noisy_obs = np.clip(obs + noise, -1.0, 1.0)
        else:
            noisy_obs = obs
            
        action_masks = env.unwrapped.valid_action_mask()
        action, _ = model.predict(noisy_obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    unwrapped = env.unwrapped
    mae = float(np.mean(np.abs(unwrapped.edge_load - unwrapped.state.anchor_allocation)))
    total_demands = unwrapped.state.total_served + unwrapped.state.total_unserved
    drop_rate = float(unwrapped.state.total_unserved / max(1, total_demands))
    
    env.close()
    return mae, drop_rate

def run_noisy_observations_fast(dataset_path, model_path, episodes, noise_levels, output):
    results = {}
    
    for std in noise_levels:
        print(f"Testowanie poziomu szumu: std = {std}")
        maes = []
        drop_rates = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(evaluate_noise_seed, 42 + ep, dataset_path, model_path, std) for ep in range(episodes)]
            for future in concurrent.futures.as_completed(futures):
                mae, drop = future.result()
                maes.append(mae)
                drop_rates.append(drop)
                
        avg_mae = np.mean(maes)
        avg_drop = np.mean(drop_rates)
        print(f" -> MAE: {avg_mae:.4f} | Drop Rate: {avg_drop*100:.2f}%")
        
        results[f"std_{std}"] = {
            "mae": float(avg_mae),
            "std_mae": float(np.std(maes)),
            "drop_rate": float(avg_drop)
        }
        
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"Wyniki zapisano do {output}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--model", type=str, default="models/ppo_orchestrator.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--output", type=str, default="results/noisy_observations.json")
    args = parser.parse_args()
    
    run_noisy_observations_fast(args.dataset, args.model, args.episodes, args.noise_levels, args.output)
