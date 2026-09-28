import os
import sys
import numpy as np
import json
import argparse
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def evaluate_noise(env, model, noise_std: float, episodes: int = 10):
    maes = []
    drop_rates = []
    
    for ep in range(episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        while not done:
            # Inject Gaussian noise into observations (simulating telemetry delays)
            if noise_std > 0.0:
                noise = np.random.normal(0, noise_std, size=obs.shape)
                noisy_obs = obs + noise
                # Clip to observation space boundaries (Gym Box -1.0 to 1.0)
                noisy_obs = np.clip(noisy_obs, -1.0, 1.0)
            else:
                noisy_obs = obs
                
            action_masks = env.unwrapped.valid_action_mask()
            action, _ = model.predict(noisy_obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        unwrapped = env.unwrapped
        mae = float(np.mean(np.abs(unwrapped.edge_load - unwrapped.state.anchor_allocation)))
        total_demands = unwrapped.state.total_served + unwrapped.state.total_unserved
        drop_rate = unwrapped.state.total_unserved / max(1, total_demands)
        
        maes.append(mae)
        drop_rates.append(drop_rate)
        
    return np.mean(maes), np.mean(drop_rates)

def run_noisy_observations(dataset_path: str, model_path: str, episodes: int, noise_levels: list[float], output: str):
    np.random.seed(42)
    print(f"Loading environment: {dataset_path}")
    env = SrsOrchestrationEnv(dataset_file=dataset_path)
    
    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path)
    
    results = {}
    for std in noise_levels:
        print(f"Testing noise level: std = {std}")
        avg_mae, avg_drop = evaluate_noise(env, model, std, episodes)
        print(f" -> MAE: {avg_mae:.4f} | Drop Rate: {avg_drop*100:.2f}%")
        results[f"std_{std}"] = {
            "mae": avg_mae,
            "drop_rate": avg_drop
        }
        
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO robustness against telemetry noise")
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--model", type=str, default="models/ppo_orchestrator.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--output", type=str, default="results/noisy_observations.json")
    args = parser.parse_args()
    
    run_noisy_observations(args.dataset, args.model, args.episodes, args.noise_levels, args.output)
