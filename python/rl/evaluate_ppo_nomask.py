import os
import argparse
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(dataset_file: str, gnn_model_path: str = None):
    def _init():
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
        
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Evaluation of PPO agent")
    parser.add_argument("--model_path", type=str, default="ppo_srs_orchestrator.zip",
                        help="Path to the saved PPO model")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with evaluation data")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument("--gnn_model", type=str, default=None,
                        help="Path to the trained GNN (e.g., models/baseline_gnn.pth)")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        return

    print(f"Reading dataset file O(1): {dataset_file}")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model {args.model_path} does not exist. Run training first!")
        return

    print("Loading model...")
    model = PPO.load(args.model_path)
    
    env_fn = make_env(dataset_file, args.gnn_model)
    env = env_fn()

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        evaluation_results = []
        
        while not done:
            # Deterministic=True during evaluation to disable exploration (randomness)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        unwrapped_env = env.unwrapped
        mae = np.mean(np.abs(unwrapped_env.edge_load - unwrapped_env.state.anchor_allocation))
        
        # Calculate the percentage of served vs Oracle
        ideal_unserved = unwrapped_env.current_case["labels"]["unserved"]
        actual_unserved = unwrapped_env.state.total_unserved
        
        print(f"--- Episode {ep + 1} ---")
        print(f"Number of steps (requests): {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Mean Absolute Error (MAE) on allocation matrix (Agent vs Oracle): {mae:.4f}")
        print(f"Dropped requests: {actual_unserved} (Oracle calculated: {ideal_unserved})")
        print()
        
        evaluation_results.append({
            "episode": ep + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "mae": float(mae),
            "actual_unserved": int(actual_unserved),
            "ideal_unserved": int(ideal_unserved)
        })

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "ppo_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results have been permanently saved in the file: {results_path}")

if __name__ == "__main__":
    main()
