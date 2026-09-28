import os
import argparse
import json
import numpy as np
import scipy.stats as stats
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(dataset_file: str, gnn_model_path: str = None):
    def _init():
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Evaluation of the PPO agent (MAE Confidence Intervals)")
    parser.add_argument("--model_path", type=str, default="models/ppo_orchestrator.zip",
                        help="Path to the saved PPO model")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with evaluation data")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of test episodes per seed (default 1)")
    parser.add_argument("--gnn_model", type=str, default="models/best_baseline_gnn.pt",
                        help="Path to the trained GNN")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of independent random seeds (N)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model {args.model_path} does not exist. Train it first!")
        return

    print(f"Loading dataset index: {dataset_file}")
    print("Loading model...")
    model = MaskablePPO.load(args.model_path, custom_objects={"clip_range": lambda _: 0.2})
    
    env_fn = make_env(dataset_file, args.gnn_model)
    env = env_fn()

    maes = []
    
    print(f"Starting evaluation for {args.seeds} random seeds...")
    
    for i in range(args.seeds):
        current_seed = args.base_seed + i
        np.random.seed(current_seed)
        
        obs, info = env.reset(seed=current_seed)
        done = False
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True, action_masks=env.unwrapped.valid_action_mask())
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            
        unwrapped_env = env.unwrapped
        mae = float(np.mean(np.abs(unwrapped_env.edge_load - unwrapped_env.state.anchor_allocation)))
        maes.append(mae)
        
        print(f"Seed {current_seed}: MAE = {mae:.4f}")

    # Statistics and 95% confidence intervals
    maes_arr = np.array(maes)
    mean_mae = np.mean(maes_arr)
    std_mae = np.std(maes_arr, ddof=1)
    
    # Student's t-distribution for confidence interval
    alpha = 0.05
    t_critical = stats.t.ppf(1.0 - alpha/2, df=args.seeds - 1)
    margin_of_error = t_critical * (std_mae / np.sqrt(args.seeds))
    
    ci_lower = mean_mae - margin_of_error
    ci_upper = mean_mae + margin_of_error
    
    print("\n" + "="*50)
    print(" STATISTICAL EVALUATION SUMMARY (MAE)")
    print("="*50)
    print(f"Number of samples (N)      : {args.seeds}")
    print(f"Mean MAE error             : {mean_mae:.6f}")
    print(f"Standard deviation         : {std_mae:.6f}")
    print(f"Margin of error (95%)      : ±{margin_of_error:.6f}")
    print(f"Confidence interval (95%)  : [{ci_lower:.6f}, {ci_upper:.6f}]")
    print("="*50)
    
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "ppo_confidence_intervals.json")
    
    output_data = {
        "N": args.seeds,
        "base_seed": args.base_seed,
        "mean_mae": float(mean_mae),
        "std_mae": float(std_mae),
        "margin_of_error": float(margin_of_error),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "raw_maes": maes
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
