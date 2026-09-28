import os
import sys
import argparse
import json
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import concurrent.futures

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def evaluate_baseline_seed(seed, dataset_file, model_path, gnn_model_path):
    env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO.load(model_path)
    
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    done = False
    
    while not done:
        action_masks = env.unwrapped.valid_action_mask()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    unwrapped = env.unwrapped
    mae = float(np.mean(np.abs(unwrapped.edge_load - unwrapped.state.anchor_allocation)))
    env.close()
    return mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--model", type=str, default="models/ppo_orchestrator.zip")
    parser.add_argument("--gnn_model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print(f"Starting parallelized PPO baseline evaluation for N={args.episodes}...")
    maes_milp = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # MILP MASK (no gnn_model)
        futures = [executor.submit(evaluate_baseline_seed, 1000+ep, args.dataset, args.model, None) for ep in range(args.episodes)]
        for f in concurrent.futures.as_completed(futures):
            maes_milp.append(f.result())
            
    mean_milp = np.mean(maes_milp)
    print(f"Mean MAE (MILP Mask, N={args.episodes}): {mean_milp:.4f}")

    print(f"Evaluating with GNN mask...")
    maes_gnn = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(evaluate_baseline_seed, 2000+ep, args.dataset, args.model, args.gnn_model) for ep in range(args.episodes)]
        for f in concurrent.futures.as_completed(futures):
            maes_gnn.append(f.result())
            
    mean_gnn = np.mean(maes_gnn)
    print(f"Mean MAE (GNN Mask, N={args.episodes}): {mean_gnn:.4f}")
    
    with open("results/ppo_baseline_1000.json", "w") as f:
        json.dump({
            "N": args.episodes,
            "mae_milp": float(mean_milp),
            "mae_gnn": float(mean_gnn)
        }, f, indent=4)
    print("Saved to results/ppo_baseline_1000.json")

if __name__ == "__main__":
    multiprocessing = __import__("multiprocessing")
    multiprocessing.set_start_method("spawn")
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
