import sys
import os
sys.path.append(os.path.abspath("."))
import argparse
import datetime
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker

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
    parser = argparse.ArgumentParser(description="Train the PPO agent on the micro-scale simulator")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with supervised dataset (e.g., dataset_test)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total number of training steps")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments (used CPU cores)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for experiment reproducibility (at least 10 for statistics)")
    parser.add_argument("--gnn_model", type=str, default="best_baseline_gnn.pt", help="Path to the trained GNN (e.g., models/baseline_gnn.pth) - closes the RL-GNN loop.")
    parser.add_argument("--output-model", type=str, default="models/ppo_orchestrator", help="Path to save the model (without .zip)")
    
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")

    if not os.path.exists(dataset_file):
        print(f"Error: File {dataset_file} not found. Make sure the dataset has been generated!")
        return

    print(f"Dataset file found: {dataset_file}")
    print(f"Initializing vectorized environment ({args.n_envs} processes) - each subprocess will load data independently...")
    
    import stable_baselines3
    stable_baselines3.common.utils.set_random_seed(args.seed)
    
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    env = make_vec_env(make_env(dataset_file, args.gnn_model), n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)

    print("Initializing MaskablePPO Agent...")
    # PPO hyperparameters chosen for learning stability in the simulation environment
    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/", 
                        seed=args.seed,
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.001,
                        policy_kwargs=dict(net_arch=[256, 256, 128]))

    print(f"Starting training ({args.timesteps} steps on {args.n_envs} cores)...")
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    model.save(args.output_model)
    print(f"Model successfully saved in {args.output_model}.zip!")

if __name__ == "__main__":
    main()
