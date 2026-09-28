import sys
import os
sys.path.append(os.path.abspath("."))
import argparse
import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from python.rl.env import SrsOrchestrationEnv

def make_env(dataset_file: str, gnn_model_path: str = None):
    def _init():
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent (NO ACTION MASKING) with GNN oracle")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with supervised dataset (e.g., dataset_test)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total number of training steps")
    parser.add_argument("--n_envs", type=int, default=6,
                        help="Number of parallel environments (used CPU cores)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for experiment reproducibility")
    parser.add_argument("--gnn_model", type=str, default=None,
                        help="Path to the trained GNN (e.g., models/baseline_gnn.pt)")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")

    if not os.path.exists(dataset_file):
        print(f"Error: File not found: {dataset_file}.")
        return

    print(f"Initializing vectorized environment ({args.n_envs} processes) without masking...")
    import stable_baselines3
    stable_baselines3.common.utils.set_random_seed(args.seed)
    
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env(make_env(dataset_file, args.gnn_model), n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, seed=args.seed)

    print("Initializing classic PPO Agent (no ActionMasker)...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/", 
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

    print(f"Starting training ({args.timesteps} steps)...")
    model.learn(total_timesteps=args.timesteps, tb_log_name="PPO_NoMask")

    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_nomask.zip"
    model.save(model_path)
    print(f"Model successfully saved in {model_path}!")

if __name__ == "__main__":
    main()
