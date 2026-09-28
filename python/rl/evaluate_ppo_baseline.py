import os
import argparse
import json
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

class BaselineObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that zeroes out the 'delta_t' part (last 40 dimensions) after receiving
    the observation from SrsOrchestrationEnv. Because of this, the PPO Baseline
    does not see GNN hints, and random weights on these inputs do not disturb predictions.
    This allows testing Baseline in the full environment with an oracle (to calculate MAE).
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        
    def observation(self, observation):
        # observation has 103 dimensions. The last 40 are delta_t (hints).
        # In BaselineEnv these dimensions were zeros, so we must zero them during tests as well.
        obs = observation.copy()
        obs[-40:] = 0.0
        return obs

def make_env(dataset_file: str):
    def _init():
        # We use the target SrsOrchestrationEnv to have the oracle embedded for calculating MAE,
        # but we don't pass gnn_model_path, because baseline doesn't use it.
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=None)
        env = BaselineObservationWrapper(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Evaluation of PPO Baseline Agent (no GNN oracle)")
    parser.add_argument("--model_path", type=str, default="models/ppo_baseline.zip",
                        help="Path to the saved PPO Baseline model")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test",
                        help="Directory with evaluation data")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model {args.model_path} does not exist. Run baseline training first!")
        return

    print("Loading Baseline model...")
    model = MaskablePPO.load(args.model_path)
    
    env_fn = make_env(dataset_file)
    env = env_fn()

    evaluation_results = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Deterministic=True
            action, _states = model.predict(obs, deterministic=True, action_masks=env.unwrapped.valid_action_mask())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        unwrapped_env = env.unwrapped
        mae = np.mean(np.abs(unwrapped_env.edge_load - unwrapped_env.state.anchor_allocation))
        
        ideal_unserved = unwrapped_env.current_case["labels"]["unserved"]
        actual_unserved = unwrapped_env.state.total_unserved
        
        print(f"--- Episode {ep + 1} ---")
        print(f"Number of steps (requests): {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"MAE Error (Agent vs MILP Oracle): {mae:.4f}")
        print(f"Dropped requests: {actual_unserved} (MILP Oracle calculated: {ideal_unserved})")
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
    results_path = os.path.join("results", "ppo_baseline_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results saved in: {results_path}")

if __name__ == "__main__":
    main()
