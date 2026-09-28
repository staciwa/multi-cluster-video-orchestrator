import os
import sys
import numpy as np

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

class RewardHackingEnv(SrsOrchestrationEnv):
    def step(self, action):
        # Get PoP ID *before* calling step, so as not to lose current event
        pop_id = self.current_arrival_event.pop_id if self.current_arrival_event else 0
        
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Overwrite reward with proportional function (Reward Hacking demonstration)
        if action == self.num_clusters:
            # Moderate reward for rejection
            reward = 0.5 
        else:
            # Proportional reward for latency, without looking at anchor
            latency = self.state.pop_distances[pop_id, action] + self.state.origin_distances[action]
            latency *= self.state.latency_per_10km_units
            
            if latency <= self.state.max_latency_units:
                # Agent is tempted with positive values for each accepted request near the cluster
                reward = 1.0 - (latency / max(1.0, self.state.max_latency_units))
            else:
                reward = -1.0
                
        return obs, reward, terminated, truncated, info

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

def mask_fn(env):
    return env.unwrapped.valid_action_mask()

def train_reward_hacking():
    print("--- Reward Hacking Training (Proportional Function) ---")
    dataset_file = "dataset_test/features_labels.jsonl"
    gnn_model_path = "models/best_baseline_gnn.pt"
    
    # Train a small model for e.g. 100k steps to show it gets stuck in a "lazy" optimum
    def _init():
        env = RewardHackingEnv(dataset_file, gnn_model_path=gnn_model_path, sim_hours=2.0)
        env = Monitor(env, "results/reward_hacking")
        env = ActionMasker(env, mask_fn)
        return env
        
    env = DummyVecEnv([_init])
    
    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    
    print("Starting training... (expected silent rejection of requests, since rejection reward is stable and guaranteed, and accepting near edge may lead to unpredictable effects).")
    model.learn(total_timesteps=100000)
    
    model.save("models/ppo_reward_hacking.zip")
    print("Training finished. Check TensorBoard logs in results/reward_hacking/")

if __name__ == "__main__":
    train_reward_hacking()
