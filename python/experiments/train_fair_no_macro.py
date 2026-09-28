import os
import sys
import numpy as np

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

class FairNoMacroEnv(SrsOrchestrationEnv):
    def _get_obs(self):
        pop_id = 0
        if self.current_arrival_event:
            pop_id = self.current_arrival_event.pop_id
            
        f_t = np.array([pop_id / float(max(1, self.num_pops))], dtype=np.float32)
        
        d_t = np.zeros(20, dtype=np.float32)
        for c_id in range(min(20, self.num_clusters)):
            dist = self.state.pop_distances[pop_id, c_id] + self.state.origin_distances[c_id]
            latency = dist * self.state.latency_per_10km_units
            d_t[c_id] = min(1.0, latency / max(1.0, self.state.max_latency_units))
            
        r_t = np.zeros(22, dtype=np.float32)
        cluster_load = np.sum(self.edge_load, axis=0)
        for c_id in range(min(20, self.num_clusters)):
            # NORMALIZATION WRT PHYSICAL CAPACITY, NOT ANCHOR!
            capacity_limit = float(self.cluster_max_capacity[c_id])
            r_t[c_id] = min(1.0, float(cluster_load[c_id]) / max(1.0, capacity_limit))
            
        last_macro_time = (self.engine.current_time // self.macro_interval) * self.macro_interval
        r_t[20] = min(1.0, float((self.engine.current_time - last_macro_time) / self.macro_interval))
        r_t[21] = min(1.0, float(self.engine.current_time / self.sim_hours))
        
        # Empty budget and delta (agent has no access to oracle)
        budget_t = np.zeros(20, dtype=np.float32)
        delta_t = np.zeros(40, dtype=np.float32)
        
        return np.concatenate([f_t, d_t, r_t, budget_t, delta_t])

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        # Fair reward based purely on throughput (no MILP guidance)
        if action == self.num_clusters:
            mask = self.valid_action_mask()
            c_feas_empty = (sum(mask[:-1]) == 0)
            if c_feas_empty:
                reward = 0.1
            else:
                reward = -1.0
        else:
            reward = 1.0
            
        return obs, reward, terminated, truncated, info

def mask_fn(env):
    return env.unwrapped.valid_action_mask()

def make_env(dataset_file, sim_hours):
    def _init():
        env = FairNoMacroEnv(dataset_file, sim_hours=sim_hours)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def train_fair_no_macro():
    print("--- PPO Training without MILP Baseline (Positive Reward for Throughput) ---")
    dataset_file = "dataset/features_labels.jsonl"
    
    num_envs = 8
    env = SubprocVecEnv([make_env(dataset_file, sim_hours=12.0) for _ in range(num_envs)])
    
    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256, n_epochs=10, learning_rate=3e-4)
    
    print("Starting training (500,000 steps)...")
    model.learn(total_timesteps=500000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_fair_no_macro.zip")
    print("Model saved to models/ppo_fair_no_macro.zip")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    train_fair_no_macro()
