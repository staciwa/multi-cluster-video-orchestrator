import os
import sys
import numpy as np
import json

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from sb3_contrib import MaskablePPO
from python.experiments.train_fair_no_macro import FairNoMacroEnv

def evaluate_fair():
    dataset_file = "dataset/features_labels.jsonl"
    env = FairNoMacroEnv(dataset_file, sim_hours=2.0)
    model = MaskablePPO.load("models/ppo_fair_no_macro.zip")
    
    episodes = 3
    maes = []
    drop_rates = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, action_masks=env.valid_action_mask(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        anchor = env.state.anchor_allocation
        current_load = env.edge_load
        mae = np.mean(np.abs(current_load - anchor))
        maes.append(mae)
        
        unserved = env.state.total_unserved
        total = env.state.total_served + env.state.total_unserved
        drop_rate = (unserved / max(1, total)) * 100
        drop_rates.append(drop_rate)
        
        print(f"Ep {ep+1}: MAE={mae:.4f}, Drop={drop_rate:.2f}%")
        
    print(f"MEAN MAE: {np.mean(maes):.4f}")
    print(f"MEAN Drop Rate: {np.mean(drop_rates):.2f}%")

if __name__ == "__main__":
    evaluate_fair()
