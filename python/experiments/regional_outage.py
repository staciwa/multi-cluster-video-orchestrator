import os
import sys

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from sb3_contrib import MaskablePPO

class RegionalOutageEnv(SrsOrchestrationEnv):
    def valid_action_mask(self):
        mask = super().valid_action_mask()
        # Disable the first 5 central clusters
        if self.current_arrival_event:
            for i in range(5):
                mask[i] = 0
        return mask

def test_regional_outage():
    print("Simulating Regional Outage (Disabling 5 central clusters)...")
    env = RegionalOutageEnv("dataset_test/features_labels.jsonl")
    model = MaskablePPO.load("models/ppo_orchestrator.zip")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, action_masks=env.valid_action_mask(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    total_reqs = max(1, env.state.total_served + env.state.total_unserved)
    drop_rate = env.state.total_unserved / total_reqs
    print(f"Regional Outage Drop Rate: {drop_rate*100:.2f}%")
    print(f"Unserved requests: {env.state.total_unserved} / {total_reqs}")

if __name__ == "__main__":
    test_regional_outage()

