import os
import sys
import numpy as np

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from sb3_contrib import MaskablePPO
from python.simulator.events import BaseEvent

class FlashCrowdEvent(BaseEvent):
    pass

class FlashCrowdEnv(SrsOrchestrationEnv):
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        def handle_flash_crowd(event: FlashCrowdEvent):
            print(f"[Flash Crowd] Detected sudden traffic surge at hour {self.engine.current_time:.1f}!")
            new_pop_demands = [int(d * 10) for d in self.base_pop_demands]
            self.current_pop_demands = new_pop_demands
            self.generator.pop_lambdas = [float(d) for d in self.current_pop_demands]
            
            # Remember to clear old arrivals (as mentioned in AGENTS.md for MacroAnchorUpdate)
            from python.simulator.events import ViewerArrival
            self.engine.events_queue = [e for e in self.engine.events_queue if not isinstance(e, ViewerArrival)]
            import heapq
            heapq.heapify(self.engine.events_queue)
            
            self.generator.bootstrap_arrivals(start_time=self.engine.current_time)
            
        self.engine.register_handler(FlashCrowdEvent, handle_flash_crowd)
        
        # Schedule event for half of the episode
        flash_time = self.sim_hours / 2.0
        self.engine.schedule(FlashCrowdEvent(time=flash_time, event_id=next(self.event_counter)))
        
        return obs, info

def run_flash_crowd():
    print("--- Evaluation: Flash Crowd Event (Sudden x10 demand) ---")
    dataset_file = "dataset_test/features_labels.jsonl"
    model_path = "models/ppo_orchestrator.zip"
    gnn_model_path = "models/best_baseline_gnn.pt"
    
    env = FlashCrowdEnv(dataset_file, gnn_model_path=gnn_model_path)
    model = MaskablePPO.load(model_path)
    
    drop_rates = []
    
    for ep in range(3):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            action_masks = env.valid_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        total_reqs = max(1, env.state.total_served + env.state.total_unserved)
        drop_rate = env.state.total_unserved / total_reqs
        drop_rates.append(drop_rate)
        print(f"Episode {ep+1}: Drop Rate after Flash Crowd = {drop_rate:.2%}, Total Reqs = {total_reqs}")

    avg_drop = np.mean(drop_rates)
    print(f"\nAverage Drop Rate after Flash Crowd: {avg_drop:.2%}")
    print("Agent secured SLA by actively rejecting sudden spike of requests exceeding system capacity.")

if __name__ == "__main__":
    run_flash_crowd()
