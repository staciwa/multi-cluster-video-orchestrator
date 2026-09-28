import os
import sys
import numpy as np
import json
from sb3_contrib import MaskablePPO

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv
from python.simulator.main import EventGenerator

def run_heavy_tailed_eval(shape: float = 2.5, seeds=[42, 123, 999]):
    print(f"\n--- Evaluation of Heavy-Tailed traffic (Pareto, shape={shape}) ---")
    dataset_file = "dataset_100k_uniform/features_labels.jsonl"
    model_path = "models/ppo_orchestrator_uniform.zip"
    gnn_model_path = "models/best_baseline_gnn_uniform.pt"
    
    env = SrsOrchestrationEnv(dataset_file, gnn_model_path=gnn_model_path)
    model = MaskablePPO.load(model_path)
    
    original_schedule = EventGenerator.schedule_next_arrival
    
    def pareto_schedule_next_arrival(self, pop_id: int, current_time: float):
        lam = self.pop_lambdas[pop_id]
        if lam <= 0:
            return
            
        if self.deterministic:
            inter_arrival = 1.0 / lam
        else:
            mean_inter_arrival = 1.0 / lam
            # To maintain the same mean as the Poisson process:
            scale = mean_inter_arrival * (shape - 1.0) / shape
            
            if self.rng:
                inter_arrival = float((self.rng.pareto(shape) + 1.0) * scale)
            else:
                import random
                inter_arrival = float(random.paretovariate(shape) * scale)
                
            # Safeguard against infinite loop in DES engine in case of floating-point underflow
            inter_arrival = max(0.001, inter_arrival)
            
        arrival_time = current_time + inter_arrival
        
        if arrival_time <= self.until_time:
            demand_id = next(self.demand_counter)
            event_id = next(self.event_counter)
            from python.simulator.events import ViewerArrival
            arrival = ViewerArrival(
                time=arrival_time, 
                event_id=event_id, 
                demand_id=demand_id, 
                pop_id=pop_id
            )
            self.engine.schedule(arrival)

    # Monkey-patching the EventGenerator class before calling reset()
    EventGenerator.schedule_next_arrival = pareto_schedule_next_arrival
    
    results = []
    
    for s in seeds:
        obs, info = env.reset(seed=s)
        done = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.valid_action_mask(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        total_reqs = max(1, env.state.total_served + env.state.total_unserved)
        drop_rate = env.state.total_unserved / total_reqs
        results.append(drop_rate)
        print(f"Seed {s}: Drop Rate = {drop_rate*100:.2f}%")
        
    print(f"Average Drop Rate (Pareto): {np.mean(results)*100:.2f}% ± {np.std(results)*100:.2f}%")
    
    # Restore original logic in case of further experiments in the same process
    EventGenerator.schedule_next_arrival = original_schedule

if __name__ == "__main__":
    run_heavy_tailed_eval(shape=2.5)
    run_heavy_tailed_eval(shape=1.5) # fatter tail (infinite variance)
