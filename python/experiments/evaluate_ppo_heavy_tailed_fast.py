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
from python.simulator.main import EventGenerator
from python.simulator.events import ViewerArrival

original_schedule = EventGenerator.schedule_next_arrival

def heavy_tailed_schedule(self, pop_id: int, current_time: float):
    lam = self.pop_lambdas[pop_id]
    if lam <= 0:
        return
        
    if self.deterministic:
        inter_arrival = 1.0 / lam
    else:
        if self.rng:
            inter_arrival = float(self.rng.pareto(3.0)) * (2.0 / lam)
        else:
            inter_arrival = float(np.random.pareto(3.0)) * (2.0 / lam)
            
    arrival_time = current_time + inter_arrival
    
    if arrival_time <= self.until_time:
        demand_id = next(self.demand_counter)
        event_id = next(self.event_counter)
        arrival = ViewerArrival(
            time=arrival_time, 
            event_id=event_id, 
            demand_id=demand_id, 
            pop_id=pop_id
        )
        self.engine.schedule(arrival)

EventGenerator.schedule_next_arrival = heavy_tailed_schedule

from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def run_single_seed(seed, dataset_file, gnn_model_path, model_path):
    model = MaskablePPO.load(model_path)
    env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=24.0, gnn_model_path=gnn_model_path)
    env = ActionMasker(env, mask_fn)
    
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.unwrapped.valid_action_mask())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    unwrapped_env = env.unwrapped
    edge_load = unwrapped_env.edge_load
    anchor_alloc = unwrapped_env.state.anchor_allocation
    mae = float(np.mean(np.abs(edge_load - anchor_alloc)))
    env.close()
    return mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/ppo_orchestrator.zip")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test")
    parser.add_argument("--gnn_model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=200)
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    maes = []
    
    print(f"Starting parallelized Heavy-Tailed evaluation (N={args.seeds})...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(run_single_seed, args.base_seed + i, dataset_file, args.gnn_model, args.model_path): i for i in range(args.seeds)}
        for future in concurrent.futures.as_completed(futures):
            mae = future.result()
            maes.append(mae)
            print(f"Simulation completed, current mean MAE: {np.mean(maes):.4f}")

    mean_mae = np.mean(maes)
    print("\n" + "="*50)
    print(f"Number of samples (N)      : {args.seeds}")
    print(f"Mean MAE error             : {mean_mae:.6f}")
    print("="*50)

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "ppo_heavy_tailed_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"N": args.seeds, "mean_mae_heavy_tailed": float(mean_mae), "raw_maes": maes}, f, indent=4)
    print(f"Saved to: {results_path}")

if __name__ == "__main__":
    main()
