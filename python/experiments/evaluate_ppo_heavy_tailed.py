import os
import sys
import argparse
import json
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

sys.path.append(os.path.abspath("."))
from python.simulator.main import EventGenerator
from python.simulator.events import ViewerArrival

# Monkey-patching EventGenerator.schedule_next_arrival method
# This directly breaks the assumptions of the Palm-Khintchine Theorem (summing independent renewal processes
# approaches Poisson). We use Pareto with a=2.0 (E[X]=1, infinite variance).
# We only patch arrivals, service times (departures) remain exponential.

original_schedule = EventGenerator.schedule_next_arrival

def heavy_tailed_schedule(self, pop_id: int, current_time: float):
    lam = self.pop_lambdas[pop_id]
    if lam <= 0:
        return
        
    if self.deterministic:
        inter_arrival = 1.0 / lam
    else:
        # Standard Pareto in numpy (rng.pareto) is (1/U^(1/a)) - 1.
        # For a=2.0 the mean is 1.0. We multiply by (1.0 / lam) to maintain the expected arrival rate.
        if self.rng:
            inter_arrival = float(self.rng.pareto(3.0)) * (2.0 / lam)
        else:
            import numpy as np
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

sys.path.append(os.path.abspath("."))
from python.rl.env import SrsOrchestrationEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

def make_env(dataset_file: str, gnn_model_path: str = None):
    def _init():
        env = SrsOrchestrationEnv(dataset_file=dataset_file, sim_hours=2.0, gnn_model_path=gnn_model_path)
        env = ActionMasker(env, mask_fn)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Test PPO robustness to Heavy-Tailed traffic")
    parser.add_argument("--model_path", type=str, default="models/ppo_orchestrator.zip")
    parser.add_argument("--dataset_dir", type=str, default="dataset_test")
    parser.add_argument("--gnn_model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=200)
    args = parser.parse_args()

    dataset_file = os.path.join(args.dataset_dir, "features_labels.jsonl")
    
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} does not exist.")
        return

    print("Loading model...")
    model = MaskablePPO.load(args.model_path)
    env_fn = make_env(dataset_file, args.gnn_model)

    maes = []
    print(f"Starting Heavy-Tailed evaluation (N={args.seeds})...")
    
    for i in range(args.seeds):
        current_seed = args.base_seed + i
        np.random.seed(current_seed)
        torch_seed = current_seed
        
        env = env_fn()
        try:
            obs, info = env.reset(seed=current_seed)
            done = False
            
            while not done:
                action, _states = model.predict(obs, deterministic=True, action_masks=env.unwrapped.valid_action_mask())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            unwrapped_env = env.unwrapped
            edge_load = unwrapped_env.edge_load
            anchor_alloc = unwrapped_env.state.anchor_allocation
            assert edge_load.shape == anchor_alloc.shape, "Shapes must match"
            
            mae = float(np.mean(np.abs(edge_load - anchor_alloc)))
            maes.append(mae)
            print(f"Seed {current_seed}: MAE = {mae:.4f}")
        finally:
            env.close()

    mean_mae = np.mean(maes)
    print("\n" + "="*50)
    print(" HEAVY-TAILED EVALUATION SUMMARY (Pareto)")
    print("="*50)
    print(f"Number of samples (N)      : {args.seeds}")
    print(f"Mean MAE error             : {mean_mae:.6f}")
    print("="*50)

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "ppo_heavy_tailed_results.json")
    
    output_data = {
        "N": args.seeds,
        "mean_mae_heavy_tailed": float(mean_mae),
        "raw_maes": maes
    }
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
