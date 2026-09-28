import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
import itertools
import random
from typing import Tuple, Dict, Any, List

from python.simulator.engine import SimulatorEngine
from python.simulator.state import SimulatorState
from python.simulator.events import ViewerArrival, ViewerDeparture, MacroAnchorUpdate
from python.simulator.main import EventGenerator
from python.orchestrator import try_load_gnn_model, predict_gnn_placement

class SrsOrchestrationEnv(gym.Env):
    """
    Gymnasium environment for Reinforcement Learning (PPO).
    Preempts (pauses) the simulator every time a new request arrives
    and waits for the agent's decision (cluster selection).
    """
    def __init__(self, dataset_file: str, sim_hours: float = 24.0, macro_interval: float = 0.25, deterministic: bool = False, gnn_model_path: str = None):
        super().__init__()
        self.dataset_file = dataset_file
        self.sim_hours = sim_hours
        self.macro_interval = macro_interval
        self.deterministic = deterministic
        
        self.gnn_bundle = None
        self.max_physical_instances_edge = None
        self.max_physical_instances_core = None
        if gnn_model_path is not None:
            import sys
            import os
            sys.path.append(os.path.abspath("."))
            from python.orchestrator import try_load_gnn_model
            self.gnn_bundle = try_load_gnn_model(gnn_model_path)
            
        self.line_offsets = []
        with open(self.dataset_file, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.line_offsets.append(offset)
                offset += len(line)
                
        # Load from the first case to get dimensions
        first_case = self._load_case(self.line_offsets[0])
        temp_state = SimulatorState(case_dict=first_case["input"]["meta"])
        self.num_pops = temp_state.num_pops
        self.num_clusters = temp_state.num_clusters
        
        # The observation vector dimension is optimized for the REFERENCE topology (20 MAN clusters, 30 PoPs).
        self.action_space = spaces.Discrete(self.num_clusters + 1)
        
        # Observation vector with a fixed dimension of 103:
        # - Active PoP identifier (1)
        # - Latencies to clusters (20)
        # - Cluster load and temporal metrics (22)
        # - Remaining local PoP budget (20)
        # - Global cluster balance and regional zones (40)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(103,), dtype=np.float32)
        assert self.num_clusters <= 20, "The observation space (103) is tailored to the REFERENCE topology (20 clusters)."
        
        self.current_arrival_event = None
        
    def _load_case(self, offset):
        import json
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return json.loads(f.readline())
        
    def _update_cluster_capacity(self, meta):
        import sys, os
        sys.path.append(os.path.join(os.path.abspath("."), "minizinc"))
        from erlang_utils import erlang_b, calculate_traffic_per_demand
        
        traffic = calculate_traffic_per_demand(traffic=meta["erlang"]["trafficPerDemand"])
        target_blocking = meta["erlang"]["blockageProbability"]
        srs_capacity = meta["srsCapacity"]
        
        self.cluster_max_capacity = np.zeros(self.num_clusters, dtype=np.int32)
        for c_id in range(self.num_clusters):
            c_res = meta["clusterResources"][c_id]
            s_res = meta["srsResources"]
            cluster_instances = min(c_res[0] // s_res[0], c_res[1] // s_res[1])
            total_channels = cluster_instances * srs_capacity
            
            n = 0
            while True:
                if erlang_b((n + 1) * traffic, total_channels) <= target_blocking:
                    n += 1
                else:
                    break
            self.cluster_max_capacity[c_id] = n

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
            
        self.engine = SimulatorEngine()
        if self.deterministic:
            self.current_case = self._load_case(self.line_offsets[0])
        else:
            self.current_case = self._load_case(self.np_random.choice(self.line_offsets))
        
        # The model operates on the average number of active sessions
        
        # Inject the allocation computed by MiniZinc (ground truth) (without erroneous cohort_size scaling)
        anchor_labels = np.array(self.current_case["labels"]["placement"], dtype=np.int32)
        
        self.state = SimulatorState(case_dict=self.current_case["input"]["meta"], anchor_labels=anchor_labels)
        self.num_clusters = self.state.num_clusters
        self.num_pops = self.state.num_pops
        assert self.num_pops <= 30, f"The dataset contains {self.num_pops} PoPs, and the environment supports max 30."
        assert self.num_clusters <= 20, f"The dataset contains {self.num_clusters} clusters, and the environment supports max 20."
        self.event_counter = itertools.count()
        self.demand_counter = itertools.count()
        
        self.edge_load = np.zeros((self.num_pops, self.num_clusters), dtype=np.int32)
        
        # Calculate the global physical capacity of clusters (for the mask)
        meta = self.current_case["input"]["meta"]
        self._update_cluster_capacity(meta)
            
        # Match the event stream intensity exactly to the expected macro demand (Erlang)
        raw_demands = self.current_case["input"]["meta"].get("popDemands", [0]*self.num_pops)
        self.base_pop_demands = [int(d) for d in raw_demands]
        self.current_pop_demands = list(self.base_pop_demands)
        # Average number of simultaneous sessions = lambda / mu. Since mu=1 (1 hour), lambda = d [1/h].
        pop_mu = self.current_case["input"]["meta"].get("popMu", [1.0]*self.num_pops)
        pop_lambdas = [float(d * m) for d, m in zip(self.current_pop_demands, pop_mu)]
        self.service_mu = float(pop_mu[0]) if len(pop_mu) > 0 else 1.0 # assume mu is similar, use the first one (or could average)
        
        self.generator = EventGenerator(self.engine, pop_lambdas, self.sim_hours, self.event_counter, self.demand_counter, deterministic=self.deterministic, rng=self.np_random)
        
        def handle_departure(event: ViewerDeparture):
            self.state.remove_viewer(event.demand_id)
            self.edge_load[event.pop_id, event.cluster_id] = max(0, self.edge_load[event.pop_id, event.cluster_id] - 1)
                
        def handle_macro_update(event: MacroAnchorUpdate):
            # Dynamic update of the macroscopic oracle (every macro_interval)
            # In a real deployment, GNN/MILP provides a new vector here for the next window.
            # Here we simulate this by sampling a new forecast (case) from the dataset,
            # which replaces the old 24-hour forecast.
            
            # Skip at time t=0, as it was done in reset()
            if event.time > 0.0:
                if self.gnn_bundle is not None:
                    # Use the GNN model for fast real-time anchor prediction!
                    if self.deterministic:
                        case_idx = (self.engine.processed_events // 1000) % len(self.line_offsets)
                        new_case = self._load_case(self.line_offsets[case_idx])
                    else:
                        new_case = self._load_case(self.np_random.choice(self.line_offsets))
                    meta = new_case["input"]["meta"]
                    
                    is_valid = True
                    if meta.get("pops", []) and len(meta.get("pops", [])) != self.num_pops:
                        print(f"[WARN] Skipping GNN update: mismatched number of PoPs ({len(meta.get('pops', []))} vs {self.num_pops})")
                        is_valid = False
                    if meta.get("clusters", []) and len(meta.get("clusters", [])) != self.num_clusters:
                        print(f"[WARN] Skipping GNN update: mismatched number of clusters ({len(meta.get('clusters', []))} vs {self.num_clusters})")
                        is_valid = False
                        
                    if not is_valid:
                        # Schedule next update and abort this one safely without crashing
                        next_time = event.time + self.macro_interval
                        if next_time <= self.sim_hours:
                            self.engine.schedule(MacroAnchorUpdate(time=next_time, event_id=next(self.event_counter)))
                        return
                    
                    predicted_anchor = predict_gnn_placement(self.gnn_bundle, meta)
                    self.state.anchor_allocation = np.array(predicted_anchor, dtype=np.int32)
                    
                    new_raw_demands = meta.get("popDemands", [0]*self.num_pops)
                    self.base_pop_demands = [int(d) for d in new_raw_demands]
                    self.current_pop_demands = list(self.base_pop_demands)
                    
                    new_pop_mu = meta.get("popMu", [1.0]*self.num_pops)
                    self.generator.pop_lambdas = [float(d * m) for d, m in zip(self.current_pop_demands, new_pop_mu)]
                    self.service_mu = float(new_pop_mu[0]) if len(new_pop_mu) > 0 else 1.0
                    
                    # Topology update (Origin and physical capacity mask)
                    self._update_cluster_capacity(meta)
                    self.state.origin_distances = np.array(meta.get("originDistances", self.state.origin_distances), dtype=np.int32)
                    self.state.pop_distances = np.array(meta.get("popDistances", self.state.pop_distances), dtype=np.int32)
                else:
                    # Update lambdas removing the event mixing effect (Offline Imitation Learning)
                    if self.deterministic:
                        # Sequential selection for fully deterministic offline mode
                        case_idx = (self.engine.processed_events // 1000) % len(self.line_offsets)
                        new_case = self._load_case(self.line_offsets[case_idx])
                    else:
                        new_case = self._load_case(self.np_random.choice(self.line_offsets))
                    meta = new_case["input"]["meta"]
                    
                    is_valid = True
                    if meta.get("pops", []) and len(meta.get("pops", [])) != self.num_pops:
                        print(f"[WARN] Skipping Offline update: mismatched number of PoPs ({len(meta.get('pops', []))} vs {self.num_pops})")
                        is_valid = False
                    if meta.get("clusters", []) and len(meta.get("clusters", [])) != self.num_clusters:
                        print(f"[WARN] Skipping Offline update: mismatched number of clusters ({len(meta.get('clusters', []))} vs {self.num_clusters})")
                        is_valid = False
                        
                    if not is_valid:
                        next_time = event.time + self.macro_interval
                        if next_time <= self.sim_hours:
                            self.engine.schedule(MacroAnchorUpdate(time=next_time, event_id=next(self.event_counter)))
                        return
                        
                    self.state.anchor_allocation = np.array(new_case["labels"]["placement"], dtype=np.int32)
                    new_raw_demands = meta.get("popDemands", [0]*self.num_pops)
                    self.base_pop_demands = [int(d) for d in new_raw_demands]
                    self.current_pop_demands = list(self.base_pop_demands)
                    
                    new_pop_mu = meta.get("popMu", [1.0]*self.num_pops)
                    self.generator.pop_lambdas = [float(d * m) for d, m in zip(self.current_pop_demands, new_pop_mu)]
                    self.service_mu = float(new_pop_mu[0]) if len(new_pop_mu) > 0 else 1.0
                    
                    # Topology update (Origin and physical capacity mask)
                    self._update_cluster_capacity(meta)
                    self.state.origin_distances = np.array(meta.get("originDistances", self.state.origin_distances), dtype=np.int32)
                    self.state.pop_distances = np.array(meta.get("popDistances", self.state.pop_distances), dtype=np.int32)
                    
                # Clearing outdated arrivals from the queue and bootstrapping new lambdas
                new_queue = []
                for q_event in self.engine.events_queue:
                    if not isinstance(q_event, ViewerArrival):
                        new_queue.append(q_event)
                self.engine.events_queue = new_queue
                heapq.heapify(self.engine.events_queue)
                self.generator.bootstrap_arrivals(start_time=self.engine.current_time)
                
                # Reset would apply to accumulated counters, but we moved away from them in favor of actual edge load (edge_load).
                # Edge load updates automatically (session births and deaths).
                
            next_time = event.time + self.macro_interval
            if next_time <= self.sim_hours:
                self.engine.schedule(MacroAnchorUpdate(time=next_time, event_id=next(self.event_counter)))
                
        self.engine.register_handler(ViewerDeparture, handle_departure)
        self.engine.register_handler(MacroAnchorUpdate, handle_macro_update)
        
        # First update at t=0
        self.engine.schedule(MacroAnchorUpdate(time=0.0, event_id=next(self.event_counter)))
        self.generator.bootstrap_arrivals()
        
        return self._step_to_next_arrival(), {}
        
    def _step_to_next_arrival(self):
        """Runs the simulation loop until the next client arrives."""
        while self.engine.events_queue:
            next_event = self.engine.events_queue[0]
            if next_event.time > self.sim_hours:
                break
                
            if isinstance(next_event, ViewerArrival):
                self.current_arrival_event = heapq.heappop(self.engine.events_queue)
                self.engine.current_time = self.current_arrival_event.time
                self.engine.processed_events += 1
                return self._get_obs()
                
            event = heapq.heappop(self.engine.events_queue)
            self.engine.current_time = event.time
            self.engine.processed_events += 1
            handler = self.engine.handlers.get(type(event))
            if handler:
                handler(event)
                
        self.current_arrival_event = None
        return self._get_obs()
        
    def _get_obs(self):
        pop_id = 0
        if self.current_arrival_event:
            pop_id = self.current_arrival_event.pop_id
            
        # F_t: 1 dimension (pop_id fraction)
        f_t = np.array([pop_id / float(max(1, self.num_pops))], dtype=np.float32)
        
        # D_t: 20 dimensions (distances)
        d_t = np.zeros(20, dtype=np.float32)
        for c_id in range(min(20, self.num_clusters)):
            dist = self.state.pop_distances[pop_id, c_id] + self.state.origin_distances[c_id]
            latency = dist * self.state.latency_per_10km_units
            d_t[c_id] = min(1.0, latency / max(1.0, self.state.max_latency_units))
            
        # R_t: 22 dimensions (20 for cluster load, 2 for global temporal metrics)
        r_t = np.zeros(22, dtype=np.float32)
        cluster_load = np.sum(self.edge_load, axis=0) # sum over PoPs
        for c_id in range(min(20, self.num_clusters)):
            # Normalize against anchor's allocated demands (capacity limit)
            capacity_limit = float(np.sum(self.state.anchor_allocation[:, c_id]))
            r_t[c_id] = min(1.0, float(cluster_load[c_id]) / max(1.0, capacity_limit))
            
        # Budget_t: 20 dimensions (remaining budget per cluster for CURRENT pop_id)
        budget_t = np.zeros(20, dtype=np.float32)
        for c_id in range(min(20, self.num_clusters)):
            anc = float(self.state.anchor_allocation[pop_id, c_id])
            cur = float(self.edge_load[pop_id, c_id])
            budget_t[c_id] = np.clip((anc - cur) / max(1.0, anc), -1.0, 1.0)
            
        # Time metrics of window and simulation episode progress
        last_macro_time = (self.engine.current_time // self.macro_interval) * self.macro_interval
        r_t[20] = min(1.0, float((self.engine.current_time - last_macro_time) / self.macro_interval)) # time in window
        r_t[21] = min(1.0, float(self.engine.current_time / self.sim_hours)) # time in episode
        
        # Delta_t: 40 dimensions (20 per cluster, 20 per region)
        delta_t = np.zeros(40, dtype=np.float32)
        diff = self.state.anchor_allocation - self.edge_load
        diff_cluster = np.sum(diff, axis=0)
        diff_pop = np.sum(diff, axis=1)
        
        # Get max demands dynamically for normalization
        max_cluster_demand = float(max(1.0, np.max(np.sum(self.state.anchor_allocation, axis=0))))
        max_pop_demand = float(max(1.0, np.max(np.sum(self.state.anchor_allocation, axis=1))))
        
        for c_id in range(min(20, self.num_clusters)):
            delta_t[c_id] = np.clip(diff_cluster[c_id] / max_cluster_demand, -1.0, 1.0)
            
        # Regional aggregation of PoP zones
        counts = np.zeros(20, dtype=np.float32)
        for p_id in range(self.num_pops):
            region_idx = p_id % 20
            delta_t[20 + region_idx] += diff_pop[p_id] / max_pop_demand
            counts[region_idx] += 1.0
            
        counts[counts == 0] = 1.0
        delta_t[20:] = np.clip(delta_t[20:] / counts, -1.0, 1.0)
        
        obs = np.concatenate([f_t, d_t, r_t, budget_t, delta_t]).astype(np.float32)
        return obs
        
    def valid_action_mask(self):
        """Returns a 1/0 array for allowed clusters and the rejection action."""
        mask = np.zeros(self.num_clusters + 1, dtype=np.int8)
        if not self.current_arrival_event:
            return mask
            
        pop_id = self.current_arrival_event.pop_id
        cluster_load = np.sum(self.edge_load, axis=0)
        
        for c_id in range(self.num_clusters):
            # Verification of QoS latency constraint
            dist = self.state.pop_distances[pop_id, c_id] + self.state.origin_distances[c_id]
            latency = dist * self.state.latency_per_10km_units
            
            # Verification of global physical cluster capacity limit (not A*)
            if latency <= self.state.max_latency_units and cluster_load[c_id] < self.cluster_max_capacity[c_id]:
                mask[c_id] = 1
                
        # Rejection action (last index) always allowed
        mask[self.num_clusters] = 1
        return mask
        
    def step(self, action: int):
        if self.current_arrival_event is None:
            return self._get_obs(), 0.0, True, False, {}
            
        cluster_id = int(action)
        pop_id = self.current_arrival_event.pop_id
        demand_id = self.current_arrival_event.demand_id
        
        reward = 0.0
        
        if cluster_id == self.num_clusters:
            # Action: Reject request
            mask = self.valid_action_mask()
            c_feas_empty = (sum(mask[:-1]) == 0)
            if np.sum(self.edge_load[pop_id, :]) >= np.sum(self.state.anchor_allocation[pop_id, :]) or c_feas_empty:
                reward = 0.1 # Correct rejection after budget exhaustion or lack of feasible clusters
            else:
                reward = -1.0 # Unjustified rejection with available budget
                
            self.state.total_unserved += 1
        else:
            # Action: Allocation to cluster
            success = self.state.allocate_viewer(demand_id, cluster_id, pop_id)
            if success:
                self.edge_load[pop_id, cluster_id] += 1
                
                anchor_val = self.state.anchor_allocation[pop_id, cluster_id]
                if self.edge_load[pop_id, cluster_id] <= anchor_val:
                    reward = 1.0 # Correct allocation within A* budget
                else:
                    reward = -1.0 # Exceeding A* budget
                
                service_time = float(self.np_random.exponential(1.0 / self.service_mu))
                departure = ViewerDeparture(
                    time=self.engine.current_time + service_time,
                    event_id=next(self.event_counter),
                    demand_id=demand_id,
                    cluster_id=cluster_id,
                    pop_id=pop_id
                )
                if departure.time <= self.sim_hours:
                    self.engine.schedule(departure)
            else:
                self.state.total_unserved += 1
                reward = -1.0 # Penalty for violating QoS
            
        # Schedule next arrival from the same city
        self.generator.schedule_next_arrival(pop_id, self.engine.current_time)
        
        obs = self._step_to_next_arrival()
        done = self.current_arrival_event is None
        
        return obs, reward, done, False, {}
