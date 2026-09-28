import gymnasium as gym
from gymnasium import spaces
import numpy as np
import heapq
import itertools
import random

from python.simulator.engine import SimulatorEngine
from python.simulator.state import SimulatorState
from python.simulator.events import ViewerArrival, ViewerDeparture, MacroAnchorUpdate
from python.simulator.main import EventGenerator
import os
import sys
sys.path.append(os.path.join(os.path.abspath("."), "minizinc"))
from erlang_utils import erlang_b, calculate_traffic_per_demand

class SrsBaselineEnv(gym.Env):
    """
    Loose environment (Baseline) for pure RL Agent (no Teacher Forcing, no MILP oracle).
    Cluster constraints are based solely on physical capacity calculated from the Erlang model.
    """
    def __init__(self, dataset_file: str, sim_hours: float = 24.0, macro_interval: float = 0.25, deterministic: bool = False):
        super().__init__()
        self.dataset_file = dataset_file
        self.sim_hours = sim_hours
        self.macro_interval = macro_interval
        self.deterministic = deterministic
        
        self.line_offsets = []
        with open(self.dataset_file, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.line_offsets.append(offset)
                offset += len(line)
                
        first_case = self._load_case(self.line_offsets[0])
        temp_state = SimulatorState(case_dict=first_case["input"]["meta"])
        self.num_pops = temp_state.num_pops
        self.num_clusters = temp_state.num_clusters
        
        self.action_space = spaces.Discrete(self.num_clusters + 1)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(103,), dtype=np.float32)
        assert self.num_clusters <= 20, "Requires a maximum of 20 clusters."
        self.current_arrival_event = None
        
    def _load_case(self, offset):
        import json
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return json.loads(f.readline())
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.engine = SimulatorEngine()
        if self.deterministic:
            self.current_case = self._load_case(self.line_offsets[0])
        else:
            self.current_case = self._load_case(random.choice(self.line_offsets))
        
        meta = self.current_case["input"]["meta"]
        
        # Empty labels, baseline does not use ground truth
        anchor_labels = np.zeros((self.num_pops, self.num_clusters), dtype=np.int32)
        self.state = SimulatorState(case_dict=meta, anchor_labels=anchor_labels)
        self.event_counter = itertools.count()
        self.demand_counter = itertools.count()
        self.edge_load = np.zeros((self.num_pops, self.num_clusters), dtype=np.int32)
        
        # Calculation of global physical capacity of clusters (instead of A* allocation)
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

        raw_demands = meta.get("popDemands", [0]*self.num_pops)
        self.base_pop_demands = [int(d) for d in raw_demands]
        self.current_pop_demands = list(self.base_pop_demands)
        pop_lambdas = [float(d) for d in self.current_pop_demands]
        
        # Retrieve service intensity from configuration
        if "erlang" in meta and "serviceRate" in meta["erlang"]:
            self.service_mu = float(meta["erlang"]["serviceRate"])
        else:
            self.service_mu = 1.0 
        
        self.generator = EventGenerator(self.engine, pop_lambdas, self.sim_hours, self.event_counter, self.demand_counter, deterministic=self.deterministic)
        
        def handle_departure(event: ViewerDeparture):
            self.state.remove_viewer(event.demand_id)
            self.edge_load[event.pop_id, event.cluster_id] -= 1
                
        def handle_macro_update(event: MacroAnchorUpdate):
            if event.time > 0.0:
                # In the loose environment, we randomly select a new scenario.
                # NOTE: The dataset guarantees a constant topology (Symmetry Breaking) for all generated 100k samples,
                # so the "Double Source of Truth" phenomenon and topology truncation are not an issue - only demand vectors change.
                new_case = self._load_case(random.choice(self.line_offsets))
                new_raw_demands = new_case["input"]["meta"].get("popDemands", [0]*self.num_pops)
                self.current_pop_demands = [int(d) for d in new_raw_demands]
                self.generator.pop_lambdas = [float(d) for d in self.current_pop_demands]
                
            next_time = event.time + self.macro_interval
            if next_time <= self.sim_hours:
                self.engine.schedule(MacroAnchorUpdate(time=next_time, event_id=next(self.event_counter)))
                
        self.engine.register_handler(ViewerDeparture, handle_departure)
        self.engine.register_handler(MacroAnchorUpdate, handle_macro_update)
        
        self.engine.schedule(MacroAnchorUpdate(time=0.0, event_id=next(self.event_counter)))
        self.generator.bootstrap_arrivals()
        
        return self._step_to_next_arrival(), {}
        
    def _step_to_next_arrival(self):
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
            
        f_t = np.array([pop_id / float(max(1, self.num_pops))], dtype=np.float32)
        
        d_t = np.zeros(20, dtype=np.float32)
        for c_id in range(min(20, self.num_clusters)):
            dist = self.state.pop_distances[pop_id, c_id] + self.state.origin_distances[c_id]
            latency = dist * self.state.latency_per_10km_units
            d_t[c_id] = min(1.0, latency / max(1.0, self.state.max_latency_units))
            
        r_t = np.zeros(22, dtype=np.float32)
        cluster_load = np.sum(self.edge_load, axis=0) 
        for c_id in range(min(20, self.num_clusters)):
            capacity_limit = float(self.cluster_max_capacity[c_id])
            r_t[c_id] = min(1.0, float(cluster_load[c_id]) / max(1.0, capacity_limit))
            
        budget_t = np.zeros(20, dtype=np.float32)
        for c_id in range(min(20, self.num_clusters)):
            anc = float(self.cluster_max_capacity[c_id])
            cur = float(cluster_load[c_id])
            budget_t[c_id] = np.clip((anc - cur) / max(1.0, anc), -1.0, 1.0)
            
        last_macro_time = (self.engine.current_time // self.macro_interval) * self.macro_interval
        r_t[20] = min(1.0, float((self.engine.current_time - last_macro_time) / self.macro_interval))
        r_t[21] = min(1.0, float(self.engine.current_time / self.sim_hours))
        
        # Empty delta_t (40 dimensions)
        # NOTE: We leave an artificial 40 zeros to preserve the total
        # observation dimension of 103 for the RL agent. This way we don't have to write a separate
        # neural network architecture (MLP with a different shape) just for the Baseline model.
        delta_t = np.zeros(40, dtype=np.float32)
        
        obs = np.concatenate([f_t, d_t, r_t, budget_t, delta_t]).astype(np.float32)
        return obs
        
    def valid_action_mask(self):
        mask = np.zeros(self.num_clusters + 1, dtype=np.int8)
        if not self.current_arrival_event:
            return mask
            
        pop_id = self.current_arrival_event.pop_id
        cluster_load = np.sum(self.edge_load, axis=0)
        
        for c_id in range(self.num_clusters):
            dist = self.state.pop_distances[pop_id, c_id] + self.state.origin_distances[c_id]
            latency = dist * self.state.latency_per_10km_units
            
            if latency <= self.state.max_latency_units and cluster_load[c_id] < self.cluster_max_capacity[c_id]:
                mask[c_id] = 1
                
        mask[self.num_clusters] = 1 # Rejection always allowed
        return mask
        
    def step(self, action: int):
        if self.current_arrival_event is None:
            return self._get_obs(), 0.0, True, False, {}
            
        cluster_id = int(action)
        pop_id = self.current_arrival_event.pop_id
        demand_id = self.current_arrival_event.demand_id
        
        reward = 0.0
        
        if cluster_id == self.num_clusters:
            # Action: REJECT REQUEST
            mask = self.valid_action_mask()
            # The last index is reject. If the sum of the others is 0, there is really no space.
            if np.sum(mask[:-1]) == 0:
                reward = 0.1 # Reward for saving from overflow
            else:
                reward = -1.0 # Penalty for unjustified rejection
                
            self.state.total_unserved += 1
        else:
            # NOTE: `allocate_viewer` in `state.py` does not itself verify `cluster_max_capacity`.
            # However, the system is watertight - the action mask above physically PREVENTS the agent from selecting a cluster
            # that lacks space (ActionMasking guarantees rejection of the node upfront).
            success = self.state.allocate_viewer(demand_id, cluster_id, pop_id)
            if success:
                self.edge_load[pop_id, cluster_id] += 1
                cluster_load = np.sum(self.edge_load[:, cluster_id])
                
                if cluster_load <= self.cluster_max_capacity[cluster_id]:
                    # Base reward for service
                    reward = 1.0 
                else:
                    reward = -1.0
                
                service_time = random.expovariate(self.service_mu)
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
                reward = -1.0 # Penalty for violating QoS constraints (allocation rejected in state)
            
        self.generator.schedule_next_arrival(pop_id, self.engine.current_time)
        obs = self._step_to_next_arrival()
        done = self.current_arrival_event is None
        
        return obs, reward, done, False, {}
