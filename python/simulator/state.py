import json
import numpy as np
from typing import Dict, Set

class SimulatorState:
    """
    Central operational memory of the DES.
    Tracks active allocations, physical constraints, and the macro-scale Anchor.
    """
    def __init__(self, case_meta_path: str = None, case_dict: dict = None, anchor_labels: np.ndarray = None):
        meta = None
        self.case_dict = None
        if case_dict:
            meta = case_dict
            self.case_dict = case_dict
        elif case_meta_path:
            with open(case_meta_path, 'r') as f:
                meta = json.load(f)
            self.case_dict = meta
                
        if meta:
            self.num_clusters = len(meta.get("clusters", [0]*10)) # fallback
            self.num_pops = len(meta.get("pops", [0]*50))
            
            # Populate distances
            self.origin_distances = np.array(meta.get("originDistances", [0]*self.num_clusters), dtype=np.float32)
            self.pop_distances = np.array(meta.get("popDistances", np.zeros((self.num_pops, self.num_clusters))), dtype=np.float32)
            self.latency_per_10km_units = meta.get("latencyPer10kmUnits", 1.0)
            self.max_latency_units = meta.get("maxLatencyUnits", 10.0)
            
        else:
            # Fallback for empty init
            self.num_clusters = 10
            self.num_pops = 50
            self.origin_distances = np.zeros(self.num_clusters, dtype=np.float32)
            self.pop_distances = np.zeros((self.num_pops, self.num_clusters), dtype=np.float32)
            self.latency_per_10km_units = 1.0
            self.max_latency_units = 10.0
            
        # Hardware limits and real-time usage
        self.max_instances_per_cluster = np.zeros(self.num_clusters, dtype=np.int32)
        self.active_instances_per_cluster = np.zeros(self.num_clusters, dtype=np.int32)
        
        # Viewer mapping: demand_id -> cluster_id
        self.active_demands: Dict[int, int] = {}
        # Demands per cluster: cluster_id -> set of demand_ids
        self.demands_per_cluster: Dict[int, Set[int]] = {c: set() for c in range(self.num_clusters)}
        
        # Telemetry
        self.total_served = 0
        self.total_unserved = 0
        self.total_latency_penalty = 0.0
        self.total_transmission_cost = 0.0
        self.total_deployment_cost = 0.0
        
        # MACRO SCALE: Anchor baseline
        if anchor_labels is not None:
            self.anchor_allocation = anchor_labels
        else:
            self.anchor_allocation = np.zeros((self.num_pops, self.num_clusters), dtype=np.int32)

    def allocate_viewer(self, demand_id: int, cluster_id: int, pop_id: int) -> bool:
        """Attempt to allocate a new viewer to a cluster. Returns True if successful."""
        # 1. Basic QoS validation
        dist = self.pop_distances[pop_id, cluster_id] + self.origin_distances[cluster_id]
        latency = dist * self.latency_per_10km_units
        if latency > self.max_latency_units:
            return False # QoS breach
            
        # 2. Add demand
        self.active_demands[demand_id] = cluster_id
        self.demands_per_cluster[cluster_id].add(demand_id)
        self.total_served += 1
        return True

    def remove_viewer(self, demand_id: int):
        """Remove viewer and release cluster resources."""
        if demand_id in self.active_demands:
            c_id = self.active_demands.pop(demand_id)
            self.demands_per_cluster[c_id].remove(demand_id)
            
    def update_anchor(self, new_anchor: np.ndarray):
        """Called by the Macro Anchor solver to update the baseline."""
        self.anchor_allocation = new_anchor
