"""
Conversion of generated REFERENCE cases (JSON) to PyTorch Geometric graphs.
Preparation of the environment for GNN and PPO agent.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

def build_hetero_graph(case_meta: Dict[str, Any], labels: Optional[Dict[str, Any]] = None) -> HeteroData:
    """
    Konwertuje metadane przypadku i opcjonalne etykiety na heterogeniczny graf (HeteroData).
    Implements Aggregated Variant: prediction of assigned demands on PoP -> Cluster edges (Edge Regression).
    """
    data = HeteroData()
    
    # 1. NODES: CLUSTER
    # Cechy klastra: [vCPU, RAM, DeploymentCost, is_central, is_edge]
    clusters = case_meta["clusters"]
    num_clusters = len(clusters)
    cluster_features = []
    
    vcpus = [res[0] for res in case_meta["clusterResources"]]
    rams = [res[1] for res in case_meta["clusterResources"]]
    costs = case_meta["clusterDeploymentCosts"]
    central_mask = [cluster["kind"] == "central" for cluster in clusters]
    edge_mask = [cluster["kind"] == "edge" for cluster in clusters]
    
    for vcpu, ram, cost, is_central, is_edge in zip(vcpus, rams, costs, central_mask, edge_mask):
        cluster_features.append([
            float(vcpu) / 64.0,
            float(ram) / 131072.0,
            float(cost) / 10.0,
            1.0 if is_central else 0.0,
            1.0 if is_edge else 0.0
        ])
        
    data['cluster'].x = torch.tensor(cluster_features, dtype=torch.float)
    
    # 2. NODES: POP
    # Cechy PoP: [lambda, mu, N (demands count)]
    num_pops = len(case_meta["pops"])
    pop_features = []
    
    for lam, mu, n_demands in zip(case_meta["popLambda"], case_meta["popMu"], case_meta["popDemands"]):
        pop_features.append([float(lam) / 100.0, float(mu) / 10.0, float(n_demands) / 100.0])
        
    data['pop'].x = torch.tensor(pop_features, dtype=torch.float)
    
    num_demands = sum(case_meta["popDemands"])
        
    # GLOBAL ATTRIBUTES: Erlang and SRS parameters
    erlang = case_meta.get("erlang", {})
    data.blockage_prob = torch.tensor([erlang.get("blockageProbability", 0.01)], dtype=torch.float)
    data.srs_capacity = torch.tensor([case_meta.get("srsCapacity", 20)], dtype=torch.float)
    data.total_demands = torch.tensor([num_demands], dtype=torch.float)
    data.total_pops = torch.tensor([num_pops], dtype=torch.float)
    data.total_clusters = torch.tensor([num_clusters], dtype=torch.float)

    # 3. EDGES: POP -> CLUSTER and CLUSTER -> POP
    # Edge features: [distance, latency_units, qos_continuous]
    p_c_edge_index = []
    p_c_edge_attr = []
    c_p_edge_index = []
    p_c_edge_y = []
    
    latency_per_10km_units = case_meta["latencyPer10kmUnits"]
    max_latency_units = case_meta["maxLatencyUnits"]
    origin_distances = case_meta.get("originDistances", [0]*num_clusters)
    
    for p_idx in range(num_pops):
        distances = case_meta["popDistances"][p_idx]
        for c_idx in range(num_clusters):
            dist = distances[c_idx] + origin_distances[c_idx]
            latency_units = dist * latency_per_10km_units
            
            # Continuous QoS feature: 1.0 for zero latency, drops to 0.0 at MaxLatency boundary
            qos_continuous = max(0.0, 1.0 - (latency_units / max_latency_units)) if max_latency_units > 0 else 0.0
            
            p_c_edge_index.append([p_idx, c_idx])
            c_p_edge_index.append([c_idx, p_idx])
            p_c_edge_attr.append([float(dist) / 1000.0, float(latency_units) / 100.0, float(qos_continuous)])
            
            if labels is not None:
                p_c_edge_y.append([float(labels['placement'][p_idx][c_idx]) / 100.0])
            
    if p_c_edge_index:
        data['pop', 'reaches', 'cluster'].edge_index = torch.tensor(p_c_edge_index, dtype=torch.long).t().contiguous()
        data['pop', 'reaches', 'cluster'].edge_attr = torch.tensor(p_c_edge_attr, dtype=torch.float)
        
        # Rewersy
        data['cluster', 'rev_reaches', 'pop'].edge_index = torch.tensor(c_p_edge_index, dtype=torch.long).t().contiguous()
        data['cluster', 'rev_reaches', 'pop'].edge_attr = torch.tensor(p_c_edge_attr, dtype=torch.float)
        
        if labels is not None:
            data['pop', 'reaches', 'cluster'].y = torch.tensor(p_c_edge_y, dtype=torch.float)
    else:
        data['pop', 'reaches', 'cluster'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['pop', 'reaches', 'cluster'].edge_attr = torch.empty((0, 3), dtype=torch.float)
        data['cluster', 'rev_reaches', 'pop'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['cluster', 'rev_reaches', 'pop'].edge_attr = torch.empty((0, 3), dtype=torch.float)
        if labels is not None:
            data['pop', 'reaches', 'cluster'].y = torch.empty((0, 1), dtype=torch.float)
    
    # 4. LABELS (for supervised learning - Aggregated Variant)
    if labels is not None:
        # Note: labels for assignments are already saved in data['pop', 'reaches', 'cluster'].y
        
        # Auxiliary Target: number of instances per cluster (often helps representations converge)
        data['cluster'].y = torch.tensor(labels['instances'], dtype=torch.float)
        
        # GlobalScore (optional for later calculating Value Function in PPO/Actor-Critic)
        data.global_score = torch.tensor([labels['scores']['globalScore']], dtype=torch.float)
        
    return data

def load_dataset(jsonl_path: Path) -> List[HeteroData]:
    """Loads the entire dataset and builds a list of graphs."""
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row["status"] == "ok":
                meta = row["input"]["meta"]
                labels = row["labels"]
                graph = build_hetero_graph(meta, labels)
                dataset.append(graph)
    return dataset