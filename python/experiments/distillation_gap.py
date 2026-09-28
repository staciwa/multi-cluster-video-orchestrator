import os
import sys
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from python.models.baseline_gnn import BaselineGNN
from minizinc.graph_builder import build_hetero_graph
from torch_geometric.loader import DataLoader

def analyze_distillation_gap(dataset_path: str, model_path: str, output: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Distillation Gap analysis on device: {device}")
    
    # Model initialization
    model = BaselineGNN(hidden_channels=128)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    graphs = []
    unserved_labels = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            graph = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
            
            # Extraction of unserved demands to evaluate the "difficulty" of the case
            unserved = record["labels"].get("unserved", 0)
            graphs.append(graph)
            unserved_labels.append(unserved)
            
    print(f"Loaded {len(graphs)} cases from {dataset_path}.")
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    
    # Error categorization
    # Group 1: No rejections (unserved == 0) - low/medium load, simpler case
    # Group 2: Rejections occur (unserved > 0) - critical load, complex MILP compromise
    mae_no_congestion = []
    mae_congestion = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            labels = batch['pop', 'reaches', 'cluster'].y
            
            # QoS masking same as in training (fair MAE)
            qos_mask = batch['pop', 'reaches', 'cluster'].edge_attr[:, 2] > 0
            preds_masked = preds.view(-1)[qos_mask]
            labels_masked = labels.view(-1)[qos_mask]
            
            if len(labels_masked) == 0:
                continue
                
            mae = F.l1_loss(preds_masked, labels_masked, reduction='mean').item()
            unserved = unserved_labels[i]
            
            if unserved == 0:
                mae_no_congestion.append(mae)
            else:
                mae_congestion.append(mae)
                
    avg_no_cong = np.mean(mae_no_congestion) if mae_no_congestion else 0.0
    avg_cong = np.mean(mae_congestion) if mae_congestion else 0.0
    
    print(f"--- Distillation Gap Results ---")
    print(f"Low/medium load (Unserved = 0): {len(mae_no_congestion)} cases | Mean MAE = {avg_no_cong:.4f}")
    print(f"Critical load           (Unserved > 0): {len(mae_congestion)} cases | Mean MAE = {avg_cong:.4f}")
    
    degradation = 0.0
    if avg_no_cong > 0:
        degradation = ((avg_cong - avg_no_cong) / avg_no_cong) * 100
        print(f"GNN precision degradation at peak: +{degradation:.1f}%")
        
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "low_load": {
                "count": len(mae_no_congestion),
                "avg_mae": avg_no_cong
            },
            "critical_load": {
                "count": len(mae_congestion),
                "avg_mae": avg_cong
            },
            "degradation_percent": degradation
        }, f, indent=4)
        
    print(f"Results saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN prediction error analysis (Distillation Gap)")
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--output", type=str, default="results/distillation_gap.json")
    args = parser.parse_args()
    
    analyze_distillation_gap(args.dataset, args.model, args.output)
