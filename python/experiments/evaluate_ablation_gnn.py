import os
import sys
import json
import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from torch_geometric.loader import DataLoader
from python.models.baseline_gnn import BaselineGNN
from python.train_baseline import load_dataset

def apply_ablation(data):
    """Application of spatial features ablation on graph or batch."""
    for edge_type in data.edge_types:
        attr = data[edge_type].edge_attr
        if attr is not None:
            assert attr.size(1) >= 3, f"Expected at least 3 edge features, got {attr.size(1)}"
            attr[:, 0] = 0.0 # dist
            attr[:, 1] = 0.0 # latency
            attr[:, 2] = 1.0 # qos (assuming ideal connection in ablation)
    return data

def evaluate_model(model, loader, device, ablate=False):
    model.eval()
    total_mae = 0
    total_mse = 0
    total_edges = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            if ablate:
                batch = batch.clone() # avoid modifying the original
                batch = apply_ablation(batch)

            preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            if isinstance(preds, dict):
                preds = preds[('pop', 'reaches', 'cluster')]
                
            labels = batch['pop', 'reaches', 'cluster'].y
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            
            total_mse += F.mse_loss(preds_flat, labels_flat).item() * labels_flat.size(0)
            total_mae += F.l1_loss(preds_flat, labels_flat, reduction='sum').item()
            total_edges += labels_flat.size(0)
            
    return total_mae / total_edges, total_mse / total_edges

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--baseline-model", type=str, default="models/best_baseline_gnn.pt")
    parser.add_argument("--ablation-model", type=str, default="models/best_ablation_gnn.pt")
    parser.add_argument("--output", type=str, default="results/gnn_ablation_comparison.json")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation on device: {device}")
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset {args.dataset} does not exist.")
        return

    import random
    random.seed(42)
    graphs = load_dataset(args.dataset)
    random.shuffle(graphs)
    split_idx = int(len(graphs) * 0.8)
    test_graphs = graphs[split_idx:]
    print(f"Evaluation (test) set contains {len(test_graphs)} graphs.")
    
    loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    # 1. Baseline Model
    baseline_mae, baseline_mse = None, None
    if os.path.exists(args.baseline_model):
        print("Evaluating baseline model (full features)...")
        baseline_model = BaselineGNN(hidden_channels=128)
        baseline_model.load_state_dict(torch.load(args.baseline_model, map_location=device, weights_only=True))
        baseline_model.to(device)
        baseline_mae, baseline_mse = evaluate_model(baseline_model, loader, device, ablate=False)
        print(f"Baseline -> MAE: {baseline_mae:.4f}, MSE: {baseline_mse:.4f}")
    else:
        print(f"Warning: Baseline model {args.baseline_model} does not exist.")

    # 2. Ablation Model
    ablation_mae, ablation_mse = None, None
    if os.path.exists(args.ablation_model):
        print("Evaluating ablation model (zeroed spatial features)...")
        ablation_model = BaselineGNN(hidden_channels=128)
        ablation_model.load_state_dict(torch.load(args.ablation_model, map_location=device, weights_only=True))
        ablation_model.to(device)
        ablation_mae, ablation_mse = evaluate_model(ablation_model, loader, device, ablate=True)
        print(f"Ablation -> MAE: {ablation_mae:.4f}, MSE: {ablation_mse:.4f}")
    else:
        print(f"Error: Ablation model {args.ablation_model} does not exist.")
        return

    # Save results
    results = {
        "baseline": {
            "mae": baseline_mae,
            "mse": baseline_mse
        },
        "ablation": {
            "mae": ablation_mae,
            "mse": ablation_mse
        },
        "degradation_mae": (ablation_mae - baseline_mae) if baseline_mae else None,
        "degradation_percent": ((ablation_mae / baseline_mae) - 1.0) * 100 if baseline_mae else None
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nComparison results saved to: {args.output}")

if __name__ == "__main__":
    main()
