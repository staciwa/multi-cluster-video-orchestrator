import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from torch_geometric.loader import DataLoader
from python.models.baseline_gnn import BaselineGNN
from graph_builder import build_hetero_graph

def load_dataset(dataset_file: str):
    """Loads JSONL records and converts them to PyTorch Geometric HeteroData objects."""
    graphs = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                graph = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Skipped invalid line {line_num} in {dataset_file}: {e}")
            
    print(f"Loaded {len(graphs)} graphs from dataset {dataset_file}.")
    return graphs

def evaluate_gnn(dataset_path: str, model_path: str, batch_size: int = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Zero-Shot Evaluation on device: {device}")
    
    # 1. Load Data
    graphs = load_dataset(dataset_path)
    if not graphs:
        print("Error: No valid graphs loaded. Terminating evaluation.")
        return None, None
        
    val_loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    # 2. Model Initialization
    model = BaselineGNN(hidden_channels=128)
    # PyTorch 2.6+ Quirks: weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 3. Evaluation
    val_loss = 0
    val_mae = 0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            
            # If model returns dict for multiple edge types, extract the proper one
            if isinstance(preds, dict):
                key = ('pop', 'reaches', 'cluster')
                if key in preds:
                    preds = preds[key]
                else:
                    preds = next(iter(preds.values()))
                
            labels = batch['pop', 'reaches', 'cluster'].y
            
            # Prevents Loss Broadcasting Bug (as per AGENTS.md)
            assert preds.shape == labels.shape, f"Prediction shapes ({preds.shape}) and labels ({labels.shape}) do not match!"
            
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            
            loss = F.mse_loss(preds_flat, labels_flat)
            val_loss += loss.item() * labels_flat.size(0)
            val_mae += F.l1_loss(preds_flat, labels_flat, reduction='sum').item()
            val_samples += labels_flat.size(0)
            
    val_mse = val_loss / val_samples
    val_mae_avg = val_mae / val_samples
    
    print("-" * 50)
    print(f"Zero-Shot Results for model {model_path} on dataset {dataset_path}:")
    print(f"MSE: {val_mse:.6f}")
    print(f"MAE: {val_mae_avg:.6f}")
    print("-" * 50)
    
    # Saving results
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "zero_shot_gnn_uniform.json")
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_path,
            "model": model_path,
            "samples": val_samples,
            "mse": val_mse,
            "mae": val_mae_avg
        }, f, indent=4, ensure_ascii=False)
    print(f"Saved results to: {results_file}")
    
    return val_mse, val_mae_avg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zero-Shot GNN Evaluation")
    parser.add_argument("--dataset", type=str, default="dataset_100k_uniform/features_labels.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--model", type=str, default="best_baseline_gnn.pt", help="Path to the trained GNN model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    if os.path.exists(args.dataset) and os.path.exists(args.model):
        evaluate_gnn(args.dataset, args.model, batch_size=args.batch_size)
    else:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset {args.dataset} does not exist.")
        if not os.path.exists(args.model):
            print(f"Error: Model {args.model} does not exist.")
