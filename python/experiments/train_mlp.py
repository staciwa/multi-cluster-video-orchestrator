import os
import sys
import json
import random
import torch
import torch.nn.functional as F
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from torch_geometric.loader import DataLoader
from python.models.baseline_mlp import BaselineMLP
from minizinc.graph_builder import build_hetero_graph

def load_dataset(dataset_file: str):
    """Loads JSONL records and converts them to PyTorch Geometric HeteroData objects."""
    graphs = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            graph = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
            graphs.append(graph)
            
    print(f"Loaded {len(graphs)} graphs from dataset {dataset_file}.")
    return graphs

def train_mlp_model(dataset_path: str, epochs: int = 100, batch_size: int = 16, output_model: str = "models/best_baseline_mlp.pt", output_results: str = "results/mlp_training.json"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training MLP on device: {device}")
    
    # 1. Load Data
    graphs = load_dataset(dataset_path)
    random.seed(42)
    torch.manual_seed(42)
    random.shuffle(graphs)
    
    # Graph size validation (MLP requires fixed size)
    num_pops = len(graphs[0]['pop'].x)
    num_clusters = len(graphs[0]['cluster'].x)
    for g in graphs:
        if len(g['pop'].x) != num_pops or len(g['cluster'].x) != num_clusters:
            print(f"ERROR: Dataset {dataset_path} contains graphs with different node sizes. MLP requires fixed size.")
            sys.exit(1)
            
    print(f"MLP dimensions: PoPs = {num_pops}, Clusters = {num_clusters}")
    
    # Validation split (80/20)
    split_idx = int(len(graphs) * 0.8)
    train_loader = DataLoader(graphs[:split_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(graphs[split_idx:], batch_size=batch_size, shuffle=False)
    
    # 2. Model Initialization
    model = BaselineMLP(num_pops=num_pops, num_clusters=num_clusters, hidden_dim=256)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    training_history = []
    
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    os.makedirs(os.path.dirname(output_results), exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_mae = 0
        total_edges = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            labels = batch['pop', 'reaches', 'cluster'].y
            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)
            
            loss = F.mse_loss(preds_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * labels_flat.size(0)
            total_mae += F.l1_loss(preds_flat, labels_flat, reduction='sum').item()
            total_edges += labels_flat.size(0)
            
        train_mse = total_loss / total_edges
        train_mae = total_mae / total_edges
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        val_edges = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                labels = batch['pop', 'reaches', 'cluster'].y
                preds_flat = preds.view(-1)
                labels_flat = labels.view(-1)
                
                loss = F.mse_loss(preds_flat, labels_flat)
                val_loss += loss.item() * labels_flat.size(0)
                val_mae += F.l1_loss(preds_flat, labels_flat, reduction='sum').item()
                val_edges += labels_flat.size(0)
                
        val_mse = val_loss / val_edges
        val_mae_metric = val_mae / val_edges
        
        print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f} | Val MSE: {val_mse:.4f}, MAE: {val_mae_metric:.4f}")
        
        training_history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae_metric
        })
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), output_model)
            print(f" -> Saved new best MLP model (Val MSE: {val_mse:.4f}) to {output_model}")
            
    # Save history
    with open(output_results, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)
    print(f"MLP training completed. Results saved to {output_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of a flat MLP for Ablation Study")
    parser.add_argument("--dataset", type=str, default="dataset_test/features_labels.jsonl")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    train_mlp_model(args.dataset, epochs=args.epochs)
