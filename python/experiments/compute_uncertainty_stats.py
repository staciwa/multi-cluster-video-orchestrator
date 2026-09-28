"""
Script computing uncertainty measures (std, CI) for experiments
lacking this statistic in the results:
  1. GNN scalability: N=20 repetitions per scale (inference loop time std)
  2. Distillation gap: MAE std per category from raw data
  3. MLP OOD: MAE std after N=10 episodes

Results saved to:
  results/uncertainty_scalability.json
  results/uncertainty_distillation.json
  results/uncertainty_mlp_ablation.json
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "minizinc"))

from python.models.baseline_gnn import BaselineGNN
from python.models.flat_mlp import FlatMLP
from minizinc.graph_builder import build_hetero_graph
from torch_geometric.loader import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(project_root, "models", "best_baseline_gnn.pt")
DATASET_TEST = os.path.join(project_root, "dataset_test", "features_labels.jsonl")
RESULTS_DIR = os.path.join(project_root, "results")


# ─────────────────────────────────────────────────────────────
# 1. 1. GNN Scalability: N=20 repetitions per scale
# ─────────────────────────────────────────────────────────────
def compute_scalability_uncertainty(n_repeats=20):
    """Measures the inference loop time std of GNN for different topology sizes."""
    print(f"\n=== [1/3] GNN Scalability (N={n_repeats} repetitions per scale) ===")

    model = BaselineGNN(hidden_channels=128)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # Load one example graph for each scale from dataset_test
    # (take the first instance, measure inference time N times)
    graphs_by_scale = {}
    with open(DATASET_TEST, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            n_pop = record["input"]["meta"].get("n_pop", len(record["input"]["meta"].get("pop_nodes", [])))
            if n_pop not in graphs_by_scale:
                graphs_by_scale[n_pop] = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
            if len(graphs_by_scale) >= 6:
                break

    results = {}
    for scale, graph in sorted(graphs_by_scale.items()):
        times = []
        # Warmup
        with torch.no_grad():
            batch = graph.to(DEVICE)
            _ = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                batch = graph.to(DEVICE)
                _ = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        mean_t = float(np.mean(times))
        std_t = float(np.std(times, ddof=1))
        results[scale] = {"mean_ms": round(mean_t, 3), "std_ms": round(std_t, 3), "n": n_repeats}
        print(f"  PoP={scale}: mean={mean_t:.2f}ms, std={std_t:.2f}ms")

    out_path = os.path.join(RESULTS_DIR, "uncertainty_scalability.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"  Saved: {out_path}")
    return results


# ─────────────────────────────────────────────────────────────
# 2. Distillation gap: std MAE per kategoria z raw danych
# ─────────────────────────────────────────────────────────────
def compute_distillation_uncertainty():
    """Computes MAE std and median per load category."""
    print(f"\n=== [2/3] Distillation gap — dispersion measures ===")

    model = BaselineGNN(hidden_channels=128)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    mae_low, mae_crit = [], []

    with open(DATASET_TEST, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            graph = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
            unserved = record["labels"].get("unserved", 0)

            with torch.no_grad():
                batch = graph.to(DEVICE)
                preds = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                labels = batch['pop', 'reaches', 'cluster'].y
                qos_mask = batch['pop', 'reaches', 'cluster'].edge_attr[:, 2] > 0
                preds_m = preds.view(-1)[qos_mask]
                labels_m = labels.view(-1)[qos_mask]
                if len(labels_m) == 0:
                    continue
                mae = F.l1_loss(preds_m, labels_m, reduction='mean').item()

            if unserved == 0:
                mae_low.append(mae)
            else:
                mae_crit.append(mae)

    def stats(arr, label):
        a = np.array(arr)
        r = {
            "n": len(a),
            "mean": round(float(np.mean(a)), 6),
            "std": round(float(np.std(a, ddof=1)), 6),
            "median": round(float(np.median(a)), 6),
            "p25": round(float(np.percentile(a, 25)), 6),
            "p75": round(float(np.percentile(a, 75)), 6),
            "p95": round(float(np.percentile(a, 95)), 6),
        }
        print(f"  {label}: N={r['n']}, mean={r['mean']:.4f}, std={r['std']:.4f}, "
              f"median={r['median']:.4f}, p95={r['p95']:.4f}")
        return r

    results = {
        "low_load": stats(mae_low, "Low load (unserved=0)"),
        "critical_load": stats(mae_crit, "Critical saturation (unserved>0)"),
    }

    out_path = os.path.join(RESULTS_DIR, "uncertainty_distillation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"  Saved: {out_path}")
    return results


# ─────────────────────────────────────────────────────────────
# 3. MLP OOD: MAE std after N=10 episodes on 100x50 topology
# ─────────────────────────────────────────────────────────────
def compute_mlp_ablation_uncertainty():
    """Computes MAE std for MLP in-dist and OOD (zero-shot 100x50)."""
    print(f"\n=== [3/3] MLP ablation — MAE std ===")

    # Load MLP in-distribution results from mlp_training.json
    with open(os.path.join(RESULTS_DIR, "mlp_training.json")) as f:
        mlp_tr = json.load(f)
    # Last epoch = MAE_val
    last_val_maes = [e["val_mae"] for e in mlp_tr[-5:]]  # ostatnie 5 epok
    print(f"  MLP in-dist val_mae (ostatnie 5 epok): {[round(x,4) for x in last_val_maes]}")
    print(f"    mean={np.mean(last_val_maes):.4f}, std={np.std(last_val_maes,ddof=1):.4f}")

    # Load zero_shot_gnn_results.json for GNN
    with open(os.path.join(RESULTS_DIR, "zero_shot_gnn_results.json")) as f:
        zs = json.load(f)
    print(f"  GNN OOD (100×50): mean_mae={zs[0]['mean_mae']}, std_mae={zs[0]['std_mae']}, N={zs[0]['episodes']}")

    # MLP OOD — must be computed from the OOD set
    # Looking for MLP model
    mlp_path = os.path.join(project_root, "models", "best_flat_mlp.pt")
    ood_dataset = os.path.join(project_root, "dataset_ood", "features_labels.jsonl")

    results = {
        "mlp_in_dist": {
            "mean_val_mae": round(float(np.mean(last_val_maes)), 6),
            "std_val_mae": round(float(np.std(last_val_maes, ddof=1)), 6),
            "note": "std after the last 5 training epochs (stability approximation, not test set error)",
        },
        "gnn_ood_100x50": {
            "n_episodes": zs[0]["episodes"],
            "mean_mae": zs[0]["mean_mae"],
            "std_mae": zs[0]["std_mae"],
        },
    }

    if not os.path.exists(mlp_path):
        print(f"  MISSING MLP model ({mlp_path}) — skipping MLP OOD evaluation")
        results["mlp_ood_100x50"] = {"error": "missing model file"}
        out_path = os.path.join(RESULTS_DIR, "uncertainty_mlp_ablation.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"  Saved: {out_path}")
        return results

    if not os.path.exists(ood_dataset):
        print(f"  MISSING OOD dataset ({ood_dataset}) — skipping MLP OOD evaluation")
        results["mlp_ood_100x50"] = {"error": "missing OOD dataset"}
        out_path = os.path.join(RESULTS_DIR, "uncertainty_mlp_ablation.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"  Saved: {out_path}")
        return results

    model_mlp = FlatMLP()
    model_mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
    model_mlp.to(DEVICE)
    model_mlp.eval()

    mae_list = []
    with open(ood_dataset, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            graph = build_hetero_graph(record["input"]["meta"], labels=record["labels"])
            with torch.no_grad():
                batch = graph.to(DEVICE)
                preds = model_mlp(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                labels = batch['pop', 'reaches', 'cluster'].y
                qos_mask = batch['pop', 'reaches', 'cluster'].edge_attr[:, 2] > 0
                preds_m = preds.view(-1)[qos_mask]
                labels_m = labels.view(-1)[qos_mask]
                if len(labels_m) == 0:
                    continue
                mae = F.l1_loss(preds_m, labels_m, reduction='mean').item()
                mae_list.append(mae)

    results["mlp_ood_100x50"] = {
        "n": len(mae_list),
        "mean_mae": round(float(np.mean(mae_list)), 6),
        "std_mae": round(float(np.std(mae_list, ddof=1)), 6),
        "median_mae": round(float(np.median(mae_list)), 6),
    }
    print(f"  MLP OOD: N={len(mae_list)}, mean={results['mlp_ood_100x50']['mean_mae']:.4f}, "
          f"std={results['mlp_ood_100x50']['std_mae']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "uncertainty_mlp_ablation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"  Saved: {out_path}")
    return results


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Computing uncertainty measures for experiments from chapter 5...")
    print(f"Device: {DEVICE}")

    r1 = compute_scalability_uncertainty(n_repeats=20)
    r2 = compute_distillation_uncertainty()
    r3 = compute_mlp_ablation_uncertainty()

    print("\n=== READY ===")
    print("Generated files:")
    print("  results/uncertainty_scalability.json")
    print("  results/uncertainty_distillation.json")
    print("  results/uncertainty_mlp_ablation.json")
