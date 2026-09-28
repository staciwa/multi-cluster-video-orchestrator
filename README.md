# Multi-Cluster Video Streaming Orchestrator

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![MiniZinc](https://img.shields.io/badge/MiniZinc-2.9.7-orange.svg)](https://www.minizinc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid orchestrator for real-time video streaming (RTC) operating in federated environments (e.g., Kubernetes managed by **Karmada**). The system combines the precision of Operations Research (MiniZinc, HiGHS) with the speed of Artificial Intelligence – Graph Neural Networks (GNN) and Reinforcement Learning (PPO).

This project minimizes operational costs while maintaining strict Quality of Service (QoS) constraints based on the Erlang B model.

## 🚀 Key Features
- **MILP Model (Erlang B)** – A formal mathematical model for video streaming constraints implemented in MiniZinc.
- **Parallel Data Generation** – A module using the HiGHS solver to create training datasets with optimality guarantees (Ground Truth).
- **Heterogeneous Graph Neural Networks (GNN)** – A model generalizing the resource allocation problem for networks of arbitrary topologies at a macro scale.
- **PPO Agent (Reinforcement Learning)** – A fast decision-maker for real-time micro-scale orchestrations.
- **K8s/Karmada Integration** – Automatic deployment of resources across edge/cloud clusters based on the computed policies.

## 📁 Repository Structure
- `python/` – Machine Learning code (PyTorch Geometric, Stable-Baselines3), discrete-event simulator (DES), and the orchestrator module.
- `minizinc/` – Optimization models in MiniZinc and scripts for generating REFERENCE topologies.
- `experiments/` – Output directory for experiment results and Karmada deployment logs.
- `pyproject.toml` – Python environment configuration using the fast `uv` package manager.
- `Dockerfile` – A ready-to-use image for running the entire environment in an isolated container.

## ⚙️ Prerequisites
- **Python 3.12** 
- **[uv](https://github.com/astral-sh/uv)** – fast Python package manager
- **[MiniZinc 2.9.7+](https://www.minizinc.org/)** with the **HiGHS** solver – must be available in `PATH`
- *(Optional)* **Kubernetes cluster with Karmada** – required only for multi-cluster deployment experiments

## 🛠 Installation

The recommended way to manage dependencies is using [uv](https://github.com/astral-sh/uv).

```bash
# 1. Clone the repository
git clone https://github.com/staciwa/multi-cluster-video-orchestrator.git
cd multi-cluster-video-orchestrator

# 2. Sync dependencies
uv sync

# Activate the virtual environment:
source .venv/bin/activate
```

Alternatively, you can use **Docker**:
```bash
docker build -t orchestrator .
docker run -it orchestrator
```

## 📊 Quickstart (Reproducibility)

The research pipeline consists of several stages: data generation, training, and evaluation.

### 1. Data Generation
Generate problem instances (REFERENCE topology) and label them using the parallel solver:

```bash
# Generation
uv run python minizinc/generate_reference_dataset.py --count 500 --output-dir dataset_raw

# Solving (labeling) using e.g., 6 workers
uv run python minizinc/build_supervision_dataset_parallel.py --input-dir dataset_raw --output-dir dataset_labeled --workers 6
```

### 2. Machine Learning Training
Train the GNN model (macro-scale):
```bash
uv run python python/train_baseline.py --dataset dataset_labeled/features_labels.jsonl --epochs 40
```

Train the PPO agent (micro-scale):
```bash
uv run python python/rl/train_ppo.py --dataset_dir dataset_labeled --timesteps 1000000
```

### 3. Experiments & Orchestration
Run evaluation experiments (e.g., OOD robustness, flash crowd, scalability) from `python/experiments/`:
```bash
uv run python python/experiments/heavy_tailed.py
uv run python python/experiments/flash_crowd.py
```

Deploy to a multi-cluster Kubernetes/Karmada environment (requires a configured `KUBECONFIG`):
```bash
uv run python python/run_karmada_experiments.py
```

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
