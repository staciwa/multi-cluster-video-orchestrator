#!/usr/bin/env python3
"""
Automated execution module for Karmada deployment experiments.
All resources are strictly isolated in a dedicated namespace: 'srs-orchestration',
so as not to affect other running services in the cluster in any way.
"""

import os
import sys
import random
import time
import argparse
import json
import datetime
from pathlib import Path
from enum import Enum
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "minizinc"))

import minizinc
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from erlang_utils import generate_erlang_table

# Environment parameters
TEST_NAMESPACE = "srs-orchestration"
APP_LABEL = "srs-thesis"
KUBECONFIG_CONTEXT = "karmada-apiserver"
MODEL_PATH = str(PROJECT_ROOT / "minizinc" / "model_erlang.mzn")
SOLVER_NAME = "highs"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "karmada_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 4 Physical Clusters in the Karmada Federation
PHYSICAL_CLUSTERS = [
    {"name": "cluster-central-1", "kind": "central", "vcpu": 36, "ram_mb": 65536, "cost": 100},
    {"name": "cluster-edge-1", "kind": "edge", "vcpu": 36, "ram_mb": 65536, "cost": 250},
    {"name": "cluster-edge-2", "kind": "edge", "vcpu": 36, "ram_mb": 65536, "cost": 250},
    {"name": "cluster-edge-3", "kind": "edge", "vcpu": 12, "ram_mb": 32768, "cost": 250},
]

# PoP (Points of Presence generating traffic)
POPS = [
    {"name": "Warszawa", "coords": (52.23, 21.01)},
    {"name": "Poznan", "coords": (52.41, 16.93)},
    {"name": "Krakow", "coords": (50.06, 19.94)},
    {"name": "Gdansk", "coords": (54.35, 18.65)},
    {"name": "Wroclaw", "coords": (51.11, 17.04)},
    {"name": "Katowice", "coords": (50.26, 19.02)},
]

# Latency matrix (units: 1 unit = 0.1 ms RTT):
# cluster-central-1 (Central DC), cluster-edge-1 (Edge 1), cluster-edge-2 (Edge 2), cluster-edge-3 (Edge 3)
POP_DISTANCES = [
    # Warszawa:  central-1(2ms), edge-1(8ms),  edge-2(18ms), edge-3(6ms)
    [20, 80, 180, 60],
    # Poznań:    central-1(18ms), edge-1(12ms), edge-2(2ms),  edge-3(14ms)
    [180, 120, 20, 140],
    # Kraków:    central-1(16ms), edge-1(14ms), edge-2(22ms), edge-3(10ms)
    [160, 140, 220, 100],
    # Gdańsk:    central-1(17ms), edge-1(15ms), edge-2(16ms), edge-3(18ms)
    [170, 150, 160, 180],
    # Wrocław:   central-1(19ms), edge-1(10ms), edge-2(11ms), edge-3(12ms)
    [190, 100, 110, 120],
    # Katowice:  central-1(15ms), edge-1(12ms), edge-2(20ms), edge-3(8ms)
    [150, 120, 200, 80],
]

def init_k8s():
    """Initializes the Kubernetes client with the specified karmada-apiserver context."""
    try:
        contexts, active_context = config.list_kube_config_contexts()
        context_names = [c['name'] for c in contexts]
        if KUBECONFIG_CONTEXT in context_names:
            config.load_kube_config(context=KUBECONFIG_CONTEXT)
            print(f"[K8S-INIT] Connected to context: {KUBECONFIG_CONTEXT}")
        else:
            config.load_kube_config()
            print(f"[K8S-INIT] Using default context: {active_context['name']}")
    except Exception as e:
        config.load_kube_config()
        print(f"[K8S-INIT] Loaded default kubeconfig ({e})")

    k8s_core = client.CoreV1Api()
    k8s_apps = client.AppsV1Api()
    k8s_custom = client.CustomObjectsApi()

    # Ensuring the existence of a dedicated namespace
    try:
        ns_list = [ns.metadata.name for ns in k8s_core.list_namespace().items]
        if TEST_NAMESPACE not in ns_list:
            ns_body = client.V1Namespace(metadata=client.V1ObjectMeta(name=TEST_NAMESPACE))
            k8s_core.create_namespace(ns_body)
            print(f"[K8S-INIT] Created secure isolated namespace: {TEST_NAMESPACE}")
        else:
            print(f"[K8S-INIT] Detected secure namespace: {TEST_NAMESPACE}")
    except Exception as e:
        print(f"[K8S-INIT] Namespace check: {e}")

    return k8s_core, k8s_apps, k8s_custom

def build_case(pop_demands, available_cluster_indices=None):
    """Creates an optimization problem instance."""
    if available_cluster_indices is None:
        available_cluster_indices = list(range(len(PHYSICAL_CLUSTERS)))

    clusters_subset = [PHYSICAL_CLUSTERS[i] for i in available_cluster_indices]
    num_clusters = len(clusters_subset)
    num_pops = len(pop_demands)
    total_demands = sum(pop_demands)

    distances_subset = [
        [POP_DISTANCES[p][c] for c in available_cluster_indices]
        for p in range(num_pops)
    ]

    cluster_resources = [[c["vcpu"], c["ram_mb"]] for c in clusters_subset]
    cluster_costs = [c["cost"] for c in clusters_subset]

    erlang_table = generate_erlang_table(
        max_demands=max(total_demands, 10),
        traffic_per_demand=1.0,
        target_blocking=0.01,
        srs_capacity=2
    )

    case_meta = {
        "clusters": clusters_subset,
        "pops": POPS,
        "popDemands": pop_demands,
        "popLambda": [float(d) for d in pop_demands],
        "popMu": [1.0] * num_pops,
        "totalDemands": total_demands,
        "clusterResources": cluster_resources,
        "srsResources": [1, 256],
        "popDistances": distances_subset,
        "clusterDeploymentCosts": cluster_costs,
        "maxLatencyUnits": 250, # 25 ms
        "latencyPer10kmUnits": 1,
        "originDistances": [0] * num_clusters,
        "srsCapacity": 2,
        "erlang": {"trafficPerDemand": 1.0, "blockageProbability": 0.01},
        "config": {
            "transmission_cost": 1,
            "migration_cost": 50,
            "unserved_penalty": 10000,
            "alpha_cost": 0.5,
            "beta_latency": 0.5
        },
        "erlangTable": erlang_table
    }
    return case_meta

def solve_milp(case_meta):
    """Solves the problem using an exact MILP solver (HiGHS)."""
    t0 = time.perf_counter()
    model = minizinc.Model(MODEL_PATH)
    solver = minizinc.Solver.lookup(SOLVER_NAME)
    instance = minizinc.Instance(solver, model)

    num_clusters = len(case_meta["clusters"])
    num_pops = len(case_meta["pops"])
    
    instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
    instance["Clusters_"] = num_clusters
    instance["Pops_"] = num_pops
    instance["TotalDemands"] = case_meta["totalDemands"]
    instance["PopDemands"] = case_meta["popDemands"]
    instance["Resources"] = case_meta["clusterResources"]
    instance["SRSResources"] = case_meta["srsResources"]
    instance["Distances"] = case_meta["popDistances"]
    instance["InitialPlacement"] = [[0] * num_clusters for _ in range(num_pops)]
    instance["DeploymentCost"] = case_meta["clusterDeploymentCosts"]
    instance["MaxLatency"] = case_meta["maxLatencyUnits"]
    instance["LatencyPer10km"] = case_meta["latencyPer10kmUnits"]
    instance["TransmissionCost"] = case_meta["config"]["transmission_cost"]
    instance["MigrationCost"] = case_meta["config"]["migration_cost"]
    instance["UnservedPenalty"] = case_meta["config"]["unserved_penalty"]
    instance["Alpha_Cost"] = case_meta["config"]["alpha_cost"]
    instance["Beta_Latency"] = case_meta["config"]["beta_latency"]
    instance["ErlangTable"] = case_meta["erlangTable"]
    instance["BlockageProbability"] = case_meta["erlang"]["blockageProbability"]
    instance["SRSCapacity"] = case_meta["srsCapacity"]
    instance["OriginDistances"] = case_meta["originDistances"]

    result = instance.solve(timeout=datetime.timedelta(seconds=60))
    t_solve = (time.perf_counter() - t0)

    if result.solution is not None:
        instances = result["Instances"]
        assigned = result["AssignedDemands"]
        obj = result.objective
        return {
            "instances": instances,
            "assigned": assigned,
            "objective": obj,
            "status": str(result.status),
            "solve_time_sec": t_solve
        }
    else:
        # Fallback
        instances = [1] * num_clusters
        return {
            "instances": instances,
            "assigned": [],
            "objective": -1,
            "status": "FALLBACK",
            "solve_time_sec": t_solve
        }

GNN_CACHED_MODEL = None

def load_gnn_model_cached(gnn_model_path="best_baseline_gnn.pt"):
    global GNN_CACHED_MODEL
    if GNN_CACHED_MODEL is None:
        import torch
        from python.models.baseline_gnn import BaselineGNN
        device = torch.device("cpu")
        model = BaselineGNN(hidden_channels=128).to(device)
        model.load_state_dict(torch.load(gnn_model_path, map_location=device, weights_only=True))
        model.eval()
        GNN_CACHED_MODEL = (model, device)
    return GNN_CACHED_MODEL

def solve_gnn(case_meta, gnn_model_path="best_baseline_gnn.pt"):
    import torch
    import torch_geometric.transforms as T
    from graph_builder import build_hetero_graph

    model, device = load_gnn_model_cached(gnn_model_path)
    
    t0_build = time.perf_counter()
    graph = build_hetero_graph(case_meta, labels=None)
    graph = T.ToUndirected()(graph)
    graph = graph.to(device)
    t_build_graph = time.perf_counter() - t0_build

    t0_forward = time.perf_counter()
    with torch.no_grad():
        preds_tensor = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
    t_forward = time.perf_counter() - t0_forward
    
    t0_dyskret = time.perf_counter()
    # Prediction scaling
    preds = preds_tensor.squeeze(-1).cpu().numpy()
    preds = np.clip(preds, 0.0, 1e9)
    num_pops = len(case_meta["pops"])
    num_clusters = len(case_meta["clusters"])
    edge_index = graph.edge_index_dict[('pop', 'reaches', 'cluster')]
    pop_demands = case_meta["popDemands"]

    cluster_demands = [0] * num_clusters
    for p in range(num_pops):
        p_edges = [(i, int(edge_index[1, i])) for i in range(len(preds)) if int(edge_index[0, i]) == p]
        sum_p = sum(preds[i] for i, c in p_edges)
        target = pop_demands[p]
        if sum_p > 1e-6:
            for i, c in p_edges:
                cluster_demands[c] += int(round(preds[i] * (target / sum_p)))
        else:
            if p_edges:
                cluster_demands[p_edges[0][1]] += target

    erlang_tab = case_meta["erlangTable"]
    instances = [
        erlang_tab[int(d)] if int(d) < len(erlang_tab) else erlang_tab[-1]
        for d in cluster_demands
    ]
    t_dyskret = time.perf_counter() - t0_dyskret

    return {
        "instances": instances,
        "cluster_demands": cluster_demands,
        "solve_time_sec": t_build_graph + t_forward + t_dyskret,
        "t_build_graph": t_build_graph,
        "t_forward": t_forward,
        "t_dyskret": t_dyskret
    }

def apply_to_karmada(k8s_core, k8s_apps, k8s_custom, instances, clusters, deployment_name="srs-thesis"):
    """Deploys PropagationPolicy and Deployment to Karmada in an isolated namespace."""
    t_api_start = time.perf_counter()
    ns = TEST_NAMESPACE
    total_replicas = max(sum(instances), 1)

    static_weight_list = []
    available_cluster_names = []
    for idx, count in enumerate(instances):
        if count > 0 and idx < len(clusters):
            c_name = clusters[idx]["name"]
            available_cluster_names.append(c_name)
            static_weight_list.append({
                "targetCluster": {"clusterNames": [c_name]},
                "weight": count
            })

    # 1. Deployment
    deploy_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "namespace": ns,
            "labels": {"app": APP_LABEL}
        },
        "spec": {
            "replicas": total_replicas,
            "selector": {"matchLabels": {"app": APP_LABEL}},
            "template": {
                "metadata": {"labels": {"app": APP_LABEL}},
                "spec": {
                    "containers": [{
                        "name": "srs",
                        "image": "ossrs/srs:5",
                        "ports": [{"containerPort": 1935}, {"containerPort": 8080}],
                        "resources": {
                            "limits": {"cpu": "500m", "memory": "256Mi"},
                            "requests": {"cpu": "100m", "memory": "128Mi"}
                        }
                    }]
                }
            }
        }
    }

    # 2. PropagationPolicy
    policy_name = f"{deployment_name}-prop"
    policy_manifest = {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {
            "name": policy_name,
            "namespace": ns
        },
        "spec": {
            "resourceSelectors": [{
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name
            }],
            "placement": {
                "replicaScheduling": {
                    "replicaDivisionPreference": "Weighted",
                    "replicaSchedulingType": "Divided",
                    "weightPreference": {"staticWeightList": static_weight_list}
                }
            }
        }
    }

    crd_args = {"group": "policy.karmada.io", "version": "v1alpha1", "plural": "propagationpolicies"}
    
    # Policy application (safe spec patching)
    conflict_count = 0
    try:
        k8s_custom.create_namespaced_custom_object(namespace=ns, body=policy_manifest, **crd_args)
    except ApiException as e:
        if e.status == 409:
            conflict_count += 1
            k8s_custom.patch_namespaced_custom_object(
                name=policy_name,
                namespace=ns,
                body={"spec": policy_manifest["spec"]},
                **crd_args
            )
        else:
            raise e

    # Deployment application (safe spec patching)
    try:
        k8s_apps.create_namespaced_deployment(namespace=ns, body=deploy_manifest)
    except ApiException as e:
        if e.status == 409:
            conflict_count += 1
            k8s_apps.patch_namespaced_deployment(name=deployment_name, namespace=ns, body={"spec": deploy_manifest["spec"]})
        else:
            raise e

    t_api_sec = time.perf_counter() - t_api_start

    # Karmada propagation measurement (waiting for ResourceBinding and replica update)
    t_prop_start = time.perf_counter()
    binding_ok = False
    binding_args = {"group": "work.karmada.io", "version": "v1alpha1", "plural": "resourcebindings"}
    
    for _ in range(30): # up to 30 seconds
        try:
            rb_name = f"{deployment_name}-deployment"
            rb = k8s_custom.get_namespaced_custom_object(name=rb_name, namespace=ns, **binding_args)
            status = rb.get("status", {})
            conditions = status.get("conditions", [])
            is_scheduled = any(c.get("type") == "Scheduled" and c.get("status") == "True" for c in conditions)
            is_applied = any(c.get("type") == "FullyApplied" and c.get("status") == "True" for c in conditions)
            if is_scheduled or is_applied or "clusters" in rb.get("spec", {}):
                binding_ok = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    t_prop_sec = time.perf_counter() - t_prop_start

    return {
        "total_replicas": total_replicas,
        "static_weights": static_weight_list,
        "t_api_sec": t_api_sec,
        "t_prop_sec": t_prop_sec,
        "conflicts": conflict_count,
        "binding_ok": binding_ok
    }

def apply_divided_to_karmada(k8s_core, k8s_apps, k8s_custom, deployment_name="srs-divided"):
    """Simulation of the Divided heuristic using the native Karmada kube-scheduler."""
    ns = TEST_NAMESPACE
    total_replicas = 20
    
    deploy_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": deployment_name, "namespace": ns, "labels": {"app": "srs-thesis"}},
        "spec": {
            "replicas": total_replicas,
            "selector": {"matchLabels": {"app": "srs-thesis"}},
            "template": {
                "metadata": {"labels": {"app": "srs-thesis"}},
                "spec": {
                    "containers": [{"name": "srs", "image": "ossrs/srs:5"}]
                }
            }
        }
    }
    
    policy_name = f"{deployment_name}-prop"
    policy_manifest = {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {"name": policy_name, "namespace": ns},
        "spec": {
            "resourceSelectors": [{"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment_name}],
            "placement": {
                "replicaScheduling": {
                    "replicaDivisionPreference": "Weighted",
                    "replicaSchedulingType": "Divided"
                }
            }
        }
    }
    
    t_api_start = time.perf_counter()
    crd_args = {"group": "policy.karmada.io", "version": "v1alpha1", "plural": "propagationpolicies"}
    
    try:
        k8s_custom.create_namespaced_custom_object(namespace=ns, body=policy_manifest, **crd_args)
    except ApiException as e:
        if e.status == 409:
            k8s_custom.patch_namespaced_custom_object(name=policy_name, namespace=ns, body={"spec": policy_manifest["spec"]}, **crd_args)
    
    try:
        k8s_apps.create_namespaced_deployment(namespace=ns, body=deploy_manifest)
    except ApiException as e:
        if e.status == 409:
            k8s_apps.patch_namespaced_deployment(name=deployment_name, namespace=ns, body={"spec": deploy_manifest["spec"]})
            
    t_api_sec = time.perf_counter() - t_api_start
    
    t_prop_start = time.perf_counter()
    binding_args = {"group": "work.karmada.io", "version": "v1alpha1", "plural": "resourcebindings"}
    
    for _ in range(30): # up to 30 seconds
        try:
            rb_name = f"{deployment_name}-deployment"
            rb = k8s_custom.get_namespaced_custom_object(name=rb_name, namespace=ns, **binding_args)
            status = rb.get("status", {})
            conditions = status.get("conditions", [])
            is_scheduled = any(c.get("type") == "Scheduled" and c.get("status") == "True" for c in conditions)
            is_applied = any(c.get("type") == "FullyApplied" and c.get("status") == "True" for c in conditions)
            if is_scheduled or is_applied or "clusters" in rb.get("spec", {}):
                break
        except Exception:
            pass
        time.sleep(0.5)

    t_prop_sec = time.perf_counter() - t_prop_start
    
    return {"t_api_sec": t_api_sec, "t_prop_sec": t_prop_sec}

def run_all_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed for generator")
    parser.add_argument("--repeat", type=int, default=1, help="Number of experiment repetitions")
    args = parser.parse_args()

    print("=" * 70)
    print(f" STARTING AUTOMATED KARMADA EXPERIMENT SUITE (N={args.repeat})")
    print("=" * 70)
    
    k8s_core, k8s_apps, k8s_custom = init_k8s()
    results_list = []

    for idx in range(args.repeat):
        current_seed = args.seed + idx
        # Seed setting added for potential future extensions (currently scenarios are fully deterministic)
        np.random.seed(current_seed)
        random.seed(current_seed)
        print(f"\n--- Cycle {idx+1}/{args.repeat} ---")
        cycle_results = {}

        # =========================================================================
        # SCENARIO 1: Baseline latency and propagation overhead
        # =========================================================================
        print("\n>>> [SCENARIO 1] Baseline latency and propagation overhead")
        pop_demands_s1 = [10, 5, 8, 12, 6, 9] # A = 50 cohorts
        case_s1 = build_case(pop_demands_s1)

        print("  -> Solving MILP (HiGHS)...")
        res_milp_s1 = solve_milp(case_s1)
        karmada_milp_s1 = apply_to_karmada(k8s_core, k8s_apps, k8s_custom, res_milp_s1["instances"], case_s1["clusters"], deployment_name="srs-milp")
        
        print("  -> Solving GNN (BaselineGNN Inference)...")
        res_gnn_s1 = solve_gnn(case_s1)
        karmada_gnn_s1 = apply_to_karmada(k8s_core, k8s_apps, k8s_custom, res_gnn_s1["instances"], case_s1["clusters"], deployment_name="srs-gnn")
        
        print("  -> Divided heuristic (kube-scheduler)...")
        karmada_divided_s1 = apply_divided_to_karmada(k8s_core, k8s_apps, k8s_custom, deployment_name="srs-divided")

        cycle_results["scenario_1"] = {
            "milp": {**res_milp_s1, **karmada_milp_s1},
            "gnn": {**res_gnn_s1, **karmada_gnn_s1},
            "divided": karmada_divided_s1
        }

        # =========================================================================
        # SCENARIO 2: Dynamic reconfiguration in a window loop (Continuous Scaling)
        # =========================================================================
        print("\n>>> [SCENARIO 2] Dynamic reconfiguration in a window loop (5 windows T_1..T_5)")
        windows_demands = [
            [4, 2, 3, 5, 2, 4],
            [10, 6, 8, 14, 8, 14],
            [22, 18, 20, 26, 16, 18],
            [12, 10, 14, 16, 8, 10],
            [5, 3, 4, 8, 4, 6]
        ]
        s2_steps = []
        
        for step_idx, dem in enumerate(windows_demands):
            case_step = build_case(dem)
            t_start_loop = time.perf_counter()
            gnn_out = solve_gnn(case_step)
            k_out = apply_to_karmada(k8s_core, k8s_apps, k8s_custom, gnn_out["instances"], case_step["clusters"], deployment_name="srs-dynamic")
            
            s2_steps.append({
                "t_loop_total_sec": time.perf_counter() - t_start_loop,
                "t_api_sec": k_out["t_api_sec"],
                "t_prop_sec": k_out["t_prop_sec"]
            })
            time.sleep(1)

        cycle_results["scenario_2"] = s2_steps
        
        # =========================================================================
        # SCENARIO 3: Real-time comparison (GNN vs MILP Benchmark)
        # =========================================================================
        print("\n>>> [SCENARIO 3] Real-time comparison (GNN vs MILP)")
        pop_demands_s3 = [15, 12, 18, 25, 10, 20] # A = 100 cohorts
        case_s3 = build_case(pop_demands_s3)
        
        milp_s3 = solve_milp(case_s3)
        gnn_s3 = solve_gnn(case_s3)
        
        diff_instances = [abs(m - g) for m, g in zip(milp_s3["instances"], gnn_s3["instances"])]
        speedup = milp_s3["solve_time_sec"] / max(gnn_s3["solve_time_sec"], 1e-6)

        print(f"  -> Replica difference (|MILP - GNN|): {diff_instances} | GNN Speedup: {speedup:.1f}x")

        cycle_results["scenario_3"] = {
            "demands": pop_demands_s3,
            "milp_time_sec": milp_s3["solve_time_sec"],
            "gnn_time_sec": gnn_s3["solve_time_sec"],
            "speedup": speedup,
            "milp_instances": milp_s3["instances"],
            "gnn_instances": gnn_s3["instances"],
            "diff_instances": diff_instances
        }

        # =========================================================================
        # SCENARIO 4: Cluster failure resilience (Failover & Dynamic Rescheduling)
        # =========================================================================
        print("\n>>> [SCENARIO 4] Edge cluster failure resilience (Failover)")
        demands_s4 = [12, 10, 14, 18, 10, 16]
        case_s4_normal = build_case(demands_s4, available_cluster_indices=[0, 1, 2, 3])
        gnn_normal = solve_gnn(case_s4_normal)
        k_normal = apply_to_karmada(k8s_core, k8s_apps, k8s_custom, gnn_normal["instances"], case_s4_normal["clusters"], deployment_name="srs-failover")

        time.sleep(1)
        t_fail_start = time.perf_counter()
        case_s4_failover = build_case(demands_s4, available_cluster_indices=[0, 1, 3])
        gnn_failover = solve_gnn(case_s4_failover)
        k_failover = apply_to_karmada(k8s_core, k8s_apps, k8s_custom, gnn_failover["instances"], case_s4_failover["clusters"], deployment_name="srs-failover")
        t_failover_sec = time.perf_counter() - t_fail_start

        cycle_results["scenario_4"] = {
            "t_failover_sec": t_failover_sec
        }

        results_list.append(cycle_results)
        
        # Cleanup
        for dep_name in ["srs-milp", "srs-gnn", "srs-dynamic", "srs-failover", "srs-divided"]:
            try:
                k8s_apps.patch_namespaced_deployment(name=dep_name, namespace=TEST_NAMESPACE, body={"spec": {"replicas": 0}})
            except Exception:
                pass

    # Full aggregation with STD
    n = len(results_list)
    
    def get_stats(data_list):
        return {"mean": float(np.mean(data_list)), "std": float(np.std(data_list, ddof=1) if len(data_list) > 1 else 0.0)}
        
    milp_dec = [r["scenario_1"]["milp"]["solve_time_sec"] for r in results_list]
    milp_api = [r["scenario_1"]["milp"]["t_api_sec"] for r in results_list]
    milp_prop = [r["scenario_1"]["milp"]["t_prop_sec"] for r in results_list]
    milp_total = [d + a + p for d, a, p in zip(milp_dec, milp_api, milp_prop)]
    
    gnn_dec = [r["scenario_1"]["gnn"]["solve_time_sec"] for r in results_list]
    gnn_api = [r["scenario_1"]["gnn"]["t_api_sec"] for r in results_list]
    gnn_prop = [r["scenario_1"]["gnn"]["t_prop_sec"] for r in results_list]
    gnn_total = [d + a + p for d, a, p in zip(gnn_dec, gnn_api, gnn_prop)]
    
    div_api = [r["scenario_1"]["divided"]["t_api_sec"] for r in results_list]
    div_prop = [r["scenario_1"]["divided"]["t_prop_sec"] for r in results_list]
    div_total = [a + p for a, p in zip(div_api, div_prop)]
    
    aggregated = {
        "milp": {
            "t_decision": get_stats(milp_dec),
            "t_api": get_stats(milp_api),
            "t_prop": get_stats(milp_prop),
            "t_total": get_stats(milp_total),
        },
        "gnn": {
            "t_decision": get_stats(gnn_dec),
            "t_api": get_stats(gnn_api),
            "t_prop": get_stats(gnn_prop),
            "t_total": get_stats(gnn_total),
        },
        "divided": {
            "t_decision": {"mean": 0.0, "std": 0.0},  # Negligible time, no calling optimization model
            "t_api": get_stats(div_api),
            "t_prop": get_stats(div_prop),
            "t_total": get_stats(div_total),
        },
        "s3_speedup": get_stats([r["scenario_3"]["speedup"] for r in results_list]),
        "s4_failover": get_stats([r["scenario_4"]["t_failover_sec"] for r in results_list])
    }

    out_json = RESULTS_DIR / "karmada_experiments_results.json"
    with open(out_json, "w") as f:

        json.dump({"repeats": args.repeat, "aggregated": aggregated, "results": results_list}, f, indent=2)
    print(f"\n[SAVE] Results saved successfully to file: {out_json}")

if __name__ == "__main__":
    run_all_experiments()
