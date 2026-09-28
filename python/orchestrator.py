import minizinc
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import os
import sys
import json
import time
import datetime
from enum import Enum
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(SCRIPT_DIR).parent

kubeconfig_path = os.path.join(SCRIPT_DIR, "karmada_config")
if os.path.exists(kubeconfig_path):
    os.environ['KUBECONFIG'] = kubeconfig_path
    print(f"[INFO] Using kubeconfig: {kubeconfig_path}")

MODEL_PATH = str(PROJECT_ROOT / "minizinc" / "model_erlang.mzn")
SOLVER_NAME = "highs"

SOLVER_TIMEOUT = 60

SRS_VCPU = 1
SRS_RAM_MB = 256
SRS_RAM_K8S = "256Mi"
SRS_RAM_LIMIT_K8S = "512Mi"

CLUSTER_NAME_PREFIX = "cluster"

sys.path.append(str(PROJECT_ROOT / "minizinc"))
from erlang_utils import generate_erlang_table

def try_load_gnn_model(gnn_model_path: str | None):
    if gnn_model_path is None:
        return None
    try:
        import sys
        import os
        import torch
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))
        from python.models.baseline_gnn import BaselineGNN
        minizinc_dir = str(PROJECT_ROOT / "minizinc")
        if minizinc_dir not in sys.path:
            sys.path.append(minizinc_dir)
        from graph_builder import build_hetero_graph

        device = torch.device('cpu')
        model = BaselineGNN(hidden_channels=128).to(device)
        model.load_state_dict(torch.load(gnn_model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"[INFO] Loaded GNN model from {gnn_model_path}")
        return model, device, build_hetero_graph
    except Exception as e:
        print(f"[WARN] Could not load GNN model: {e}")
        return None

def predict_gnn_placement(gnn_bundle, meta):
    import torch

    import math
    model, device, build_graph = gnn_bundle
    graph = build_graph(meta, labels=None)
    graph = graph.to(device)

    with torch.no_grad():
        assigned_demands = model(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
        
    # assigned_demands is [num_edges, 1] containing continuous demand predictions
    import numpy as np
    preds = assigned_demands.squeeze(-1).cpu().numpy()
    # Masking negative logic to 0
    preds = np.clip(preds, 0.0, 1e9)
    
    # Reversing cohort scaling (x100) forced in graph_builder.py
    preds = preds * 100.0
    
    num_pops = meta.get("pops", [])
    num_clusters = meta.get("clusters", [])
    
    placement_pred = [[0] * len(num_clusters) for _ in range(len(num_pops))]
    
    edge_index = graph.edge_index_dict[('pop', 'reaches', 'cluster')].cpu().numpy()
    pop_demands = meta.get("popDemands", [meta.get("demandToPop", []).count(i) for i in range(len(num_pops))])
    
    for p in range(len(num_pops)):
        p_edges = []
        for i in range(len(preds)):
            if int(edge_index[0, i]) == p:
                p_edges.append((i, int(edge_index[1, i])))
        
        target_sum = pop_demands[p]
        sum_preds = sum(preds[i] for i, c in p_edges)
        
        if math.isnan(sum_preds) or not math.isfinite(sum_preds):
            sum_preds = 0.0
        
        # Keep unserved - scale only to edge sum
        assigned_total = int(round(sum_preds))
        assigned_total = min(assigned_total, int(target_sum))
        
        if sum_preds > 1e-6:
            scaled = [preds[i] * (assigned_total / sum_preds) for i, c in p_edges]
        else:
            scaled = [0.0] * len(p_edges)
            
        floored = [math.floor(x) for x in scaled]
        remainders = [(scaled[j] - floored[j], j) for j in range(len(scaled))]
        remainders.sort(key=lambda x: x[0], reverse=True)
        
        diff = int(assigned_total - sum(floored))
        for k in range(min(diff, len(remainders))):
            idx = remainders[k][1]
            floored[idx] += 1
            
        for j, (i, c) in enumerate(p_edges):
            placement_pred[p][c] = int(floored[j])
            
    return placement_pred

def build_cluster_map(cluster_names: list[str]) -> dict[int, str]:
    return {i + 1: name for i, name in enumerate(cluster_names)}

def load_case(case_path: Path) -> dict:
    with open(case_path, "r") as f:
        return json.load(f)

def solve_allocation(case_path: Path | None = None, gnn_bundle = None):
    print(f"--- 1. Solving model ({MODEL_PATH}) ---")

    model = minizinc.Model(MODEL_PATH)
    solver = minizinc.Solver.lookup(SOLVER_NAME)
    instance = minizinc.Instance(solver, model)

    previous_state_file = Path("/tmp/previous_placement.json")
    if case_path is not None:
        case_data = load_case(case_path)
        input_data = case_data.get("input", {})
        meta = input_data.get("meta", {})
        config_dict = input_data.get("config", {})
        
        clusters = meta.get("clusters", [])
        num_clusters = len(clusters)
        num_pops = len(meta.get("pops", []))
        pop_demands = meta.get("popDemands", [])
        total_demands = sum(pop_demands)
        pop_distances = meta.get("popDistances", [])
        erlang = meta.get("erlang", {})
        
        erlang_table = generate_erlang_table(
            max_demands=total_demands,
            traffic_per_demand=erlang["trafficPerDemand"],
            target_blocking=erlang["blockageProbability"],
            srs_capacity=meta["srsCapacity"]
        )

        # NOTE: In a production environment, InitialPlacement should be read 
        # directly from the Karmada/Kubernetes API, not from a local file, 
        # to avoid discrepancies in case of manual interventions.
        initial_placement = [[0] * num_clusters for _ in range(num_pops)]
        is_valid_saved_state = False
        
        if previous_state_file.exists():
            try:
                with open(previous_state_file, "r") as f:
                    saved_placement = json.load(f)
                
                if len(saved_placement) == num_pops and len(saved_placement[0]) == num_clusters:
                    sum_saved = sum(sum(row) for row in saved_placement)
                    sum_meta = sum(meta["popDemands"])
                    if sum_saved == sum_meta:
                        initial_placement = saved_placement
                        is_valid_saved_state = True
                        print("[INFO] Loaded consistent state from previous_placement.json")
                    else:
                        print(f"[WARN] Inconsistent sum of demands. Using GNN/zeros.")
                else:
                    print("[WARN] Inconsistent dimensions of previous_placement.json. Using GNN/zeros.")
            except Exception as e:
                print(f"[WARN] Error reading previous_placement.json: {e}.")

        if gnn_bundle is not None:
            t0 = time.time()
            try:
                gnn_placement = predict_gnn_placement(gnn_bundle, meta)
                print(f"[GNN] Prediction generated in {time.time()-t0:.2f}s")
                if not is_valid_saved_state:
                    initial_placement = gnn_placement
                    print("[INFO] GNN prediction used as initial state (warm-start).")
            except Exception as e:
                print(f"[WARN] Error during GNN prediction: {e}. Skipping surrogate.")

        instance["ResourceTypes"] = ["vCPU", "RAM"]
        instance["Clusters_"] = num_clusters
        instance["Pops_"] = num_pops
        instance["TotalDemands"] = total_demands
        instance["PopDemands"] = pop_demands
        instance["Resources"] = meta["clusterResources"]
        instance["SRSResources"] = meta["srsResources"]
        instance["Distances"] = pop_distances
        instance["InitialPlacement"] = initial_placement
        instance["DeploymentCost"] = meta["clusterDeploymentCosts"]
        instance["MaxLatency"] = meta["maxLatencyUnits"]
        instance["LatencyPer10km"] = meta["latencyPer10kmUnits"]
        instance["TransmissionCost"] = config_dict["transmission_cost"]
        instance["MigrationCost"] = config_dict["migration_cost"]
        instance["UnservedPenalty"] = config_dict["unserved_penalty"]
        instance["Alpha_Cost"] = config_dict["alpha_cost"]
        instance["Beta_Latency"] = config_dict["beta_latency"]
        instance["ErlangTable"] = erlang_table
        instance["BlockageProbability"] = erlang["blockageProbability"]
        instance["SRSCapacity"] = meta["srsCapacity"]
        instance["OriginDistances"] = meta["originDistances"]
        cluster_names = [c["name"] for c in clusters]
    else:
        instance["ResourceTypes"] = ["vCPU", "RAM"]
        instance["Clusters_"] = 2
        instance["Pops_"] = 2
        instance["PopDemands"] = [2, 2]
        instance["TotalDemands"] = 4
        instance["Resources"] = [[8, 4096], [16, 8192]]
        instance["SRSResources"] = [SRS_VCPU, SRS_RAM_MB]
        instance["Distances"] = [[10, 200], [600, 20]]
        initial_placement = [[2, 0], [0, 2]]
        instance["InitialPlacement"] = initial_placement
        num_clusters = 2
        instance["DeploymentCost"] = [100, 500]
        instance["MaxLatency"] = 250
        instance["LatencyPer10km"] = 1
        instance["TransmissionCost"] = 1
        instance["MigrationCost"] = 50
        instance["UnservedPenalty"] = 10000
        instance["Alpha_Cost"] = 0.5
        instance["Beta_Latency"] = 0.5
        instance["BlockageProbability"] = 0.01
        instance["SRSCapacity"] = 2
        erlang_table = generate_erlang_table(
            max_demands=4,
            traffic_per_demand=1,
            target_blocking=0.01,
            srs_capacity=2
        )
        instance["ErlangTable"] = erlang_table
        instance["OriginDistances"] = [0, 0]
        cluster_names = [f"{CLUSTER_NAME_PREFIX}-{i+1}" for i in range(2)]

    result = instance.solve(timeout=datetime.timedelta(seconds=SOLVER_TIMEOUT))

    if result.solution is None:
        print("[WARN] Model did not find a solution within the timeout! Using fallback (previous state).")
        per_cluster_demands = [sum(row[c] for row in initial_placement) for c in range(num_clusters)]
        instances = [erlang_table[int(d)] if int(d) < len(erlang_table) else erlang_table[-1] for d in per_cluster_demands]
        cluster_map = build_cluster_map(cluster_names)
        return instances, cluster_map

    print(f"Status: {result.status}")
    if result.status != minizinc.result.Status.OPTIMAL_SOLUTION:
        print("[WARN] The returned solver solution may be suboptimal (time limit).")
    print(f"Objective function: {result.objective}")
    print(f"Placement (demands -> clusters): {result['AssignedDemands']}")
    print(f"Instances (instances per cluster): {result['Instances']}")

    try:
        temp_file = Path("/tmp/previous_placement.json.tmp")
        with open(temp_file, "w") as f:
            json.dump(result["AssignedDemands"], f)
        os.replace(temp_file, previous_state_file)
    except Exception as e:
        print(f"[WARN] Failed to atomically save previous_placement.json: {e}")

    cluster_map = build_cluster_map(cluster_names)
    return result["Instances"], cluster_map

def generate_manifests_dict(instances_per_cluster, cluster_map, srs_resources=(1, 256)):
    print("--- 2. Generating YAML manifests ---")
    print(f"    Instances per cluster: {instances_per_cluster}")

    total_replicas = sum(instances_per_cluster)
    print(f"    Total number of replicas: {total_replicas}")

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "srs", "namespace": "default", "labels": {"app": "srs"}},
        "spec": {
            "replicas": total_replicas,
            "selector": {"matchLabels": {"app": "srs"}},
            "template": {
                "metadata": {"labels": {"app": "srs"}},
                "spec": {
                    "containers": [{
                        "name": "srs",
                        "image": "ossrs/srs:5",
                        "ports": [{"containerPort": 1935}, {"containerPort": 8080}, {"containerPort": 1985}],
                        "resources": {
                            "limits": {
                                "cpu": str(srs_resources[0]),
                                "memory": f"{srs_resources[1] * 2}Mi"
                            },
                            "requests": {
                                "cpu": str(srs_resources[0]),
                                "memory": f"{srs_resources[1]}Mi"
                            }
                        }
                    }]
                }
            }
        }
    }

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "srs-service", "namespace": "default"},
        "spec": {
            "type": "NodePort",
            "selector": {"app": "srs"},
            "ports": [
                {"name": "rtmp", "port": 1935, "targetPort": 1935},
                {"name": "http", "port": 8080, "targetPort": 8080},
                {"name": "webrtc", "port": 1985, "targetPort": 1985}
            ]
        }
    }

    static_weight_list = []
    available_clusters = []

    for idx, count in enumerate(instances_per_cluster):
        mzn_id = idx + 1
        if count > 0:
            c_name = cluster_map.get(mzn_id)
            if c_name:
                available_clusters.append(c_name)
                static_weight_list.append({
                    "targetCluster": {"clusterNames": [c_name]},
                    "weight": count
                })
                print(f"    Cluster {c_name}: {count} instances")

    policy_deploy = {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {"name": "srs-propagation", "namespace": "default"},
        "spec": {
            "resourceSelectors": [{"apiVersion": "apps/v1", "kind": "Deployment", "name": "srs"}],
            "placement": {
                "replicaScheduling": {
                    "replicaDivisionPreference": "Weighted",
                    "replicaSchedulingType": "Divided",
                    "weightPreference": {"staticWeightList": static_weight_list}
                }
            }
        }
    }

    policy_service = {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {"name": "srs-service-propagation", "namespace": "default"},
        "spec": {
            "resourceSelectors": [{"apiVersion": "v1", "kind": "Service", "name": "srs-service"}],
            "placement": {
                "clusterAffinity": {"clusterNames": available_clusters}
            }
        }
    }

    return deployment, service, policy_deploy, policy_service

def apply_to_karmada(deploy, svc, p_deploy, p_svc, dry_run=False):
    print("\n--- 3. Deploying to Karmada (API) ---")

    if dry_run:
        print("[DRY-RUN] Test mode (dry-run). Skipping physical Karmada API calls.")
        print(f"[DRY-RUN] Deployment replicas: {deploy['spec']['replicas']}")
        print(f"[DRY-RUN] PropagationPolicy targets: {[w['targetCluster']['clusterNames'][0] + ': ' + str(w['weight']) for w in p_deploy['spec']['placement']['replicaScheduling']['weightPreference']['staticWeightList']]}")
        return

    config.load_kube_config()

    k8s_apps = client.AppsV1Api()
    k8s_core = client.CoreV1Api()
    k8s_custom = client.CustomObjectsApi()
    ns = "default"
    
    # Cluster validation
    try:
        cluster_crd_args = {"group": "cluster.karmada.io", "version": "v1alpha1", "plural": "clusters"}
        existing_clusters = k8s_custom.list_cluster_custom_object(**cluster_crd_args)
        known_cluster_names = [item["metadata"]["name"] for item in existing_clusters.get("items", [])]
        
        valid_clusters = []
        for c in p_svc["spec"]["placement"]["clusterAffinity"]["clusterNames"]:
            if c in known_cluster_names:
                valid_clusters.append(c)
            else:
                print(f"[CRITICAL WARNING] Cluster '{c}' does not exist in the Karmada registry! It will be skipped.")
        
        if not valid_clusters:
            print("[ERROR] None of the target clusters exist. Aborting deployment.")
            return

        p_svc["spec"]["placement"]["clusterAffinity"]["clusterNames"] = valid_clusters
        
        valid_weights = []
        for w in p_deploy["spec"]["placement"]["replicaScheduling"]["weightPreference"]["staticWeightList"]:
            c = w["targetCluster"]["clusterNames"][0]
            if c in known_cluster_names:
                valid_weights.append(w)
        p_deploy["spec"]["placement"]["replicaScheduling"]["weightPreference"]["staticWeightList"] = valid_weights

        new_total = sum(w["weight"] for w in valid_weights)
        if new_total != deploy["spec"]["replicas"]:
            print(f"[WARN] Reducing replicas from {deploy['spec']['replicas']} to {new_total} (clusters excluded).")
            deploy["spec"]["replicas"] = new_total
    except ApiException as e:
        print(f"[WARN] Failed to verify Karmada clusters: {e}")

    # Handling scaling to zero (total_replicas == 0)
    if deploy["spec"]["replicas"] == 0:
        print("[INFO] No replicas to deploy. Removing (or scaling to 0) from Karmada.")
        try:
            k8s_apps.patch_namespaced_deployment(name="srs", namespace=ns, body={"spec": {"replicas": 0}})
            print("[OK] Scaled srs Deployment to 0.")
        except ApiException as e:
            print(f"[WARN] Failed to scale Deployment: {e}")
        return

    def apply_obj(api_func_create, api_func_patch, obj, name, kind):
        try:
            api_func_create(namespace=ns, body=obj)
            print(f"[OK] Created {kind}: {name}")
        except ApiException as e:
            if e.status == 409:
                try:
                    if kind == "Service":
                        api_func_patch(name=name, namespace=ns, body={"spec": obj["spec"]})
                    else:
                        api_func_patch(name=name, namespace=ns, body={"spec": obj["spec"]})
                    print(f"[OK] Updated {kind}: {name}")
                except ApiException as ex:
                    print(f"[ERROR] Error patching {kind} {name}: {ex}")
            else:
                print(f"[ERROR] Error {kind} {name}: {e}")

    crd_args = {"group": "policy.karmada.io", "version": "v1alpha1", "plural": "propagationpolicies"}

    # FIRST we create PropagationPolicy
    policies_ok = True
    for pol in [p_deploy, p_svc]:
        name = pol['metadata']['name']
        try:
            k8s_custom.create_namespaced_custom_object(namespace=ns, body=pol, **crd_args)
            print(f"[OK] Created Policy: {name}")
        except ApiException as e:
            if e.status == 409:
                try:
                    existing = k8s_custom.get_namespaced_custom_object(name=name, namespace=ns, **crd_args)
                    pol['metadata']['resourceVersion'] = existing['metadata']['resourceVersion']
                    k8s_custom.replace_namespaced_custom_object(name=name, namespace=ns, body=pol, **crd_args)
                    print(f"[OK] Updated (replaced) Policy: {name}")
                except Exception as ex:
                    print(f"[ERROR] Error during replace Policy {name}: {ex}")
                    policies_ok = False
            else:
                print(f"[ERROR] Error Policy {name}: {e}")
                policies_ok = False

    if not policies_ok:
        print("[ERROR] Error creating propagation policies. Halting deployment of target resources.")
        return

    # ONLY THEN we send manifests, so Karmada controller immediately assigns them to policies!
    apply_obj(k8s_apps.create_namespaced_deployment, k8s_apps.patch_namespaced_deployment, deploy, "srs", "Deployment")
    apply_obj(k8s_core.create_namespaced_service, k8s_core.patch_namespaced_service, svc, "srs-service", "Service")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Karmada Orchestrator for SRS")
    parser.add_argument("case", nargs="?", help="Path to reference_case_XXX.json")
    parser.add_argument("--gnn-model", help="Path to trained GNN model (best_model.pt) for warm start")
    parser.add_argument("--dry-run", action="store_true", help="Simulation mode without physical Kubernetes/Karmada API calls")
    parser.add_argument("--once", action="store_true", help="Single iteration instead of continuous loop")
    args = parser.parse_args()

    try:
        case_path = Path(args.case) if args.case else None
        
        print("--- Loading GNN model (Warm-Up) ---")
        gnn_bundle = try_load_gnn_model(args.gnn_model)
        
        # Instead of an offline script, the orchestrator functions as a continuous loop (Continuous Control Loop)
        print("--- Starting continuous orchestration loop (Reconciliation Loop) ---")
        iteration = 1
        
        # Reading SRS resources from file (if provided)
        srs_res = [SRS_VCPU, SRS_RAM_MB]
        if case_path is not None:
            with open(case_path, "r") as f:
                temp_meta = json.load(f)
                srs_res = temp_meta.get("srsResources", srs_res)
        
        while True:
            print(f"\n[Iteration {iteration}] Querying status and recalculating allocations...")
            try:
                instances, cluster_map = solve_allocation(case_path, gnn_bundle=gnn_bundle)
                deploy, svc, p_deploy, p_svc = generate_manifests_dict(instances, cluster_map, srs_res)
                apply_to_karmada(deploy, svc, p_deploy, p_svc, dry_run=args.dry_run)
            except Exception as e:
                print(f"[ERROR] Error in decision loop: {e}")
            
            if args.once:
                print("[INFO] Completed single iteration (--once).")
                break

            # TODO (Future Work): Here, instead of sleep, there will eventually be querying 
            # of the Prometheus or Kubernetes API for actual load/replicas in the cloud (e.g., workloads API).
            # In the simulated POC environment, we simply sleep the process.
            time.sleep(300) # Every 5 minutes
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nStopped orchestration loop.")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
