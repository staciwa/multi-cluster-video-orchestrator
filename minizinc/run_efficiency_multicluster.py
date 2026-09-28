
import sys
import os
import json
from datetime import timedelta
from enum import Enum

# Add current directory to path
sys.path.append(os.getcwd())

from minizinc import Instance, Model, Solver
from erlang_utils import generate_erlang_table, erlang_b

# ============================================================================ 
# KONFIGURACJA TESTU WIELOKLASTROWEGO
# ============================================================================ 
# 3 Clusters, 60 Demands (~20 per cluster)
# Goal: Show that savings scale independently in each cluster

SCENARIO_MULTICLUSTER = {
    "name": "Erlang Efficiency Multi-Cluster (60 demands, 3 clusters)",
    "data": {
        "SRSCapacity": 1,        # Generic: 1 demand = 1 instance
        "SRSResources": [1, 1],
        "MaxLatency": 50,        # Forces locality
        "TransmissionCost": 1,
        "LatencyPer10km": 1,
        "MigrationCost": 0,
        "UnservedPenalty": 10000,
        "Clusters_": 3,
        "Demands_": 60,
        "Resources": [
            [100, 100], [100, 100], [100, 100] # Large clusters
        ],
        "DeploymentCost": [10, 10, 10],
        # Distances crafted to divide demands into 3 groups
        # 0-19 -> Klaster 0
        # 20-39 -> Klaster 1
        # 40-59 -> Klaster 2
        "Distances": [], 
        "InitialPlacement": [[0]*3 for _ in range(60)],
        "Alpha_Cost": 1.0,
        "Beta_Latency": 0.0
    }
}

# Distance generation
dists = []
for i in range(60):
    row = []
    # Assignment to group
    target_cluster = 0
    if i >= 20: target_cluster = 1
    if i >= 40: target_cluster = 2
    
    for c in range(3):
        if c == target_cluster:
            row.append(5)   # Bardzo blisko (5km)
        else:
            row.append(100) # Far (100km) > MaxLatency(50ms/10km=500km? No, LatencyPer10km=1ms -> 50km max)
            # MaxLatency=50, LatencyPer10km=1.
            # Distance 100 -> latency 100*1 = 100ms > 50ms.
            # Therefore it MUST select the closest cluster.
    dists.append(row)
SCENARIO_MULTICLUSTER["data"]["Distances"] = dists


# Parametry Erlanga
BLOCKAGE_PROB = 0.01      # 1%
TRAFFIC_PER_DEMAND = 0.2  # 20% activity

def run_model(model_file, data, is_erlang=False):
    solver = Solver.lookup("highs")
    model = Model(model_file)
    instance = Instance(solver, model)
    
    instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
    instance["SRSResources"] = data["SRSResources"]
    instance["MaxLatency"] = data["MaxLatency"]
    instance["TransmissionCost"] = data["TransmissionCost"]
    instance["LatencyPer10km"] = data["LatencyPer10km"]
    instance["MigrationCost"] = data["MigrationCost"]
    instance["UnservedPenalty"] = data["UnservedPenalty"]
    instance["Clusters_"] = data["Clusters_"]
    instance["Demands_"] = data["Demands_"]
    instance["Resources"] = data["Resources"]
    instance["Distances"] = data["Distances"]
    instance["InitialPlacement"] = data["InitialPlacement"]
    instance["DeploymentCost"] = data["DeploymentCost"]
    instance["Alpha_Cost"] = data["Alpha_Cost"]
    instance["Beta_Latency"] = data["Beta_Latency"]

    if is_erlang:
        erlang_table = generate_erlang_table(
            data["Demands_"],
            TRAFFIC_PER_DEMAND,
            BLOCKAGE_PROB
        )
        instance["ErlangTable"] = erlang_table
        instance["BlockageProbability"] = BLOCKAGE_PROB
    else:
        instance["SRSCapacity"] = data["SRSCapacity"]

    result = instance.solve(timeout=timedelta(seconds=60))
    
    if result.solution is None:
        return None
        
    output_str = str(result.solution)
    start = output_str.find('{')
    end = output_str.rfind('}') + 1
    if start != -1:
        return json.loads(output_str[start:end])
    return None

def main():
    print(f"\n{'='*60}")
    print(f" TEST WIELOKLASTROWY: Generic vs Erlang")
    print(f"{ '='*60}")
    print(f"Number of demands: 60 (distributed evenly over 3 clusters)")
    print(f"Generic: 1 instance = 1 demand (no overbooking)")
    print(f"Erlang:  Traffic 0.2 Erl/demand, Max P_B = 1%")
    print(f"{'-'*60}")

    # 1. Generic
    print("1. Uruchamianie modelu GENERYCZNEGO...", end=" ")
    res_gen = run_model("model_generic.mzn", SCENARIO_MULTICLUSTER["data"], is_erlang=False)
    if res_gen:
        print("Gotowe.")
    else:
        print("Error.")
        return

    # 2. Erlang
    print("2. Uruchamianie modelu ERLANGA......", end=" ")
    res_erl = run_model("model_erlang.mzn", SCENARIO_MULTICLUSTER["data"], is_erlang=True)
    if res_erl:
        print("Gotowe.")
    else:
        print("Error.")
        return

    # Presentation of results
    print(f"\n{'='*60}")
    print(f" {'CLUSTER':<10} | {'GENERIC (Inst)':<15} | {'ERLANG (Inst)':<15} | {'SAVINGS':<10}")
    print(f"{'-'*60}")
    
    total_gen = 0
    total_erl = 0
    
    inst_gen_list = res_gen["instances"]
    inst_erl_list = res_erl["instances"]
    
    # Check assignment (demands per cluster)
    demands_gen = res_gen.get("demandsPerCluster", []) # Generic might not output this field directly usually, but logic holds
    # Actually generic output json usually has instances. We can assume assignment follows distance logic (20 per cluster).
    
    for c in range(3):
        g = inst_gen_list[c]
        e = inst_erl_list[c]
        total_gen += g
        total_erl += e
        
        save_pct = 0
        if g > 0:
            save_pct = (g - e) / g * 100
            
        print(f" Cluster {c+1:<2} | {g:<15} | {e:<15} | {save_pct:>8.1f}%")
        
    print(f"{'-'*60}")
    print(f" {'SUMA':<10} | {total_gen:<15} | {total_erl:<15} | {((total_gen-total_erl)/total_gen*100):>8.1f}%")
    print(f"{ '='*60}")

    # Theoretical verification for one cluster
    # We expect 20 demands -> 4 Erlangs
    # Let's check how many instances are needed for 4 Erl and P_B 1%
    expected_traffic = 20 * TRAFFIC_PER_DEMAND
    print(f"\nTheoretical analysis (per cluster):")
    print(f"Input traffic: 20 demands * 0.2 Erl = {expected_traffic:.1f} Erl")
    
    # Let's find min k
    for k in range(1, 25):
        pb = erlang_b(expected_traffic, k)
        if pb <= BLOCKAGE_PROB:
            print(f"Wymagane instancje (teoretycznie): {k} (P_B={pb*100:.3f}%)")
            break
            
if __name__ == "__main__":
    main()
