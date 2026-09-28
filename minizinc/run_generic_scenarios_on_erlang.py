
import sys
import os
import json
from datetime import timedelta
from enum import Enum

# Add current directory to path
sys.path.append(os.getcwd())

from minizinc import Instance, Model, Solver
from erlang_utils import generate_erlang_table
from test_scenarios import (
    SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5,
    SCENARIO_6A, SCENARIO_6B, SCENARIO_6C, SCENARIO_6D,
    SCENARIO_7, SCENARIO_8
)

MODEL_FILE = "model_erlang.mzn"
SOLVER_NAME = "highs"

# Adaptive parameters for Erlang model
BLOCKAGE_PROB = 0.01  # 1%
TRAFFIC_PER_DEMAND = 0.5 # 0.5 Erlang per demand

def run_scenario_on_erlang(scenario_num, scenario_obj):
    print(f"\n{'='*80}")
    print(f" SCENARIO {scenario_num} (on Erlang model): {scenario_obj['name']}")
    print(f" Parametry dodatkowe: P_B={BLOCKAGE_PROB}, Traffic={TRAFFIC_PER_DEMAND} Erl")
    print(f"{'='*80}")
    
    data = scenario_obj["data"].copy()
    
    # Generowanie tablicy Erlanga
    erlang_table = generate_erlang_table(
        data["Demands_"],
        TRAFFIC_PER_DEMAND,
        BLOCKAGE_PROB
    )
    
    try:
        solver = Solver.lookup(SOLVER_NAME)
        model = Model([MODEL_FILE])
        instance = Instance(solver, model)
        
        # Injecting Erlang parameters
        instance["ErlangTable"] = erlang_table
        instance["BlockageProbability"] = BLOCKAGE_PROB
        
        # Standardowe parametry ze scenariusza
        instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
        # instance["SRSCapacity"] - IGNORED in Erlang model (replaced by ErlangTable)
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
        
        result = instance.solve(timeout=timedelta(seconds=60))
        
        if result.solution is None:
            print("Result: UNSATISFIABLE (No solution)")
            return
            
        # Parsing results
        output_str = str(result.solution)
        start = output_str.find('{')
        end = output_str.rfind('}') + 1
        
        if start != -1 and end > 0:
            res_json = json.loads(output_str[start:end])
            
            print(f"\n📋 WYNIKI:")
            print(f"   Instances:      {res_json.get('instances', [])}")
            print(f"   Unserved:       {res_json.get('unserved', 0)}")
            print(f"   Migrations:     {res_json.get('migrations', 0)}")
            print(f"   Total Cost:     {res_json.get('totalCost', 0)}")
            print(f"   Total Latency:  {res_json.get('totalLatency', 0)}")
            
            # Verification if Erlang table worked
            demands_per_cluster = res_json.get('demandsPerCluster', [])
            instances = res_json.get('instances', [])
            print(f"   Demands/Cluster:{demands_per_cluster}")
            
            check_ok = True
            for i, (dem, inst) in enumerate(zip(demands_per_cluster, instances)):
                req_inst = erlang_table[dem]
                if inst < req_inst:
                    print(f"   ❌ ERROR in Cluster {i+1}: has {inst} instances, requires {req_inst} for {dem} demands!")
                    check_ok = False
            
            if check_ok:
                print("   ✅ Weryfikacja Erlanga: OK")

        else:
            print("Failed to parse JSON output.")
            print(output_str)

    except Exception as e:
        print(f"Error during execution: {e}")

def run_all():
    run_scenario_on_erlang(1, SCENARIO_1)
    run_scenario_on_erlang(2, SCENARIO_2)
    run_scenario_on_erlang(3, SCENARIO_3)
    run_scenario_on_erlang(4, SCENARIO_4)
    run_scenario_on_erlang(5, SCENARIO_5)
    
    # Scenariusz 6 ma warianty
    print("\n--- Scenariusz 6 (Warianty) ---")
    run_scenario_on_erlang("6A", SCENARIO_6A)
    run_scenario_on_erlang("6B", SCENARIO_6B)
    run_scenario_on_erlang("6C", SCENARIO_6C)
    run_scenario_on_erlang("6D", SCENARIO_6D)
    
    run_scenario_on_erlang(7, SCENARIO_7)
    run_scenario_on_erlang(8, SCENARIO_8)

if __name__ == "__main__":
    run_all()
