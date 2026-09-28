
import sys
import os
from datetime import timedelta
import json

# Add current directory to path
sys.path.append(os.getcwd())

from minizinc import Instance, Model, Solver
from erlang_utils import generate_erlang_table, erlang_b
from test_scenarios import SCENARIO_1
from enum import Enum

MODEL_FILE = "model_erlang.mzn"
SOLVER_NAME = "highs"

# Parameters for adaptation
BLOCKAGE_PROB = 0.01
TRAFFIC_PER_DEMAND = 0.5

def run_adapted_scenario():
    print(f"Running Adapted Scenario 1 with P_B={BLOCKAGE_PROB}, Traffic={TRAFFIC_PER_DEMAND}")
    
    data = SCENARIO_1["data"].copy()
    
    # Generate Erlang Table
    erlang_table = generate_erlang_table(
        data["Demands_"],
        TRAFFIC_PER_DEMAND,
        BLOCKAGE_PROB
    )
    
    solver = Solver.lookup(SOLVER_NAME)
    model = Model([MODEL_FILE])
    instance = Instance(solver, model)
    
    # Set parameters
    instance["ErlangTable"] = erlang_table
    instance["BlockageProbability"] = BLOCKAGE_PROB
    
    # Standard parameters from scenario
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
    
    result = instance.solve(timeout=timedelta(seconds=60))
    
    if result.solution:
        print("Solution found!")
        print(result.solution)
        # Check specific generic model stats if available or calculate them
        # Note: model_erlang might have different output structure, need to check
    else:
        print("No solution found.")

if __name__ == "__main__":
    run_adapted_scenario()
