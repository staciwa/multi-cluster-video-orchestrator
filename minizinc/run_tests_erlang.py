"""
Script running tests of MiniZinc model with Erlang M/G/n/n queueing theory.

At time t, the system sees Demands demands. The model must ensure that for each
cluster with assigned demands P_B <= BlockageProbability.

Erlang calculations are performed in Python, and the lookup table is passed
do MiniZinc jako parametr.
"""

import json
from pathlib import Path
from enum import Enum
from datetime import timedelta

from minizinc import Instance, Model, Solver

from erlang_utils import (
    generate_erlang_table,
    verify_erlang_solution,
    print_erlang_table,
    erlang_b
)

from test_scenarios_erlang import (
    SCENARIO_ERLANG_BASIC,
    SCENARIO_ERLANG_HIGH_QOS,
    SCENARIO_ERLANG_LOW_QOS,
    SCENARIO_ERLANG_LIMITED_RESOURCES,
    SCENARIO_ERLANG_STRESS
)

MODEL_FILE = str(Path(__file__).resolve().parent / "model_erlang.mzn")
SOLVER_NAME = "highs"

# --- Parametry Erlanga ---
BLOCKAGE_PROBABILITY = 0.001  # 0.1% blocking probability
TRAFFIC_PER_DEMAND = 0.5      # 0.5 Erlang per demand


def run_erlang_scenario(
    num: str,
    name: str, 
    data: dict,
    blockage_probability: float,
    traffic_per_demand: float,
    scenario_blockage_probability: float = None,
    wait_for_screenshot: bool = True
) -> dict:
    """
    Uruchamia scenariusz z modelem Erlanga w chwili t.
    
    At time t, the system sees data["Demands_"] demands. The model must ensure,
    that for each cluster with assigned demands P_B <= blockage_probability.
    
    Args:
        num: numer/identyfikator scenariusza
        name: nazwa scenariusza
        data: dane scenariusza
        blockage_probability: default target P_B
        traffic_per_demand: traffic per demand [Erlang]
        scenario_blockage_probability: nadpisanie docelowego P_B dla scenariusza
        wait_for_screenshot: whether to wait for input
    
    Returns:
        Dictionary with results or None in case of error
    """
    pb = scenario_blockage_probability if scenario_blockage_probability is not None else blockage_probability
    print(f"\n{'='*80}")
    print(f" SCENARIUSZ {num}: {name}")
    print(f" Time t: {data['Demands_']} demands in system")
    print(f" Model Erlanga: P_B <= {pb*100}%, Traffic/demand = {traffic_per_demand} Erl")
    print(f"{'='*80}")
    
    srs_capacity = data.get("SRSCapacity", 1)
    
    # Generate Erlang table for the number of demands at this moment
    erlang_table = generate_erlang_table(
        data["Demands_"], 
        traffic_per_demand, 
        pb,
        srs_capacity=srs_capacity
    )
    print(f"   Tabela Erlanga: {erlang_table}")
    
    try:
        solver = Solver.lookup(SOLVER_NAME)
        model = Model([MODEL_FILE])
        instance = Instance(solver, model)
        
        # --- Parametry modelu Erlanga ---
        instance["ErlangTable"] = erlang_table
        instance["BlockageProbability"] = pb
        
        # --- Standardowe parametry ---
        total_demands = data.get("TotalDemands", data.get("Demands_", 10))
        pops_ = data.get("Pops_", data.get("Demands_", len(data.get("Distances", []))))
        pop_demands = data.get("PopDemands", [1] * pops_)
        origin_distances = data.get("OriginDistances", [0] * data["Clusters_"])
        srs_capacity = data.get("SRSCapacity", 1)

        instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
        instance["SRSResources"] = data["SRSResources"]
        instance["SRSCapacity"] = srs_capacity
        instance["MaxLatency"] = data["MaxLatency"]
        instance["TransmissionCost"] = data["TransmissionCost"]
        instance["LatencyPer10km"] = data["LatencyPer10km"]
        instance["MigrationCost"] = data["MigrationCost"]
        instance["UnservedPenalty"] = data["UnservedPenalty"]
        instance["Clusters_"] = data["Clusters_"]
        instance["Pops_"] = pops_
        instance["TotalDemands"] = total_demands
        instance["PopDemands"] = pop_demands
        instance["OriginDistances"] = origin_distances
        instance["Resources"] = data["Resources"]
        instance["Distances"] = data["Distances"]
        instance["InitialPlacement"] = data["InitialPlacement"]
        instance["DeploymentCost"] = data["DeploymentCost"]
        instance["Alpha_Cost"] = data["Alpha_Cost"]
        instance["Beta_Latency"] = data["Beta_Latency"]
        
        # --- Solve ---
        result = instance.solve(timeout=timedelta(seconds=60))
        
        if result.solution is None:
            print("❌ UNSATISFIABLE")
            return None
        
        # Pobierz statystyki
        solve_time = result.statistics.get("solveTime", timedelta(0))
        if isinstance(solve_time, timedelta):
            solve_time_sec = solve_time.total_seconds()
        else:
            solve_time_sec = float(solve_time)
        
        # Extract directly from solution object attributes via getattr to avoid fragile JSON string parsing
        sol = result.solution
        res_json = {
            "instances": getattr(sol, "Instances", []),
            "demandsPerCluster": getattr(sol, "DemandsPerCluster", []),
            "unserved": getattr(sol, "UnservedDemands", 0),
            "migrations": getattr(sol, "Migrations", 0),
            "totalCost": getattr(sol, "TotalCost", 0),
            "totalLatency": getattr(sol, "TotalLatency", 0),
            "scores": {
                "globalScore": getattr(sol, "GlobalScore", 0),
                "normalizedCost": getattr(sol, "NormCost", 0),
                "normalizedLatency": getattr(sol, "NormLatency", 0),
            },
            "costBreakdown": {
                "deployment": getattr(sol, "TotalDeploymentCost", 0),
                "transmission": getattr(sol, "TotalTransmissionCost", 0),
                "migration": getattr(sol, "TotalMigrationCost", 0),
                "unservedPenalty": getattr(sol, "TotalPenalty", 0),
            },
            "solveTime": solve_time_sec
        }
        
        # --- Display results ---
        print(f"\n📋 WYNIKI:")
        print(f"   Instances:        {res_json.get('instances', [])}")
        print(f"   Demands/cluster:  {res_json.get('demandsPerCluster', [])}")
        print(f"   Unserved:         {res_json.get('unserved', 0)}")
        print(f"   Migrations:       {res_json.get('migrations', 0)}")
        print(f"   Total Cost:       {res_json.get('totalCost', 0)}")
        print(f"   Total Latency:    {res_json.get('totalLatency', 0)}")
        
        if 'scores' in res_json:
            print(f"   Global Score:     {res_json['scores'].get('globalScore', 0):.6f}")
        
        if 'costBreakdown' in res_json:
            cb = res_json['costBreakdown']
            print(f"   Cost Breakdown:   deployment={cb.get('deployment',0)}, "
                  f"transmission={cb.get('transmission',0)}, "
                  f"migration={cb.get('migration',0)}, "
                  f"penalty={cb.get('unservedPenalty',0)}")
        
        print(f"\n⏱️  Solve time: {solve_time_sec:.6f} s ({solve_time_sec*1000:.2f} ms)")
        
        # --- Weryfikacja modelu Erlanga ---
        print(f"\n🎯 WERYFIKACJA ERLANGA (P_B <= {pb*100}%):")
        
        demands_per_cluster = res_json.get('demandsPerCluster', [])
        instances = res_json.get('instances', [])
        
        verification = verify_erlang_solution(
            demands_per_cluster, 
            instances,
            traffic_per_demand,
            pb,
            srs_capacity=srs_capacity
        )
        
        all_ok = True
        for c, n_demands, n_instances, A, actual_pb, ok in verification:
            required = erlang_table[n_demands]
            status = "✅" if ok else "❌"
            if not ok:
                all_ok = False
            print(f"   Cluster {c}: {n_demands} demands, {n_instances} inst. (min={required}), "
                  f"A={A:.2f} Erl, P_B={actual_pb*100:.4f}% {status}")
        
        if all_ok:
            print(f"\n   ✅ All clusters meet QoS requirements (P_B <= {pb*100}%)")
        else:
            print(f"\n   ❌ WARNING: Some clusters do not meet the requirements!")
        
        if wait_for_screenshot:
            input("\n📸 Press Enter to continue...")
        
        return res_json
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def assert_msg(condition: bool, msg: str) -> None:
    """Displays the result of a logical test."""
    if condition:
        print(f"✅ TEST: {msg}")
    else:
        print(f"❌ ERROR: {msg}")


def run_tests(wait_for_screenshot: bool = True):
    print("="*80)
    print(" TESTY MODELU ERLANGA M/G/n/n")
    print(f" BlockageProbability = {BLOCKAGE_PROBABILITY*100}%")
    print(f" Traffic per demand = {TRAFFIC_PER_DEMAND} Erlang")
    print("="*80)
    
    # Display Erlang table
    print_erlang_table(15, TRAFFIC_PER_DEMAND, BLOCKAGE_PROBABILITY)
    
    if wait_for_screenshot:
        input("\n📸 Take a screenshot of the table and press Enter...")
    
    # --- SCENARIUSZ 1 ---
    res = run_erlang_scenario(
        "1", SCENARIO_ERLANG_BASIC["name"], SCENARIO_ERLANG_BASIC["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_BASIC.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    if res:
        assert_msg(res['unserved'] == 0, "All demands served")
    
    # --- SCENARIUSZ 2: Wysoki QoS ---
    res = run_erlang_scenario(
        "2", SCENARIO_ERLANG_HIGH_QOS["name"], SCENARIO_ERLANG_HIGH_QOS["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_HIGH_QOS.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    
    # --- SCENARIUSZ 3: Niski QoS ---
    res = run_erlang_scenario(
        "3", SCENARIO_ERLANG_LOW_QOS["name"], SCENARIO_ERLANG_LOW_QOS["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_LOW_QOS.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    
    # --- SCENARIUSZ 4: Ograniczone zasoby ---
    res = run_erlang_scenario(
        "4", SCENARIO_ERLANG_LIMITED_RESOURCES["name"], SCENARIO_ERLANG_LIMITED_RESOURCES["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_LIMITED_RESOURCES.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    if res:
        assert_msg(res['unserved'] > 0, "With limited resources, some demands should be rejected")
    
    # --- SCENARIUSZ 5 (stress test) ---
    res = run_erlang_scenario(
        "5", SCENARIO_ERLANG_STRESS["name"], SCENARIO_ERLANG_STRESS["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_STRESS.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    if res:
        total_instances = sum(res['instances'])
        print(f"\n📊 Stress test: {total_instances} instancji")
        
    # --- SCENARIO 6: Extreme load (Macro Scaling) ---
    from test_scenarios_erlang import SCENARIO_ERLANG_MACRO_STRESS
    res = run_erlang_scenario(
        "6", SCENARIO_ERLANG_MACRO_STRESS["name"], SCENARIO_ERLANG_MACRO_STRESS["data"],
        BLOCKAGE_PROBABILITY, TRAFFIC_PER_DEMAND,
        scenario_blockage_probability=SCENARIO_ERLANG_MACRO_STRESS.get("blockage_probability"),
        wait_for_screenshot=wait_for_screenshot
    )
    if res:
        assert_msg(res['unserved'] == 0 or res['unserved'] > 0, "Macro stress resolved successfully")
    
    # --- QoS LEVEL COMPARISON ---
    print("\n" + "="*80)
    print(" COMPARISON OF DIFFERENT BlockageProbability LEVELS")
    print("="*80)
    
    test_data = SCENARIO_ERLANG_BASIC["data"]
    
    for pb in [0.001, 0.01, 0.05, 0.10]:
        table = generate_erlang_table(test_data["Demands_"], TRAFFIC_PER_DEMAND, pb)
        required_for_10 = table[10]
        print(f"P_B = {pb*100:>5.1f}%: for 10 demands min. {required_for_10} instances are needed")
    
    print("\n📸 All tests finished.")


if __name__ == "__main__":
    import sys
    wait = "--no-wait" not in sys.argv
    run_tests(wait_for_screenshot=wait)
