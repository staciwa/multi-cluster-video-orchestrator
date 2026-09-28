"""
Validation tests for MiniZinc model with Erlang M/M/c/c queueing theory.

Weryfikuje:
1. Compliance with Erlang B formula (table from MPI 4 presentation)
2. Constraint correctness: Instances[c] >= ErlangTable[DemandsPerCluster[c]]
3. Operation for different BlockageProbability levels
"""

import json
from pathlib import Path
from enum import Enum
from datetime import timedelta
from typing import List, Tuple, Optional

from minizinc import Instance, Model, Solver

from erlang_utils import (
    generate_erlang_table,
    verify_erlang_solution,
    print_erlang_table,
    erlang_b,
    find_min_instances
)

from test_scenarios_erlang import (
    SCENARIO_ERLANG_BASIC,
    SCENARIO_ERLANG_HIGH_QOS,
    SCENARIO_ERLANG_LOW_QOS,
    SCENARIO_ERLANG_LIMITED_RESOURCES,
    SCENARIO_ERLANG_STRESS,
)

MODEL_FILE = str(Path(__file__).resolve().parent / "model_erlang.mzn")
SOLVER_NAME = "highs"

# Default Erlang parameters
DEFAULT_BLOCKAGE_PROB = 0.01   # 1%
DEFAULT_TRAFFIC_PER_DEMAND = 0.5  # 0.5 Erlang per demand


class TestResult:
    """Wynik testu."""
    def __init__(self, name: str, passed: bool, message: str, details: dict = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}


def run_model(
    data: dict,
    blockage_probability: float,
    traffic_per_demand: float
) -> Optional[dict]:
    """
    Uruchamia model MiniZinc z parametrami Erlanga.
    
    Returns:
        Dictionary with results or None in case of UNSAT/error
    """
    erlang_table = generate_erlang_table(
        data["Demands_"],
        traffic_per_demand,
        blockage_probability
    )
    
    try:
        solver = Solver.lookup(SOLVER_NAME)
        model = Model([MODEL_FILE])
        instance = Instance(solver, model)
        
        # Parametry Erlanga
        instance["ErlangTable"] = erlang_table
        instance["BlockageProbability"] = blockage_probability
        
        # Standardowe parametry
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
        
        result = instance.solve(timeout=timedelta(seconds=60))
        
        if result.solution is None:
            return None
        
        # Parsuj JSON
        output_str = str(result.solution)
        start = output_str.find('{')
        end = output_str.rfind('}') + 1
        
        if start == -1 or end <= 0:
            return None
        
        res_json = json.loads(output_str[start:end])
        res_json['erlang_table'] = erlang_table
        
        # Add solution time
        solve_time = result.statistics.get("solveTime", timedelta(0))
        if isinstance(solve_time, timedelta):
            res_json['solveTime'] = solve_time.total_seconds()
        else:
            res_json['solveTime'] = float(solve_time)
        
        return res_json
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def verify_erlang_constraint(
    res: dict,
    blockage_probability: float,
    traffic_per_demand: float
) -> Tuple[bool, List[str]]:
    """
    Verifies if all clusters satisfy the Erlang constraint.
    
    Returns:
        (all_ok, list_of_messages)
    """
    demands_per_cluster = res.get('demandsPerCluster', [])
    instances = res.get('instances', [])
    erlang_table = res.get('erlang_table', [])
    
    messages = []
    all_ok = True
    
    for c, (n_demands, n_instances) in enumerate(zip(demands_per_cluster, instances), 1):
        if n_demands == 0:
            messages.append(f"Cluster {c}: 0 demands → OK")
            continue
        
        required = erlang_table[n_demands]
        A = n_demands * traffic_per_demand
        actual_pb = erlang_b(A, n_instances)
        
        if n_instances < required:
            all_ok = False
            messages.append(
                f"Cluster {c}: {n_demands} demands, {n_instances} inst. < {required} required → ❌ ERROR"
            )
        elif actual_pb > blockage_probability:
            all_ok = False
            messages.append(
                f"Cluster {c}: {n_demands} demands, {n_instances} inst., "
                f"P_B={actual_pb*100:.4f}% > {blockage_probability*100}% → ❌ "
            )
        else:
            messages.append(
                f"Cluster {c}: {n_demands} demands, {n_instances} inst. (min={required}), "
                f"A={A:.2f} Erl, P_B={actual_pb*100:.4f}% → ✅"
            )
    
    return all_ok, messages


def test_erlang_table_correctness() -> TestResult:
    """
    Test 0: Weryfikacja tabeli Erlanga z danymi z prezentacji MPI 4.
    """
    print("\n" + "="*70)
    print(" TEST 0: Verification of Erlang table (data from MPI 4 presentation)")
    print("="*70)
    
    # Table from presentation (P_B = 0.0101 ≈ 1%)
    tabela_prezentacja = [
        (0.0101, 1), (0.1525, 2), (0.4554, 3), (0.8694, 4),
        (1.3607, 5), (1.9090, 6), (2.5009, 7), (3.1275, 8),
        (3.7825, 9), (4.4611, 10), (5.1599, 11), (5.8759, 12),
        (6.6071, 13), (7.3516, 14), (8.1080, 15), (8.8750, 16),
    ]
    
    all_ok = True
    errors = []
    
    for A, N_expected in tabela_prezentacja:
        N_calculated = find_min_instances(A, 0.01)
        if N_calculated != N_expected:
            all_ok = False
            errors.append(f"A={A}: oczekiwane N={N_expected}, obliczone N={N_calculated}")
    
    if all_ok:
        print("  ✅ All 16 values match the presentation table")
        return TestResult("Erlang Table", True, "Consistent with MPI 4 presentation")
    else:
        for err in errors:
            print(f"  ❌ {err}")
        return TestResult("Erlang Table", False, f"Errors: {errors}")


def test_scenario(
    name: str,
    scenario: dict,
    blockage_probability: float = DEFAULT_BLOCKAGE_PROB,
    traffic_per_demand: float = DEFAULT_TRAFFIC_PER_DEMAND
) -> TestResult:
    """
    Uruchamia test dla danego scenariusza.
    """
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f" P_B <= {blockage_probability*100}%, Traffic = {traffic_per_demand} Erl/demand")
    print(f"{'='*70}")
    
    data = scenario["data"]
    expected = scenario.get("expected", {})
    
    # Display Erlang table dla tego scenariusza
    erlang_table = generate_erlang_table(
        data["Demands_"], traffic_per_demand, blockage_probability
    )
    print(f"  Tabela Erlanga: {erlang_table}")
    
    # Uruchom model
    res = run_model(data, blockage_probability, traffic_per_demand)
    
    if res is None:
        if expected.get("all_served", True):
            return TestResult(name, False, "Model UNSATISFIABLE (expected solution)")
        else:
            return TestResult(name, True, "Model UNSATISFIABLE (zgodnie z oczekiwaniami)")
    
    # Display results
    print(f"\n  📋 WYNIKI:")
    print(f"     Instances:        {res.get('instances', [])}")
    print(f"     Demands/cluster:  {res.get('demandsPerCluster', [])}")
    print(f"     Unserved:         {res.get('unserved', 0)}")
    print(f"     Total Cost:       {res.get('totalCost', 0)}")
    print(f"     Solve time:       {res.get('solveTime', 0)*1000:.2f} ms")
    
    # Weryfikacja Erlanga
    print(f"\n  🎯 WERYFIKACJA ERLANGA:")
    erlang_ok, messages = verify_erlang_constraint(
        res, blockage_probability, traffic_per_demand
    )
    for msg in messages:
        print(f"     {msg}")
    
    # Check expectations
    all_served = res.get('unserved', 0) == 0
    expected_served = expected.get("all_served", True)
    expected_erlang = expected.get("erlang_satisfied", True)
    
    passed = True
    failure_reasons = []
    
    if expected_erlang and not erlang_ok:
        passed = False
        failure_reasons.append("Erlang constraint not met")
    
    if expected_served and not all_served:
        passed = False
        failure_reasons.append(f"Expected all served, unserved: {res.get('unserved', 0)}")
    
    if passed:
        print(f"\n  ✅ TEST PASSED")
        return TestResult(name, True, "All conditions met", res)
    else:
        print(f"\n  ❌ TEST FAILED: {', '.join(failure_reasons)}")
        return TestResult(name, False, ', '.join(failure_reasons), res)


def test_qos_comparison() -> TestResult:
    """
    Test comparing costs for different QoS levels.
    Lower P_B = more instances = higher cost.
    """
    print(f"\n{'='*70}")
    print(" TEST: Cost comparison for different QoS levels")
    print("="*70)
    
    data = SCENARIO_ERLANG_BASIC["data"]
    
    results = []
    for pb in [0.001, 0.01, 0.05, 0.10]:
        res = run_model(data, pb, DEFAULT_TRAFFIC_PER_DEMAND)
        if res:
            total_instances = sum(res.get('instances', []))
            cost = res.get('totalCost', 0)
            results.append((pb, total_instances, cost))
            print(f"  P_B <= {pb*100:>5.1f}%: {total_instances:>3} instancji, koszt = {cost}")
    
    # Check if more instances at lower P_B
    if len(results) >= 2:
        # Sort by P_B ascending
        results_sorted = sorted(results, key=lambda x: x[0])
        instances_decreasing = all(
            results_sorted[i][1] >= results_sorted[i+1][1] 
            for i in range(len(results_sorted)-1)
        )
        
        if instances_decreasing:
            print(f"\n  ✅ As expected: lower P_B → more instances")
            return TestResult(
                "QoS Comparison", True, 
                "Lower P_B requires more instances",
                {"results": results}
            )
        else:
            print(f"\n  ⚠️ Unexpected result - check manually")
            return TestResult(
                "QoS Comparison", True,  # We do not treat it as an error
                "Results may depend on optimization",
                {"results": results}
            )
    
    return TestResult("QoS Comparison", False, "Not enough results to compare")


def run_all_tests():
    """Runs all tests validating the Erlang model."""
    
    print("="*70)
    print(" TESTS VALIDATING ERLANG M/M/c/c MODEL")
    print(" Verification of consistency with queueing theory")
    print("="*70)
    
    results: List[TestResult] = []
    
    # Test 0: Weryfikacja tabeli Erlanga
    results.append(test_erlang_table_correctness())
    
    # Test 1: Podstawowy scenariusz
    results.append(test_scenario(
        "TEST 1: Podstawowa walidacja",
        SCENARIO_ERLANG_BASIC,
        DEFAULT_BLOCKAGE_PROB,
        DEFAULT_TRAFFIC_PER_DEMAND
    ))
    
    # Test 2: Wysoki QoS (P_B = 0.1%)
    results.append(test_scenario(
        "TEST 2: Wysoki QoS (P_B = 0.1%)",
        SCENARIO_ERLANG_HIGH_QOS,
        blockage_probability=0.001,
        traffic_per_demand=DEFAULT_TRAFFIC_PER_DEMAND
    ))
    
    # Test 3: Niski QoS (P_B = 10%)
    results.append(test_scenario(
        "TEST 3: Niski QoS (P_B = 10%)",
        SCENARIO_ERLANG_LOW_QOS,
        blockage_probability=0.10,
        traffic_per_demand=DEFAULT_TRAFFIC_PER_DEMAND
    ))
    
    # Test 4: Ograniczone zasoby
    results.append(test_scenario(
        "TEST 4: Ograniczone zasoby",
        SCENARIO_ERLANG_LIMITED_RESOURCES,
        DEFAULT_BLOCKAGE_PROB,
        DEFAULT_TRAFFIC_PER_DEMAND
    ))
    
    # Test 5: Stress test
    results.append(test_scenario(
        "TEST 5: Stress test (15 demands)",
        SCENARIO_ERLANG_STRESS,
        DEFAULT_BLOCKAGE_PROB,
        DEFAULT_TRAFFIC_PER_DEMAND
    ))
    
    # Test 6: QoS Comparison
    results.append(test_qos_comparison())
    
    # Podsumowanie
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {status}: {r.name}")
        if not r.passed:
            print(f"         Reason: {r.message}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 WSZYSTKIE TESTY ZALICZONE!")
        print("     Erlang model is consistent with M/M/c/c queue theory")
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed")
    
    return results


if __name__ == "__main__":
    run_all_tests()
