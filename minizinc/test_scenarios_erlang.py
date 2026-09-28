"""
Test scenarios for MiniZinc model with Erlang M/M/c/c queueing theory.

Each scenario tests a different aspect of the Erlang model:
1. Basic validation - do instances satisfy P_B <= BlockageProbability
2. Different QoS levels (BlockageProbability)
3. Limited resources - does the model cope when QoS cannot be met
4. Cost comparison for different P_B
"""

import random

LATENCY_UNIT_SCALE = 10  # 0.1 ms units

# ============================================================================
# SCENARIUSZ 1: Podstawowa walidacja Erlanga
# 10 demands, 4 clusters - checks if instances satisfy P_B
# ============================================================================
SCENARIO_ERLANG_BASIC = {
    "name": "Basic Erlang validation (10 demands, 4 clusters)",
    "description": "Sprawdza czy liczba instancji gwarantuje P_B <= BlockageProbability",
    "data": {
        "SRSResources": [2, 4],
        "MaxLatency": 250 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 500,
        "Clusters_": 4,
        "Demands_": 10,
        "Resources": [
            [16, 32],   # max 8 instancji
            [20, 40],   # max 10 instancji
            [24, 48],   # max 12 instancji
            [20, 40],   # max 10 instancji
        ],
        "DeploymentCost": [60, 50, 40, 45],
        "Distances": [
            [20, 80, 150, 200],
            [30, 70, 140, 190],
            [60, 30, 100, 150],
            [70, 40, 90, 140],
            [100, 60, 50, 100],
            [120, 80, 40, 90],
            [150, 100, 60, 50],
            [160, 110, 70, 40],
            [80, 60, 120, 110],
            [90, 55, 115, 105],
        ],
        "InitialPlacement": [[0]*4 for _ in range(10)],
        "Alpha_Cost": 0.7,
        "Beta_Latency": 0.3
    },
    "expected": {
        "all_served": True,
        "erlang_satisfied": True,  # all clusters must satisfy P_B
    }
}

# ============================================================================
# SCENARIUSZ 2: Wysoki QoS (P_B = 0.1%)
# Requires more instances than with lower QoS
# ============================================================================
SCENARIO_ERLANG_HIGH_QOS = {
    "name": "High QoS - P_B = 0.1% (8 demands, 3 clusters)",
    "description": "At low P_B more instances are needed",
    "blockage_probability": 0.001,  # 0.1%
    "data": {
        "SRSResources": [2, 4],
        "MaxLatency": 200 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 1000,
        "Clusters_": 3,
        "Demands_": 8,
        "Resources": [
            [20, 40],   # max 10 instancji
            [24, 48],   # max 12 instancji
            [20, 40],   # max 10 instancji
        ],
        "DeploymentCost": [50, 40, 45],
        "Distances": [
            [30, 100, 150],
            [35, 90, 145],
            [100, 25, 120],
            [110, 30, 115],
            [140, 120, 28],
            [145, 125, 32],
            [80, 70, 90],
            [85, 75, 85],
        ],
        "InitialPlacement": [[0]*3 for _ in range(8)],
        "Alpha_Cost": 0.5,
        "Beta_Latency": 0.5
    },
    "expected": {
        "all_served": True,
        "erlang_satisfied": True,
    }
}

# ============================================================================
# SCENARIUSZ 3: Niski QoS (P_B = 10%)
# Requires fewer instances - cheaper solution
# ============================================================================
SCENARIO_ERLANG_LOW_QOS = {
    "name": "Low QoS - P_B = 10% (8 demands, 3 clusters)",
    "description": "Przy wysokim P_B potrzeba mniej instancji",
    "blockage_probability": 0.10,  # 10%
    "data": {
        "SRSResources": [2, 4],
        "MaxLatency": 200 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 1000,
        "Clusters_": 3,
        "Demands_": 8,
        "Resources": [
            [20, 40],
            [24, 48],
            [20, 40],
        ],
        "DeploymentCost": [50, 40, 45],
        "Distances": [
            [30, 100, 150],
            [35, 90, 145],
            [100, 25, 120],
            [110, 30, 115],
            [140, 120, 28],
            [145, 125, 32],
            [80, 70, 90],
            [85, 75, 85],
        ],
        "InitialPlacement": [[0]*3 for _ in range(8)],
        "Alpha_Cost": 0.5,
        "Beta_Latency": 0.5
    },
    "expected": {
        "all_served": True,
        "erlang_satisfied": True,
    }
}

# ============================================================================
# SCENARIUSZ 4: Ograniczone zasoby
# Clusters do not have enough resources - some demands unserved
# ============================================================================
SCENARIO_ERLANG_LIMITED_RESOURCES = {
    "name": "Limited resources (6 demands, 2 small clusters)",
    "description": "Checks behavior when resources are insufficient for QoS",
    "data": {
        "SRSResources": [2, 4],
        "MaxLatency": 200 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 500,
        "Clusters_": 2,
        "Demands_": 6,
        "Resources": [
            [6, 12],    # max 3 instances - not enough for 6 demands with P_B=1%
            [8, 16],    # max 4 instancje
        ],
        "DeploymentCost": [50, 50],
        "Distances": [
            [30, 100],
            [40, 90],
            [50, 80],
            [60, 70],
            [70, 60],
            [80, 50],
        ],
        "InitialPlacement": [[0]*2 for _ in range(6)],
        "Alpha_Cost": 0.5,
        "Beta_Latency": 0.5
    },
    "expected": {
        "all_served": False,  # some demands may be unserved
        "erlang_satisfied": True,  # but those served must have P_B OK
    }
}

# ============================================================================
# SCENARIO 5: Heavy load
# Stress test - many demands
# ============================================================================
SCENARIO_ERLANG_STRESS = {
    "name": "Stress test (15 demands, 4 clusters)",
    "description": "Performance test with a larger number of demands",
    "data": {
        "SRSResources": [2, 4],
        "MaxLatency": 300 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 500,
        "Clusters_": 4,
        "Demands_": 15,
        "Resources": [
            [20, 40],   # max 10 instancji
            [24, 48],   # max 12 instancji
            [30, 60],   # max 15 instancji
            [20, 40],   # max 10 instancji
        ],
        "DeploymentCost": [60, 50, 35, 45],
        "Distances": [
            [20, 80, 150, 200],
            [30, 70, 140, 190],
            [40, 60, 130, 180],
            [60, 30, 100, 150],
            [70, 40, 90, 140],
            [80, 50, 80, 130],
            [100, 60, 50, 100],
            [120, 80, 40, 90],
            [130, 90, 35, 80],
            [150, 100, 60, 50],
            [160, 110, 70, 40],
            [170, 120, 80, 35],
            [80, 60, 120, 110],
            [90, 55, 115, 105],
            [95, 50, 110, 100],
        ],
        "InitialPlacement": [[0]*4 for _ in range(15)],
        "Alpha_Cost": 0.6,
        "Beta_Latency": 0.4
    },
    "expected": {
        "all_served": True,
        "erlang_satisfied": True,
    }
}

# ============================================================================
# SCENARIO 6: Extreme load (Macro Scaling)
# Many demands, SRSCapacity > 1
# ============================================================================
SCENARIO_ERLANG_MACRO_STRESS = {
    "name": "Macro Stress Test (300 demands, 4 clusters, SRSCapacity=20)",
    "description": "Test validating high load (A > 100 Erl) with instance capacity > 1",
    "data": {
        "SRSCapacity": 20,
        "SRSResources": [2, 4],
        "MaxLatency": 300 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 500,
        "Clusters_": 4,
        "Demands_": 300,
        "Resources": [
            [200, 400],   # max 100 instancji
            [240, 480],   # max 120 instancji
            [300, 600],   # max 150 instancji
            [200, 400],   # max 100 instancji
        ],
        "DeploymentCost": [60, 50, 35, 45],
        "Distances": [[random.randint(10, 200) for _ in range(4)] for _ in range(300)],
        "InitialPlacement": [[0]*4 for _ in range(300)],
        "Alpha_Cost": 0.6,
        "Beta_Latency": 0.4
    },
    "expected": {
        "erlang_satisfied": True,
    }
}
