"""
Module containing test data for SRS optimization scenarios.
Each scenario tests a different aspect of the MiniZinc model.
"""

import random

LATENCY_UNIT_SCALE = 10  # 0.1 ms units

# =========================================================================
# SCENARIO 1: Load distribution in a multi-cluster environment
# =========================================================================
# 10 demands, 4 clusters with different resources and costs
# Clusters simulate: edge (close, expensive), regional (medium), cloud (distant, cheap)

SCENARIO_1 = {
    "name": "Load distribution in a multi-cluster environment (10 demands, 4 clusters)",
    "data": {
        "SRSCapacity": 3,  # each instance handles 3 demands
        "SRSResources": [2, 4],  # 2 vCPU, 4 GB RAM per instance
        "MaxLatency": 250 * LATENCY_UNIT_SCALE,  # max 250ms - realistic limit for video transmission
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,  # 1ms per 10km
        "MigrationCost": 20,
        "UnservedPenalty": 500,
        "Clusters_": 4,
        "Demands_": 10,
        "Resources": [
            [8, 16],   # Klaster 1 (Edge): 4 instancje max
            [16, 32],  # Klaster 2 (Regional): 8 instancji max
            [32, 64],  # Klaster 3 (Cloud A): 16 instancji max
            [24, 48],  # Klaster 4 (Cloud B): 12 instancji max
        ],
        "DeploymentCost": [80, 50, 30, 35],  # Edge more expensive, cloud cheaper
        # Distances: demands from various locations
        "Distances": [
            [10, 50, 200, 180],   # d1: blisko edge
            [15, 45, 190, 175],   # d2: blisko edge
            [100, 20, 150, 140],  # d3: blisko regional
            [120, 25, 145, 135],  # d4: blisko regional
            [200, 100, 30, 50],   # d5: blisko cloud A
            [210, 110, 35, 45],   # d6: blisko cloud A
            [180, 90, 40, 25],    # d7: blisko cloud B
            [190, 95, 45, 30],    # d8: blisko cloud B
            [80, 60, 120, 110],   # d9: intermediate location
            [90, 55, 115, 105],   # d10: intermediate location
        ],
        "InitialPlacement": [[0]*4 for _ in range(10)],  # new demands
        "Alpha_Cost": 1.0,
        "Beta_Latency": 0.0
    }
}

# =========================================================================
# SCENARIO 2: Resource saturation and demand selection
# =========================================================================
# 15 demands, but only 3 clusters with limited resources
# Model must decide which demands to reject

SCENARIO_2 = {
    "name": "Limited resources (15 demands, 3 clusters)",
    "data": {
        "SRSCapacity": 2,
        "SRSResources": [4, 8],  # demanding instances
        "MaxLatency": 200 * LATENCY_UNIT_SCALE,  # ograniczenie QoS
        "TransmissionCost": 3,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 30,
        "UnservedPenalty": 200,  # low penalty - worth rejecting
        "Clusters_": 3,
        "Demands_": 15,
        "Resources": [
            [12, 24],  # max 3 instances (6 demands)
            [8, 16],   # max 2 instances (4 demands)
            [16, 32],  # max 4 instances (8 demands)
        ],  # In total max 18 demands, but QoS constraints will reduce this
        "DeploymentCost": [60, 80, 40],
        "Distances": [
            [20, 100, 150],  # d1
            [25, 90, 145],   # d2
            [30, 80, 140],   # d3
            [150, 30, 100],  # d4
            [160, 25, 95],   # d5
            [170, 35, 90],   # d6
            [200, 180, 40],  # d7
            [190, 175, 35],  # d8
            [100, 100, 100], # d9 - equidistant
            [500, 500, 500], # d10 - too far for all!
            [450, 480, 520], # d11 - za daleko
            [50, 60, 200],   # d12
            [55, 65, 195],   # d13
            [180, 40, 120],  # d14
            [185, 45, 115],  # d15
        ],
        "InitialPlacement": [[0]*3 for _ in range(15)],
        "Alpha_Cost": 1.0,
        "Beta_Latency": 0.0
    }
}

# =========================================================================
# SCENARIO 3: Mass migration with cost change
# =========================================================================
# Simulation of scenario where prices in one cluster rise sharply
# 8 demands initially in Cluster 1, which is now 10x more expensive

SCENARIO_3 = {
    "name": "Mass migration (8 demands migrate from an expensive cluster)",
    "data": {
        "SRSCapacity": 4,
        "SRSResources": [2, 4],
        "MaxLatency": 60 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 1,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 15,  # umiarkowany koszt migracji
        "UnservedPenalty": 1000,
        "Clusters_": 4,
        "Demands_": 8,
        "Resources": [
            [16, 32], [16, 32], [16, 32], [16, 32]
        ],
        "DeploymentCost": [500, 50, 55, 60],  # Klaster 1 nagle bardzo drogi!
        "Distances": [
            [30, 40, 50, 60],
            [35, 45, 55, 65],
            [25, 50, 40, 55],
            [40, 35, 45, 50],
            [50, 30, 35, 45],
            [55, 40, 30, 40],
            [45, 55, 50, 35],
            [60, 50, 55, 30],
        ],
        # All demands were in Cluster 1 (now expensive)
        "InitialPlacement": [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        "Alpha_Cost": 1.0,
        "Beta_Latency": 0.0
    }
}

# =========================================================================
# SCENARIUSZ 4: Heterogeniczne klastry (edge vs cloud)
# =========================================================================
# 12 demands, 5 clusters with very different characteristics
# Edge clusters: small, expensive, close
# Cloud clusters: large, cheap, distant

SCENARIO_4 = {
    "name": "Heterogeneous edge vs cloud clusters (12 demands, 5 clusters)",
    "data": {
        "SRSCapacity": 5,
        "SRSResources": [2, 2],
        "MaxLatency": 200 * LATENCY_UNIT_SCALE,  # sufficient for all clusters
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 25,
        "UnservedPenalty": 800,
        "Clusters_": 5,
        "Demands_": 12,
        "Resources": [
            [4, 4],    # Edge 1: tylko 2 instancje
            [4, 4],    # Edge 2: tylko 2 instancje
            [6, 6],    # Edge 3: tylko 3 instancje
            [40, 40],  # Cloud 1: 20 instancji
            [60, 60],  # Cloud 2: 30 instancji
        ],
        "DeploymentCost": [100, 95, 90, 25, 20],
        "Distances": [
            [15, 60, 100, 200, 250],   # d1 - blisko Edge1
            [20, 55, 95, 190, 240],    # d2 - blisko Edge1
            [50, 10, 50, 180, 230],    # d3 - blisko Edge2
            [55, 15, 45, 175, 225],    # d4 - blisko Edge2
            [80, 40, 20, 150, 200],    # d5 - blisko Edge3
            [90, 45, 25, 145, 195],    # d6 - blisko Edge3
            [150, 140, 130, 40, 60],   # d7 - blisko Cloud1
            [160, 150, 140, 35, 55],   # d8 - blisko Cloud1
            [170, 160, 150, 45, 50],   # d9 - blisko Cloud1
            [180, 170, 160, 50, 45],   # d10 - blisko Cloud2
            [100, 80, 60, 100, 120],   # d11 - intermediate
            [110, 90, 70, 95, 115],    # d12 - intermediate
        ],
        "InitialPlacement": [[0]*5 for _ in range(12)],
        "Alpha_Cost": 0.7,
        "Beta_Latency": 0.3
    }
}

# =========================================================================
# SCENARIO 5: Stress test - large scale
# =========================================================================
# 20 demands, 6 clusters - performance and correctness test

def generate_scenario_5(seed=42):
    """Generates data for scenario 5 (stress test)."""
    random.seed(seed)
    
    num_demands = 20
    num_clusters = 6
    
    # Generate realistic distances (demand groups close to different clusters)
    distances = []
    for d in range(num_demands):
        row = []
        home_cluster = d % num_clusters  # each demand has a "home" cluster
        for c in range(num_clusters):
            if c == home_cluster:
                row.append(random.randint(20, 60))  # blisko
            else:
                row.append(random.randint(100, 400))  # daleko
        distances.append(row)
    
    return {
        "name": f"Stress test ({num_demands} demands, {num_clusters} clusters)",
        "data": {
            "SRSCapacity": 4,
            "SRSResources": [2, 4],
            "MaxLatency": 500 * LATENCY_UNIT_SCALE,  # high limit for stress test
            "TransmissionCost": 1,
            "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
            "MigrationCost": 10,
            "UnservedPenalty": 300,
            "Clusters_": num_clusters,
            "Demands_": num_demands,
            "Resources": [
                [16, 32], [20, 40], [24, 48], [16, 32], [20, 40], [24, 48]
            ],
            "DeploymentCost": [50, 45, 40, 55, 50, 35],
            "Distances": distances,
            "InitialPlacement": [[0]*num_clusters for _ in range(num_demands)],
            "Alpha_Cost": 0.5,
            "Beta_Latency": 0.5
        }
    }

# Pre-generated for consistency
SCENARIO_5 = generate_scenario_5()

# =========================================================================
# SCENARIO 6: Influence of Alpha/Beta parameter
# =========================================================================
# The same scenario run with different weights
# KEY: Cluster 1 is CHEAPEST but FARTHEST
#           Cluster 4 is MOST EXPENSIVE but CLOSEST

SCENARIO_6_BASE = {
    "SRSCapacity": 3,
    "SRSResources": [2, 4],
    "MaxLatency": 200 * LATENCY_UNIT_SCALE,  # high limit so as not to block distant clusters
    "TransmissionCost": 1,  # low transmission cost so deployment cost dominates
    "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
    "MigrationCost": 20,
    "UnservedPenalty": 5000,  # high penalty so as not to reject
    "Clusters_": 4,
    "Demands_": 8,
    "Resources": [
        [20, 40], [20, 40], [20, 40], [20, 40]
    ],
    # Cluster 1: CHEAPEST but FARTHEST (deployment)
    # Cluster 4: MOST EXPENSIVE but CLOSEST
    "DeploymentCost": [10, 30, 60, 120],  # clear difference
    "Distances": [
        [180, 120, 70, 20],   # d1 - Klaster 1 daleki (180km), Klaster 4 bliski (20km)
        [175, 115, 65, 25],   # d2
        [185, 125, 75, 15],   # d3
        [170, 110, 60, 30],   # d4
        [190, 130, 80, 10],   # d5
        [165, 105, 55, 35],   # d6
        [195, 135, 85, 18],   # d7
        [160, 100, 50, 28],   # d8
    ],
    "InitialPlacement": [[0]*4 for _ in range(8)],
}

def get_scenario_6_variant(alpha, beta):
    """Returns variant of scenario 6 with specific weights."""
    data = SCENARIO_6_BASE.copy()
    data["Alpha_Cost"] = alpha
    data["Beta_Latency"] = beta
    return data

SCENARIO_6A = {
    "name": "Priorytet KOSZT (α=1.0, β=0.0)",
    "data": get_scenario_6_variant(1.0, 0.0)
}

SCENARIO_6B = {
    "name": "LATENCY priority (α=0.2, β=0.8)",
    "data": get_scenario_6_variant(0.2, 0.8)
}

SCENARIO_6C = {
    "name": "BALANS (α=0.5, β=0.5)",
    "data": get_scenario_6_variant(0.5, 0.5)
}

SCENARIO_6D = {
    "name": "Koszt z QoS (α=0.8, β=0.2)",
    "data": get_scenario_6_variant(0.8, 0.2)
}

# =========================================================================
# SCENARIO 7: Emergency scenario - cluster unavailable
# =========================================================================
# Simulation of one cluster failure - demands must be redirected

SCENARIO_7 = {
    "name": "Failure scenario - cluster 1 unavailable (10 demands, 4 clusters)",
    "data": {
        "SRSCapacity": 3,
        "SRSResources": [2, 4],
        "MaxLatency": 100 * LATENCY_UNIT_SCALE,
        "TransmissionCost": 2,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 30,
        "UnservedPenalty": 500,
        "Clusters_": 4,
        "Demands_": 10,
        "Resources": [
            [2, 4],    # Cluster 1: only 1 instance (overload/failure simulation)
            [16, 32],  # Klaster 2: normalny
            [20, 40],  # Klaster 3: normalny
            [12, 24],  # Klaster 4: normalny
        ],
        "DeploymentCost": [5000, 50, 50, 50],  # Klaster 1: ekstremalnie drogi (symulacja awarii)
        "Distances": [
            [10, 60, 80, 100],   # d1 - was close to C1 (now unavailable)
            [15, 55, 75, 95],    # d2 - was close to C1
            [20, 50, 70, 90],    # d3 - was close to C1
            [80, 20, 40, 60],    # d4 - blisko K2
            [85, 25, 45, 65],    # d5 - blisko K2
            [100, 60, 25, 50],   # d6 - blisko K3
            [110, 70, 30, 55],   # d7 - blisko K3
            [120, 80, 50, 20],   # d8 - blisko K4
            [130, 90, 60, 25],   # d9 - blisko K4
            [50, 40, 45, 55],    # d10 - evenly
        ],
        # Demands 1-3 were in Cluster 1 (now failure)
        "InitialPlacement": [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        "Alpha_Cost": 0.6,
        "Beta_Latency": 0.4
    }
}

# =========================================================================
# SCENARIO 8: Differentiated QoS requirements
# =========================================================================
# Low MaxLatency forces the use of only close clusters

SCENARIO_8 = {
    "name": "Low MaxLatency (12 demands, 5 clusters)",
    "data": {
        "SRSCapacity": 2,
        "SRSResources": [2, 4],
        "MaxLatency": 40 * LATENCY_UNIT_SCALE,  # restrykcyjny limit QoS
        "TransmissionCost": 3,
        "LatencyPer10km": 1 * LATENCY_UNIT_SCALE,
        "MigrationCost": 20,
        "UnservedPenalty": 400,
        "Clusters_": 5,
        "Demands_": 12,
        "Resources": [
            [12, 24], [12, 24], [12, 24], [12, 24], [12, 24]
        ],
        "DeploymentCost": [40, 45, 50, 55, 60],
        # Each demand close to one cluster (15-35km), far from the rest (100-200km)
        "Distances": [
            [15, 100, 150, 180, 200],   # d1 -> K1 (15km = 15ms < 40ms OK)
            [20, 110, 160, 190, 210],   # d2 -> K1
            [120, 18, 140, 170, 195],   # d3 -> K2
            [130, 22, 145, 175, 200],   # d4 -> K2
            [150, 140, 25, 165, 185],   # d5 -> K3
            [160, 150, 30, 170, 190],   # d6 -> K3
            [175, 165, 155, 17, 180],   # d7 -> K4
            [180, 170, 160, 23, 185],   # d8 -> K4
            [190, 180, 170, 160, 16],   # d9 -> K5
            [195, 185, 175, 165, 21],   # d10 -> K5
            [500, 500, 500, 500, 500],  # d11 -> za daleko od wszystkich!
            [550, 550, 550, 550, 550],  # d12 -> za daleko od wszystkich!
        ],
        "InitialPlacement": [[0]*5 for _ in range(12)],
        "Alpha_Cost": 0.5,
        "Beta_Latency": 0.5
    }
}

# =========================================================================
# Dictionary of all scenarios for easy access
# =========================================================================

ALL_SCENARIOS = {
    1: SCENARIO_1,
    2: SCENARIO_2,
    3: SCENARIO_3,
    4: SCENARIO_4,
    5: SCENARIO_5,
    "6A": SCENARIO_6A,
    "6B": SCENARIO_6B,
    "6C": SCENARIO_6C,
    "6D": SCENARIO_6D,
    7: SCENARIO_7,
    8: SCENARIO_8,
}

def get_scenario(num):
    """Pobiera scenariusz po numerze."""
    return ALL_SCENARIOS.get(num)

def get_scenario_data(num):
    """Pobiera tylko dane scenariusza."""
    scenario = get_scenario(num)
    return scenario["data"] if scenario else None

def get_scenario_name(num):
    """Gets the scenario name."""
    scenario = get_scenario(num)
    return scenario["name"] if scenario else None
