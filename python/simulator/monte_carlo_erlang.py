#!/usr/bin/env python3
"""
Monte Carlo Simulation for Erlang B (M/M/c/c) Queueing Model and Cohort Aggregation Error Quantification.
"""

import math
import numpy as np
import heapq
from typing import Dict, List, Tuple


def erlang_b(A: float, c: int) -> float:
    """Numerical recursive Erlang B formula."""
    if c == 0:
        return 1.0
    if A <= 0.0:
        return 0.0
    b = 1.0
    for i in range(1, c + 1):
        b = (A * b) / (i + A * b)
    return b


def find_min_channels(A: float, target_pb: float = 0.01, max_c: int = 1000) -> int:
    """Finds minimum channels c such that Erlang_B(A, c) <= target_pb."""
    if A <= 0:
        return 0
    for c in range(1, max_c + 1):
        if erlang_b(A, c) <= target_pb:
            return c
    return max_c


def simulate_individual_arrivals(
    A: float,
    channels: int,
    num_warmup: int = 100_000,
    num_measured: int = 2_000_000,
    num_batches: int = 20,
    mu: float = 1.0,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Simulates continuous-time M/M/c/c queue with individual Poisson arrivals.
    Uses a warmup phase to reach steady-state, followed by the Method of Batch Means
    to compute reliable 95% Confidence Intervals for autocorrelated queueing processes.
    Returns: (empirical_pb, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    lam = A * mu
    total_arrivals = num_warmup + num_measured
    
    inter_arrivals = rng.exponential(1.0 / lam, size=total_arrivals)
    service_times = rng.exponential(1.0 / mu, size=total_arrivals)
    
    current_time = 0.0
    busy_departures = []  # min-heap of active departure times
    
    # Warmup phase
    for i in range(num_warmup):
        current_time += inter_arrivals[i]
        while busy_departures and busy_departures[0] <= current_time:
            heapq.heappop(busy_departures)
        if len(busy_departures) < channels:
            heapq.heappush(busy_departures, current_time + service_times[i])
            
    # Measurement phase with Batch Means
    batch_size = num_measured // num_batches
    batch_rates = []
    
    for b in range(num_batches):
        batch_blocked = 0
        for j in range(batch_size):
            idx = num_warmup + b * batch_size + j
            current_time += inter_arrivals[idx]
            while busy_departures and busy_departures[0] <= current_time:
                heapq.heappop(busy_departures)
            if len(busy_departures) < channels:
                heapq.heappush(busy_departures, current_time + service_times[idx])
            else:
                batch_blocked += 1
        batch_rates.append(batch_blocked / float(batch_size))
        
    p_emp = float(np.mean(batch_rates))
    # Student-t critical value for df = num_batches - 1 (df=19 -> t=2.093)
    std_err = float(np.std(batch_rates, ddof=1) / math.sqrt(num_batches))
    t_crit = 2.093 if num_batches == 20 else 1.96
    ci_low = max(0.0, p_emp - t_crit * std_err)
    ci_high = p_emp + t_crit * std_err
    return p_emp, ci_low, ci_high


def simulate_batch_arrivals(
    A: float,
    channels: int,
    batch_size: int = 10,
    num_batches: int = 50_000,
    mu: float = 1.0,
    seed: int = 42
) -> float:
    """
    Simulates M^B/M/c/c queue where arrivals occur in batches of size B.
    Total sessions = num_batches * batch_size.
    Returns: empirical session blocking probability.
    """
    rng = np.random.default_rng(seed)
    lam_batch = (A * mu) / float(batch_size)
    
    inter_arrivals = rng.exponential(1.0 / lam_batch, size=num_batches)
    
    current_time = 0.0
    busy_departures = []
    total_sessions = num_batches * batch_size
    blocked_sessions = 0
    
    for dt in inter_arrivals:
        current_time += dt
        while busy_departures and busy_departures[0] <= current_time:
            heapq.heappop(busy_departures)
            
        # For each session in the batch
        service_durations = rng.exponential(1.0 / mu, size=batch_size)
        for st in service_durations:
            if len(busy_departures) < channels:
                heapq.heappush(busy_departures, current_time + st)
            else:
                blocked_sessions += 1
                
    return blocked_sessions / float(total_sessions)


def run_monte_carlo_suite():
    print("=" * 85)
    print(" MONTE CARLO ERROR QUANTIFICATION: ERLANG B vs INDIVIDUAL / BATCH SESSIONS")
    print(" Target Blocking Probability: P_b <= 1.00% (0.0100)")
    print("=" * 85)
    print(f"{'A [Erl]':>7} | {'Channels c':>10} | {'P_b Theory':>12} | {'P_b Ind (MC)':>14} | {'95% CI':>16} | {'Rel. Error':>10}")
    print("-" * 85)
    
    test_loads = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    results = []
    for A in test_loads:
        c = find_min_channels(A, target_pb=0.01)
        pb_theory = erlang_b(A, c)
        pb_mc, ci_l, ci_h = simulate_individual_arrivals(A, c, num_warmup=100_000, num_measured=2_000_000, num_batches=20, seed=42)
        rel_err = abs(pb_mc - pb_theory) / pb_theory if pb_theory > 0 else 0.0
        
        results.append({
            "A": A,
            "c": c,
            "pb_theory": pb_theory,
            "pb_mc": pb_mc,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "rel_error": rel_err
        })
        print(f"{A:>7.1f} | {c:>10} | {pb_theory*100:>11.4f}% | {pb_mc*100:>13.4f}% | [{ci_l*100:.3f}%, {ci_h*100:.3f}%] | {rel_err*100:>9.2f}%")
        
    print("=" * 85)
    print("\nBatch Arrival Sensitivity (Impact of Cohort / Batch Clustering at A=50 Erl, c=64):")
    print("-" * 65)
    print(f"{'Batch Size B':>14} | {'P_b Batch (MC)':>16} | {'Inflation Factor':>18}")
    print("-" * 65)
    c_50 = find_min_channels(50.0, target_pb=0.01)
    base_pb = erlang_b(50.0, c_50)
    batch_results = []
    for b_size in [1, 2, 5, 10, 25, 50]:
        if b_size == 1:
            pb_b = base_pb
        else:
            pb_b = simulate_batch_arrivals(50.0, c_50, batch_size=b_size, num_batches=30_000, seed=42)
        inflation = pb_b / base_pb if base_pb > 0 else 1.0
        batch_results.append({
            "batch_size_B": b_size,
            "pb_batch_mc": pb_b,
            "inflation_factor": inflation
        })
        print(f"{b_size:>14} | {pb_b*100:>15.4f}% | {inflation:>17.2f}x")
    print("=" * 65)
    
    import json
    import os
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "monte_carlo_results.json")
    payload = {
        "description": "Monte Carlo validation of Erlang B M/M/c/c queue with warmup (100k) and Batch Means (20 batches x 100k sessions) for 95% CI",
        "individual_arrivals_ci": results,
        "batch_arrivals_sensitivity": batch_results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[INFO] Successfully saved Monte Carlo results to: {out_file}")


if __name__ == "__main__":
    run_monte_carlo_suite()

