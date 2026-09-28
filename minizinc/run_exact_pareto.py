import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from minizinc import Model, Solver, Instance
from build_supervision_dataset_parallel import build_instance_data, load_json
from enum import Enum
import concurrent.futures


def solve_for_weights(alpha, beta, base_meta):
    solver = Solver.lookup("highs")
    model_path = Path(__file__).parent / "model_erlang.mzn"
    model = Model([model_path])

    meta = json.loads(json.dumps(base_meta))
    meta["config"]["alpha_cost"] = alpha
    meta["config"]["beta_latency"] = beta

    instance = Instance(solver, model)
    data = build_instance_data(meta)
    instance["ResourceTypes"] = Enum("ResourceTypes", ["vCPU", "RAM"])
    for k, v in data.items():
        if k in [
            "SRSResources",
            "MaxLatency",
            "TransmissionCost",
            "LatencyPer10km",
            "MigrationCost",
            "UnservedPenalty",
            "Clusters_",
            "Pops_",
            "TotalDemands",
            "PopDemands",
            "Resources",
            "Distances",
            "InitialPlacement",
            "DeploymentCost",
            "Alpha_Cost",
            "Beta_Latency",
            "ErlangTable",
            "BlockageProbability",
            "SRSCapacity",
            "OriginDistances",
        ]:
            instance[k] = v

    # TIMEOUT=None: strict mathematical optimum for MILP
    res = instance.solve(timeout=None)

    if res.solution is not None:
        try:
            labels = json.loads(res.solution._output_item)
            return {
                "alpha": alpha,
                "beta": beta,
                "cost": labels["totalCost"],
                "latency": labels["totalLatency"],
                "norm_cost": labels["scores"]["normalizedCost"],
                "norm_latency": labels["scores"]["normalizedLatency"],
            }
        except Exception:
            return None
    return None


CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "exact_pareto_results.json"
)
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tex", "img")


def run_exact(force=False):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force:
        print(f"Loading saved Pareto results from {CACHE_PATH}...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        case_path = os.path.join(
            os.path.dirname(__file__), "generated_test", "reference_case_006.json"
        )
        if not os.path.exists(case_path):
            print(f"Error: {case_path} not found")
            return

        base_meta = load_json(Path(case_path))
        alphas = np.linspace(0.0, 1.0, 51)
        results = []
        print(
            f"Running Pareto experiment (N={len(alphas)} points, MILP solver without time limit)..."
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            futures = []
            for a in alphas:
                futures.append(
                    executor.submit(
                        solve_for_weights, float(a), float(1.0 - a), base_meta
                    )
                )

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res:
                    results.append(res)
                print(f"Completed {i + 1}/{len(alphas)}")

        results.sort(key=lambda x: x["alpha"])
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Saved results to {CACHE_PATH}")

    results.sort(key=lambda x: x["alpha"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "mathtext.fontset": "cm",
            "lines.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    # Subtle, deterministic jitter to make overlapping points visible
    np.random.seed(42)
    lats_raw = np.array([r["norm_latency"] for r in results])
    costs_raw = np.array([r["norm_cost"] for r in results])
    alphas = np.array([r["alpha"] for r in results])

    jitter_x = np.random.normal(0, 0.022, len(results))
    jitter_y = np.random.normal(0, 0.18, len(results))

    lats_jit = lats_raw + jitter_x
    costs_jit = costs_raw + jitter_y

    # Circles denoting physical vertices of the MILP feasible space
    ax.scatter(
        [0.0],
        [941.38],
        s=650,
        facecolors="none",
        edgecolors="#2b5c8f",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        zorder=2,
    )
    ax.scatter(
        [2.4455],
        [929.92],
        s=1600,
        facecolors="none",
        edgecolors="#c0392b",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        zorder=2,
    )

    # Drawing points preserving the alpha weight palette
    sc = ax.scatter(
        lats_jit,
        costs_jit,
        c=alphas,
        cmap="coolwarm",
        alpha=0.85,
        edgecolors="black",
        linewidth=0.6,
        s=48,
        zorder=4,
    )

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Waga $\alpha$ (koszt) [$\beta = 1 - \alpha$]", labelpad=10)

    # Annotations with arrows for both clusters (with arrows reaching the edge of the circles)
    ax.annotate(
        r"Cluster 1: 9 points"
        + "\n"
        + r"($\alpha \in [0{,}00; 0{,}16], \ \beta \in [0{,}84; 1{,}00]$)"
        + "\n"
        + r"Latency priority $\approx 0{,}0$",
        xy=(0.0, 941.38),
        xytext=(0.35, 942.0),
        arrowprops=dict(
            arrowstyle="->",
            color="#1e4369",
            lw=1.3,
            shrinkB=16,
            connectionstyle="arc3,rad=-0.1",
        ),
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f0f4f8",
            edgecolor="#2b5c8f",
            alpha=0.92,
        ),
        fontsize=9.5,
        color="#1e4369",
        zorder=5,
    )

    ax.annotate(
        r"Skupisko 2: 42 punkty"
        + "\n"
        + r"($\alpha \in [0{,}18; 1{,}00], \ \beta \in [0{,}00; 0{,}82]$)"
        + "\n"
        + r"Priorytet kosztu $\approx 930$",
        xy=(2.4455, 929.92),
        xytext=(1.25, 933.5),
        arrowprops=dict(
            arrowstyle="->",
            color="#7a2416",
            lw=1.3,
            shrinkB=24,
            connectionstyle="arc3,rad=-0.1",
        ),
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fdf2f0",
            edgecolor="#c0392b",
            alpha=0.92,
        ),
        fontsize=9.5,
        color="#7a2416",
        zorder=5,
    )

    ax.set_xlabel("Normalized latency")
    ax.set_ylabel("Znormalizowany koszt")
    ax.set_xlim(-0.25, 2.75)
    ax.set_ylim(927.0, 945.5)
    ax.grid(True, linestyle="--", alpha=0.55)

    out_pdf = os.path.join(IMG_DIR, "wrazliwosc_pareto_dokladna.pdf")
    out_png = os.path.join(IMG_DIR, "wrazliwosc_pareto_dokladna.png")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved chart to {out_pdf} and {out_png}")


if __name__ == "__main__":
    force_recompute = "--force" in sys.argv
    run_exact(force=force_recompute)
