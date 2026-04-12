"""
Demo script: evolutionary game dynamics examples.

Run with:
    python -m examples.demo
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.core.replicator import ReplicatorODE, PRESET_GAMES
from src.core.ess import ESSChecker
from src.core.moran import MoranProcess
from src.core.jacobian import JacobianAnalyzer
from src.core.mutations import ReplicatorMutator
from src.viz.simplex import SimplexPlotter, plot_time_series


def demo_rps():
    """Rock-Paper-Scissors: closed orbits on the simplex."""
    print("=" * 60)
    print("DEMO 1: Rock-Paper-Scissors")
    print("=" * 60)

    A = PRESET_GAMES["Rock-Paper-Scissors"]["matrix"]
    labels = PRESET_GAMES["Rock-Paper-Scissors"]["strategies"]

    # ESS analysis
    checker = ESSChecker(A)
    results = checker.analyze_all()
    for r in results:
        status = "ESS" if r.is_ess else ("Nash" if r.is_nash else "---")
        print(f"  [{status:>4}] {r.label}: {r.details}")

    # Jacobian analysis
    analyzer = JacobianAnalyzer(A)
    interior = np.array([1/3, 1/3, 1/3])
    fp = analyzer.analyze_fixed_point(interior, "Interior")
    print(f"\n  Interior equilibrium stability: {fp.stability.value}")
    print(f"  Eigenvalues: {fp.eigenvalues}")

    # Phase portrait
    trajectories = [
        np.array([0.6, 0.2, 0.2]),
        np.array([0.2, 0.6, 0.2]),
        np.array([0.2, 0.2, 0.6]),
        np.array([0.1, 0.1, 0.8]),
        np.array([0.4, 0.5, 0.1]),
    ]
    plotter = SimplexPlotter(A, labels)
    fig = plotter.phase_portrait(
        trajectories=trajectories,
        title="Rock-Paper-Scissors: Closed Orbits",
    )
    fig.savefig("rps_phase_portrait.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: rps_phase_portrait.png\n")


def demo_hawk_dove():
    """Hawk-Dove: mixed ESS with V/C probability."""
    print("=" * 60)
    print("DEMO 2: Hawk-Dove (V=2, C=5)")
    print("=" * 60)

    V, C = 2.0, 5.0
    A = np.array([
        [(V - C) / 2, V],
        [0, V / 2],
    ])

    checker = ESSChecker(A)
    hd = checker.hawk_dove_ess(V, C)
    print(f"  ESS: {hd['strategy']}")
    print(f"  p*(Hawk) = V/C = {hd['p_hawk']:.4f}")

    # Time series
    ode = ReplicatorODE(A)
    x0 = np.array([0.9, 0.1])
    result = ode.solve(x0, t_span=(0.0, 30.0))
    print(f"  Final state: Hawk={result.final_state[0]:.4f}, Dove={result.final_state[1]:.4f}")
    print(f"  Expected ESS: Hawk={V/C:.4f}, Dove={1-V/C:.4f}")

    fig = plot_time_series(A, x0, strategy_labels=["Hawk", "Dove"],
                           title="Hawk-Dove Dynamics (V=2, C=5)")
    fig.savefig("hawk_dove_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: hawk_dove_timeseries.png\n")


def demo_moran():
    """Moran process: neutral drift fixation probability."""
    print("=" * 60)
    print("DEMO 3: Moran Process (Neutral Drift)")
    print("=" * 60)

    N = 20
    A = np.array([[1.0, 1.0], [1.0, 1.0]])  # Neutral
    mp = MoranProcess(A, population_size=N, intensity=0.0)

    print(f"  Population size: {N}")
    print(f"  Expected fixation prob (1/N): {1/N:.4f}")

    result = mp.fixation_probability(initial_A=1, n_simulations=20000)
    print(f"  Monte Carlo fixation prob: {result.fixation_probability:.4f}")

    exact = mp.fixation_probability_exact(initial_A=1)
    print(f"  Exact fixation prob: {exact:.4f}\n")


def demo_mutations():
    """Replicator-mutator: effect of mutation on RPS."""
    print("=" * 60)
    print("DEMO 4: Replicator-Mutator (RPS with Mutation)")
    print("=" * 60)

    A = PRESET_GAMES["Rock-Paper-Scissors"]["matrix"]

    for mu in [0.0, 0.01, 0.05, 0.1]:
        rm = ReplicatorMutator(A, mutation_rate=mu)
        result = rm.solve(np.array([0.6, 0.2, 0.2]), t_span=(0.0, 200.0))
        eq = result.x[:, -1]
        print(f"  mu={mu:.2f}: equilibrium = [{eq[0]:.4f}, {eq[1]:.4f}, {eq[2]:.4f}]")

    print()


def main():
    demo_rps()
    demo_hawk_dove()
    demo_moran()
    demo_mutations()
    print("All demos complete.")


if __name__ == "__main__":
    main()
