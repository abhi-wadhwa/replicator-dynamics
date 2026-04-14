"""
Streamlit application for interactive evolutionary game dynamics.

Features:
  - Game selector: preset games or custom payoff matrix
  - Simplex phase portrait with vector field and trajectories
  - Time series of population fractions
  - ESS analysis results
  - Moran process simulation
  - Bifurcation diagram with mutation
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.core.ess import ESSChecker
from src.core.jacobian import JacobianAnalyzer
from src.core.moran import MoranProcess
from src.core.mutations import ReplicatorMutator
from src.core.replicator import PRESET_GAMES
from src.viz.simplex import (
    SimplexPlotter,
    plot_bifurcation,
    plot_moran_trajectory,
    plot_time_series,
)


def main() -> None:
    st.set_page_config(
        page_title="Replicator Dynamics",
        page_icon="🧬",
        layout="wide",
    )

    st.title("Evolutionary Game Dynamics Simulator")
    st.markdown(
        "Explore replicator equations, evolutionarily stable strategies, "
        "and stochastic Moran processes interactively."
    )

    # ── Sidebar: game selection ──────────────────────────────────────

    st.sidebar.header("Game Configuration")

    game_mode = st.sidebar.radio(
        "Game mode",
        ["Preset Games", "Custom Payoff Matrix"],
    )

    if game_mode == "Preset Games":
        game_name = st.sidebar.selectbox(
            "Select game",
            list(PRESET_GAMES.keys()),
        )
        game = PRESET_GAMES[game_name]
        payoff_matrix = game["matrix"].copy()
        strategy_labels = game["strategies"]
        st.sidebar.info(game["description"])

        # Allow tweaking preset payoff values
        st.sidebar.subheader("Adjust Payoffs")
        n = payoff_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                payoff_matrix[i, j] = st.sidebar.number_input(
                    f"A[{strategy_labels[i]},{strategy_labels[j]}]",
                    value=float(payoff_matrix[i, j]),
                    step=0.1,
                    format="%.2f",
                    key=f"payoff_{i}_{j}",
                )
    else:
        n = st.sidebar.selectbox("Number of strategies", [2, 3, 4], index=1)
        strategy_labels = [
            st.sidebar.text_input(f"Strategy {i} name", value=f"S{i}", key=f"label_{i}")
            for i in range(n)
        ]
        st.sidebar.subheader("Payoff Matrix")
        payoff_matrix = np.zeros((n, n))
        for i in range(n):
            cols = st.sidebar.columns(n)
            for j in range(n):
                payoff_matrix[i, j] = cols[j].number_input(
                    f"[{i},{j}]",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"custom_{i}_{j}",
                )

    n = payoff_matrix.shape[0]

    # ── Display payoff matrix ────────────────────────────────────────

    st.subheader("Payoff Matrix")
    import pandas as pd
    df = pd.DataFrame(
        payoff_matrix,
        index=strategy_labels,
        columns=strategy_labels,
    )
    st.dataframe(df.style.format("{:.2f}"), use_container_width=False)

    # ── Tabs for different analyses ──────────────────────────────────

    tabs = st.tabs([
        "Phase Portrait" if n == 3 else "Time Series",
        "Time Series",
        "ESS Analysis",
        "Stability (Jacobian)",
        "Moran Process",
        "Mutation-Selection",
    ])

    # ── Tab 1: Simplex phase portrait (3 strategies) or time series ──

    with tabs[0]:
        if n == 3:
            st.subheader("Simplex Phase Portrait")

            col1, col2 = st.columns([3, 1])

            with col2:
                show_vf = st.checkbox("Show vector field", value=True)
                show_fp = st.checkbox("Show fixed points", value=True)
                show_basins = st.checkbox("Show basins of attraction", value=False)
                vf_res = st.slider("Vector field density", 10, 40, 20)
                st.slider("Integration time", 10.0, 200.0, 50.0, step=5.0)

                st.markdown("**Add trajectories**")
                n_traj = st.number_input("Number of trajectories", 1, 20, 5)

                trajectories = []
                for i in range(int(n_traj)):
                    rng = np.random.default_rng(seed=42 + i)
                    default = rng.dirichlet(np.ones(3))
                    x0_val = st.text_input(
                        f"Trajectory {i+1} (x0,x1,x2)",
                        value=f"{default[0]:.3f},{default[1]:.3f},{default[2]:.3f}",
                        key=f"traj_{i}",
                    )
                    try:
                        vals = [float(v.strip()) for v in x0_val.split(",")]
                        if len(vals) == 3 and abs(sum(vals) - 1.0) < 0.01:
                            trajectories.append(np.array(vals))
                    except ValueError:
                        pass

            with col1:
                plotter = SimplexPlotter(payoff_matrix, strategy_labels)
                fig = plotter.phase_portrait(
                    trajectories=trajectories if trajectories else None,
                    show_vector_field=show_vf,
                    show_fixed_points=show_fp,
                    show_basins=show_basins,
                    vector_resolution=vf_res,
                    title="Phase Portrait",
                )
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.subheader("Time Series (Phase portrait requires 3 strategies)")
            x0_input = st.text_input(
                "Initial condition (comma-separated, must sum to 1)",
                value=",".join([f"{1/n:.3f}"] * n),
            )
            try:
                x0 = np.array([float(v.strip()) for v in x0_input.split(",")])
                if len(x0) == n and np.isclose(x0.sum(), 1.0):
                    fig = plot_time_series(
                        payoff_matrix, x0,
                        strategy_labels=strategy_labels,
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("Initial condition must have correct dimension and sum to 1.")
            except ValueError:
                st.error("Invalid input format.")

    # ── Tab 2: Time series ───────────────────────────────────────────

    with tabs[1]:
        st.subheader("Population Dynamics Over Time")

        col1, col2 = st.columns([3, 1])

        with col2:
            t_end_ts = st.slider("Time horizon", 10.0, 500.0, 50.0, step=10.0,
                                 key="ts_tend")
            x0_ts = st.text_input(
                "Initial condition",
                value=",".join([f"{1/n:.3f}"] * n),
                key="ts_x0",
            )

        with col1:
            try:
                x0 = np.array([float(v.strip()) for v in x0_ts.split(",")])
                if len(x0) == n and np.isclose(x0.sum(), 1.0):
                    fig = plot_time_series(
                        payoff_matrix, x0,
                        t_span=(0.0, t_end_ts),
                        strategy_labels=strategy_labels,
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("Initial condition dimension mismatch or doesn't sum to 1.")
            except ValueError:
                st.error("Invalid input format.")

    # ── Tab 3: ESS analysis ──────────────────────────────────────────

    with tabs[2]:
        st.subheader("Evolutionarily Stable Strategy Analysis")

        checker = ESSChecker(payoff_matrix)
        results = checker.analyze_all()

        for r in results:
            status = "ESS" if r.is_ess else ("Nash" if r.is_nash else "Not Nash")
            icon = {"ESS": "🟢", "Nash": "🟡", "Not Nash": "🔴"}[status]

            with st.expander(f"{icon} {r.label} -- {status}"):
                st.write(f"**Strategy:** {np.array2string(r.strategy, precision=4)}")
                st.write(f"**Is Nash Equilibrium:** {r.is_nash}")
                st.write(f"**Is ESS:** {r.is_ess}")
                st.write(f"**Details:** {r.details}")

        # Hawk-Dove specific analysis
        if n == 2:
            st.markdown("---")
            st.subheader("Hawk-Dove Parametric Analysis")
            V = st.number_input("Value of resource (V)", value=2.0, step=0.1)
            C = st.number_input("Cost of fighting (C)", value=3.0, step=0.1)
            hd_result = checker.hawk_dove_ess(V, C)
            st.write(f"**ESS type:** {hd_result['type']}")
            st.write(f"**Strategy:** {hd_result['strategy']}")
            st.write(f"**p*(Hawk) = V/C = {hd_result['p_hawk']:.4f}**")

    # ── Tab 4: Jacobian stability analysis ───────────────────────────

    with tabs[3]:
        st.subheader("Fixed Point Stability Analysis")

        analyzer = JacobianAnalyzer(payoff_matrix)
        fp_results = analyzer.full_analysis()

        for fp in fp_results:
            eigvals_str = ", ".join(
                [f"{e:.4f}" for e in fp.eigenvalues]
            )
            with st.expander(f"{fp.label} -- {fp.stability.value}"):
                st.write(f"**Point:** {np.array2string(fp.point, precision=4)}")
                st.write(f"**Eigenvalues:** {eigvals_str}")
                st.write(f"**Stability:** {fp.stability.value}")
                st.write("**Jacobian:**")
                st.dataframe(
                    pd.DataFrame(fp.jacobian).style.format("{:.6f}"),
                    use_container_width=False,
                )

    # ── Tab 5: Moran process ─────────────────────────────────────────

    with tabs[4]:
        st.subheader("Moran Process (Finite Population)")

        if n != 2:
            st.warning("Moran process simulation requires a 2-strategy game. "
                       "Please use a 2x2 payoff matrix.")
        else:
            col1, col2 = st.columns([3, 1])

            with col2:
                pop_size = st.slider("Population size (N)", 10, 200, 50, key="moran_N")
                intensity = st.slider("Selection intensity (w)", 0.0, 2.0, 1.0,
                                      step=0.1, key="moran_w")
                initial_A = st.slider("Initial # of type A", 1, pop_size - 1, 1,
                                      key="moran_init")
                n_sims = st.slider("Number of simulations", 100, 50000, 5000,
                                   step=100, key="moran_nsim")

                run_moran = st.button("Run Moran Simulation")

            with col1:
                if run_moran:
                    mp = MoranProcess(payoff_matrix, population_size=pop_size,
                                     intensity=intensity)

                    with st.spinner("Running Monte Carlo..."):
                        result = mp.fixation_probability(
                            initial_A=initial_A,
                            n_simulations=n_sims,
                        )

                    st.metric("Fixation Probability", f"{result.fixation_probability:.4f}")
                    st.metric("Neutral Drift (1/N)", f"{1/pop_size:.4f}")

                    exact_prob = mp.fixation_probability_exact(initial_A)
                    st.metric("Exact (analytical)", f"{exact_prob:.4f}")

                    st.write(f"Fixations: {result.n_fixations}/{result.n_simulations}")
                    st.write(f"Mean fixation time: {result.mean_fixation_time:.0f} steps")

                    # Show a sample trajectory
                    st.markdown("**Sample trajectory:**")
                    fixated, traj = mp.simulate_with_trajectory(initial_A)
                    fig = plot_moran_trajectory(traj, pop_size)
                    st.pyplot(fig)
                    plt.close(fig)

    # ── Tab 6: Mutation-selection dynamics ────────────────────────────

    with tabs[5]:
        st.subheader("Replicator-Mutator Dynamics")

        col1, col2 = st.columns([3, 1])

        with col2:
            mu = st.slider("Mutation rate (mu)", 0.0, 0.5, 0.01,
                           step=0.005, key="mut_rate")
            t_end_mut = st.slider("Integration time", 10.0, 500.0, 100.0,
                                  step=10.0, key="mut_tend")

            x0_mut_str = st.text_input(
                "Initial condition",
                value=",".join([f"{1/n:.3f}"] * n),
                key="mut_x0",
            )
            show_bifurcation = st.checkbox("Show bifurcation diagram", value=False)

        with col1:
            try:
                x0 = np.array([float(v.strip()) for v in x0_mut_str.split(",")])
                if len(x0) == n and np.isclose(x0.sum(), 1.0):
                    rm = ReplicatorMutator(payoff_matrix, mutation_rate=mu)
                    result = rm.solve(x0, t_span=(0.0, t_end_mut))

                    fig, ax = plt.subplots(figsize=(10, 5))
                    colors = plt.cm.Set2(np.linspace(0, 1, n))
                    for i in range(n):
                        label = strategy_labels[i] if i < len(strategy_labels) else f"S{i}"
                        ax.plot(result.t, result.x[i], label=label,
                                color=colors[i], linewidth=2)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Population Fraction")
                    ax.set_title(f"Replicator-Mutator (mu={mu:.3f})")
                    ax.set_ylim(-0.05, 1.05)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(f"**Equilibrium:** {np.array2string(result.x[:, -1], precision=4)}")

                    if show_bifurcation:
                        st.markdown("---")
                        fig = plot_bifurcation(
                            payoff_matrix,
                            strategy_labels=strategy_labels,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.error("Invalid initial condition.")
            except ValueError:
                st.error("Invalid input format.")

    # ── Footer ───────────────────────────────────────────────────────

    st.markdown("---")
    st.caption(
        "Built with Streamlit. Implements replicator equations, "
        "ESS analysis, Moran process, and Jacobian stability classification."
    )


if __name__ == "__main__":
    main()
