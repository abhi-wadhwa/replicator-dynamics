"""
Replicator ODE solver for evolutionary game dynamics.

The replicator equation describes how strategy frequencies evolve:
    dx_i/dt = x_i * [f_i(x) - f_bar(x)]

where f_i(x) is the fitness of strategy i against population state x,
and f_bar(x) is the mean fitness of the population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass
class ReplicatorResult:
    """Container for ODE integration results."""

    t: NDArray[np.float64]
    x: NDArray[np.float64]  # shape (n_strategies, n_timepoints)
    payoff_matrix: NDArray[np.float64]
    x0: NDArray[np.float64]
    success: bool
    message: str

    @property
    def n_strategies(self) -> int:
        return self.x.shape[0]

    @property
    def final_state(self) -> NDArray[np.float64]:
        return self.x[:, -1]


class ReplicatorODE:
    """
    Solve the replicator equation for symmetric games.

    dx_i/dt = x_i * [(Ax)_i - x^T A x]

    where A is the payoff matrix and x is the population state vector.
    """

    def __init__(self, payoff_matrix: NDArray[np.float64]) -> None:
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError(f"Payoff matrix must be square, got shape {self.A.shape}")
        self.n_strategies = n

    def fitness(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute fitness vector f(x) = Ax."""
        return self.A @ x

    def mean_fitness(self, x: NDArray[np.float64]) -> float:
        """Compute mean fitness f_bar = x^T A x."""
        return float(x @ self.A @ x)

    def rhs(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Right-hand side of the replicator ODE.

        dx_i/dt = x_i * [f_i(x) - f_bar(x)]
        """
        # Clip to avoid numerical issues at boundaries
        x = np.clip(x, 0.0, 1.0)
        x = x / x.sum()  # Renormalize

        f = self.fitness(x)
        f_bar = float(x @ f)
        return x * (f - f_bar)

    def solve(
        self,
        x0: NDArray[np.float64],
        t_span: tuple[float, float] = (0.0, 50.0),
        n_points: int = 1000,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> ReplicatorResult:
        """
        Integrate the replicator ODE from initial condition x0.

        Parameters
        ----------
        x0 : array-like
            Initial population fractions (must sum to 1).
        t_span : tuple
            (t_start, t_end) integration interval.
        n_points : int
            Number of output time points.
        method : str
            Integration method for solve_ivp (default RK45).
        rtol, atol : float
            Relative and absolute tolerances.

        Returns
        -------
        ReplicatorResult with time array and trajectory.
        """
        x0 = np.asarray(x0, dtype=np.float64)
        if len(x0) != self.n_strategies:
            raise ValueError(
                f"Initial condition has {len(x0)} entries, "
                f"expected {self.n_strategies}"
            )
        if not np.isclose(x0.sum(), 1.0):
            raise ValueError(f"Initial condition must sum to 1, got {x0.sum():.6f}")

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            self.rhs,
            t_span,
            x0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        # Clip and renormalize solution
        x = np.clip(sol.y, 0.0, 1.0)
        for i in range(x.shape[1]):
            s = x[:, i].sum()
            if s > 0:
                x[:, i] /= s

        return ReplicatorResult(
            t=sol.t,
            x=x,
            payoff_matrix=self.A,
            x0=x0,
            success=sol.success,
            message=sol.message,
        )

    def velocity_field(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute dx/dt at a given population state."""
        return self.rhs(0.0, x)


class MultiPopulationReplicator:
    """
    Multi-population (asymmetric) replicator dynamics.

    For a two-population game with payoff matrices A (row player)
    and B (column player):
        dx_i/dt = x_i * [(Ay)_i - x^T A y]
        dy_j/dt = y_j * [(B^T x)_j - y^T B^T x]

    where x is population 1 state and y is population 2 state.
    """

    def __init__(
        self,
        payoff_A: NDArray[np.float64],
        payoff_B: NDArray[np.float64],
    ) -> None:
        self.A = np.asarray(payoff_A, dtype=np.float64)
        self.B = np.asarray(payoff_B, dtype=np.float64)
        if self.A.shape != self.B.shape:
            raise ValueError(
                f"Payoff matrices must have same shape: "
                f"A={self.A.shape}, B={self.B.shape}"
            )
        self.m, self.n = self.A.shape  # m strategies for pop1, n for pop2

    def rhs(self, t: float, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """RHS for concatenated state z = [x; y]."""
        x = np.clip(z[: self.m], 0.0, 1.0)
        y = np.clip(z[self.m :], 0.0, 1.0)

        sx, sy = x.sum(), y.sum()
        if sx > 0:
            x = x / sx
        if sy > 0:
            y = y / sy

        # Population 1: row player
        fx = self.A @ y
        fx_bar = float(x @ fx)
        dx = x * (fx - fx_bar)

        # Population 2: column player
        fy = self.B.T @ x
        fy_bar = float(y @ fy)
        dy = y * (fy - fy_bar)

        return np.concatenate([dx, dy])

    def solve(
        self,
        x0: NDArray[np.float64],
        y0: NDArray[np.float64],
        t_span: tuple[float, float] = (0.0, 50.0),
        n_points: int = 1000,
        method: str = "RK45",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Integrate two-population replicator dynamics.

        Returns (t, x_trajectory, y_trajectory).
        """
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)
        z0 = np.concatenate([x0, y0])

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.rhs, t_span, z0, method=method, t_eval=t_eval)

        x_traj = sol.y[: self.m, :]
        y_traj = sol.y[self.m :, :]

        # Normalize
        for i in range(x_traj.shape[1]):
            sx = x_traj[:, i].sum()
            sy = y_traj[:, i].sum()
            if sx > 0:
                x_traj[:, i] /= sx
            if sy > 0:
                y_traj[:, i] /= sy

        return sol.t, x_traj, y_traj


# ── Preset payoff matrices ──────────────────────────────────────────

PRESET_GAMES: dict[str, dict] = {
    "Rock-Paper-Scissors": {
        "matrix": np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64),
        "strategies": ["Rock", "Paper", "Scissors"],
        "description": "Cyclic dominance with zero-sum payoffs. Interior fixed point is a center.",
    },
    "Hawk-Dove": {
        "matrix": np.array([
            [-1, 2],
            [0, 1],
        ], dtype=np.float64),  # V=2, C=3 => (V-C)/2=-0.5 rounded, V/2=1
        "strategies": ["Hawk", "Dove"],
        "description": "Contest over resource V=2, cost C=3. ESS is mixed.",
    },
    "Coordination": {
        "matrix": np.array([
            [2, 0],
            [0, 1],
        ], dtype=np.float64),
        "strategies": ["A", "B"],
        "description": "Two pure ESS: all-A and all-B, with an unstable interior equilibrium.",
    },
    "Snowdrift": {
        "matrix": np.array([
            [1, 3],
            [0, 2],
        ], dtype=np.float64),
        "strategies": ["Cooperate", "Defect"],
        "description": "Also known as Chicken or Hawk-Dove variant. Coexistence equilibrium.",
    },
    "Prisoner's Dilemma": {
        "matrix": np.array([
            [3, 0],
            [5, 1],
        ], dtype=np.float64),
        "strategies": ["Cooperate", "Defect"],
        "description": "Defect dominates. Single ESS at all-Defect.",
    },
    "RPS-Asymmetric": {
        "matrix": np.array([
            [0, -1, 2],
            [1, 0, -1],
            [-2, 1, 0],
        ], dtype=np.float64),
        "strategies": ["Rock", "Paper", "Scissors"],
        "description": "Asymmetric RPS: trajectories spiral outward from interior.",
    },
}
