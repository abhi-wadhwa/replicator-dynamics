"""
Replicator-mutator equation: replicator dynamics with mutation.

The replicator-mutator equation combines selection and mutation:
    dx_i/dt = sum_j x_j * f_j(x) * Q_{ji} - x_i * f_bar(x)

where Q is the mutation matrix: Q_{ji} is the probability that an
offspring of type j becomes type i (Q_{ji} = P(j -> i)).

Special cases:
  - Q = I: standard replicator equation (no mutation)
  - Q uniform: pure mutation, no selection memory
  - Small uniform mutation: Q = (1-mu)*I + mu/(n-1)*(J-I)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass
class ReplicatorMutatorResult:
    """Result of replicator-mutator integration."""

    t: NDArray[np.float64]
    x: NDArray[np.float64]  # shape (n_strategies, n_timepoints)
    payoff_matrix: NDArray[np.float64]
    mutation_matrix: NDArray[np.float64]
    x0: NDArray[np.float64]
    success: bool


class ReplicatorMutator:
    """
    Solve the replicator-mutator equation.

    dx_i/dt = sum_j x_j f_j(x) Q_{ji} - x_i f_bar(x)
    """

    def __init__(
        self,
        payoff_matrix: NDArray[np.float64],
        mutation_matrix: NDArray[np.float64] | None = None,
        mutation_rate: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        payoff_matrix : array
            n x n payoff matrix A.
        mutation_matrix : array or None
            n x n mutation matrix Q where Q[j, i] = P(type j mutates to type i).
            Each row must sum to 1. If None, constructed from mutation_rate.
        mutation_rate : float
            If mutation_matrix is None, construct uniform mutation:
            Q = (1 - mu) * I + mu / (n-1) * (J - I)
            where mu = mutation_rate.
        """
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        self.n = self.A.shape[0]

        if mutation_matrix is not None:
            self.Q = np.asarray(mutation_matrix, dtype=np.float64)
            if self.Q.shape != (self.n, self.n):
                raise ValueError(
                    f"Mutation matrix shape {self.Q.shape} doesn't match "
                    f"payoff matrix size {self.n}"
                )
        else:
            self.Q = self._uniform_mutation_matrix(mutation_rate)

        self.mu = mutation_rate

    def _uniform_mutation_matrix(self, mu: float) -> NDArray[np.float64]:
        """
        Construct uniform mutation matrix.

        Q[j, i] = (1 - mu) if i == j, else mu / (n - 1).
        """
        if self.n == 1:
            return np.ones((1, 1))
        Q = np.full((self.n, self.n), mu / (self.n - 1))
        np.fill_diagonal(Q, 1.0 - mu)
        return Q

    def rhs(self, t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Right-hand side of the replicator-mutator equation.

        dx_i/dt = sum_j x_j * f_j(x) * Q_{ji} - x_i * f_bar(x)
        """
        x = np.clip(x, 0.0, 1.0)
        s = x.sum()
        if s > 0:
            x = x / s

        # Fitness vector
        f = self.A @ x  # f_j = sum_k A[j,k] x_k

        # Mean fitness
        f_bar = float(x @ f)

        # Mutation-selection term: sum_j x_j f_j Q_{ji} for each i
        # = (Q^T @ (x * f))_i
        selection_mutation = self.Q.T @ (x * f)

        # dx_i/dt = selection_mutation_i - x_i * f_bar
        return selection_mutation - x * f_bar

    def solve(
        self,
        x0: NDArray[np.float64],
        t_span: tuple[float, float] = (0.0, 50.0),
        n_points: int = 1000,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> ReplicatorMutatorResult:
        """
        Integrate the replicator-mutator equation.

        Parameters
        ----------
        x0 : array
            Initial population state (must sum to 1).
        t_span : tuple
            Integration time interval.
        n_points : int
            Number of output time points.
        method : str
            ODE solver method.

        Returns
        -------
        ReplicatorMutatorResult with time array and trajectory.
        """
        x0 = np.asarray(x0, dtype=np.float64)
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
        )

        # Clip and renormalize
        x = np.clip(sol.y, 0.0, 1.0)
        for i in range(x.shape[1]):
            s = x[:, i].sum()
            if s > 0:
                x[:, i] /= s

        return ReplicatorMutatorResult(
            t=sol.t,
            x=x,
            payoff_matrix=self.A,
            mutation_matrix=self.Q,
            x0=x0,
            success=sol.success,
        )

    def equilibrium_with_mutation(self) -> NDArray[np.float64] | None:
        """
        Find the equilibrium of the replicator-mutator equation.

        For small mutation rates, this is close to the ESS of the
        underlying game. We find it by long-time integration.
        """
        # Start from uniform distribution
        x0 = np.ones(self.n) / self.n
        result = self.solve(x0, t_span=(0.0, 500.0), n_points=5000)

        if result.success:
            return result.x[:, -1]
        return None

    def bifurcation_data(
        self,
        mu_values: NDArray[np.float64] | None = None,
        x0: NDArray[np.float64] | None = None,
        t_end: float = 200.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute equilibrium state as a function of mutation rate.

        Returns (mu_values, equilibria) where equilibria[i, :] is the
        equilibrium state at mu_values[i].
        """
        if mu_values is None:
            mu_values = np.linspace(0.0, 0.5, 50)
        if x0 is None:
            x0 = np.ones(self.n) / self.n

        equilibria = np.zeros((len(mu_values), self.n))

        for i, mu in enumerate(mu_values):
            self.Q = self._uniform_mutation_matrix(mu)
            result = self.solve(x0, t_span=(0.0, t_end), n_points=2000)
            equilibria[i] = result.x[:, -1]

        # Restore original mutation matrix
        self.Q = self._uniform_mutation_matrix(self.mu)

        return mu_values, equilibria
