"""
Moran process: finite-population stochastic evolutionary dynamics.

In each time step:
  1. Select an individual for reproduction with probability proportional
     to fitness.
  2. The offspring replaces a uniformly random individual (possibly itself).

Key quantities:
  - Fixation probability: chance that a single mutant takes over.
  - Neutral drift: fixation probability = 1/N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MoranResult:
    """Result of Moran process simulation."""

    fixation_probability: float
    n_simulations: int
    n_fixations: int
    population_size: int
    mean_fixation_time: float
    trajectories: list[list[int]] | None = None


class MoranProcess:
    """
    Simulate the Moran process for 2-strategy games.

    Population of N individuals, each playing strategy A or B.
    Fitness is determined by payoff matrix interactions.
    """

    def __init__(
        self,
        payoff_matrix: NDArray[np.float64],
        population_size: int = 100,
        intensity: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        payoff_matrix : 2x2 array
            Payoff matrix [[a, b], [c, d]] where a = A vs A, b = A vs B, etc.
        population_size : int
            Number of individuals N.
        intensity : float
            Selection intensity w. Fitness = 1 - w + w * payoff.
            w=0 is neutral drift, w=1 is strong selection.
        """
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        if self.A.shape != (2, 2):
            raise ValueError(f"Moran process requires 2x2 payoff matrix, got {self.A.shape}")
        self.N = population_size
        self.w = intensity

    def _fitness(self, n_A: int) -> tuple[float, float]:
        """
        Compute fitness of type A and type B given n_A individuals of type A.

        Each A interacts with (n_A - 1) other A's and (N - n_A) B's.
        Each B interacts with n_A A's and (N - n_A - 1) other B's.
        """
        n_B = self.N - n_A
        a, b, c, d = self.A[0, 0], self.A[0, 1], self.A[1, 0], self.A[1, 1]

        if n_A == 0:
            f_A = 0.0
        else:
            # Average payoff for type A
            if self.N > 1:
                f_A = (a * (n_A - 1) + b * n_B) / (self.N - 1)
            else:
                f_A = 0.0

        if n_B == 0:
            f_B = 0.0
        else:
            if self.N > 1:
                f_B = (c * n_A + d * (n_B - 1)) / (self.N - 1)
            else:
                f_B = 0.0

        # Apply selection intensity
        fitness_A = 1.0 - self.w + self.w * f_A
        fitness_B = 1.0 - self.w + self.w * f_B

        return max(fitness_A, 1e-12), max(fitness_B, 1e-12)

    def simulate_once(
        self,
        initial_A: int = 1,
        max_steps: int = 1_000_000,
        record_trajectory: bool = False,
    ) -> tuple[bool, int, list[int] | None]:
        """
        Run one Moran process simulation.

        Parameters
        ----------
        initial_A : int
            Initial number of type-A individuals.
        max_steps : int
            Maximum number of steps before declaring timeout.
        record_trajectory : bool
            If True, record n_A at each step.

        Returns
        -------
        (fixated, steps, trajectory)
            fixated: True if A fixated (n_A = N), False if extinct (n_A = 0).
            steps: number of steps taken.
            trajectory: list of n_A values if recorded.
        """
        rng = np.random.default_rng()
        n_A = initial_A
        trajectory = [n_A] if record_trajectory else None

        for step in range(max_steps):
            if n_A == 0:
                return False, step, trajectory
            if n_A == self.N:
                return True, step, trajectory

            f_A, f_B = self._fitness(n_A)
            n_B = self.N - n_A

            # Total fitness
            total_fitness = f_A * n_A + f_B * n_B

            # Probability that A is chosen for reproduction
            p_reprod_A = (f_A * n_A) / total_fitness

            # Probability that a B dies (replaced by A offspring)
            # = P(A reproduces) * P(B is replaced) = p_reprod_A * (n_B / N)
            p_A_increase = p_reprod_A * (n_B / self.N)

            # Probability that A dies (replaced by B offspring)
            p_A_decrease = (1 - p_reprod_A) * (n_A / self.N)

            r = rng.random()
            if r < p_A_increase:
                n_A += 1
            elif r < p_A_increase + p_A_decrease:
                n_A -= 1
            # else: no change (same type reproduces and replaces same type)

            if record_trajectory and trajectory is not None:
                trajectory.append(n_A)

        # Timeout: treat as whatever the majority is
        return n_A > self.N // 2, max_steps, trajectory

    def fixation_probability(
        self,
        initial_A: int = 1,
        n_simulations: int = 10000,
        max_steps: int = 1_000_000,
    ) -> MoranResult:
        """
        Estimate fixation probability via Monte Carlo simulation.

        Parameters
        ----------
        initial_A : int
            Initial number of type-A individuals.
        n_simulations : int
            Number of independent runs.
        max_steps : int
            Max steps per run.

        Returns
        -------
        MoranResult with fixation probability estimate.
        """
        n_fixations = 0
        total_time = 0

        for _ in range(n_simulations):
            fixated, steps, _ = self.simulate_once(
                initial_A=initial_A,
                max_steps=max_steps,
                record_trajectory=False,
            )
            if fixated:
                n_fixations += 1
                total_time += steps

        fix_prob = n_fixations / n_simulations
        mean_time = total_time / max(n_fixations, 1)

        return MoranResult(
            fixation_probability=fix_prob,
            n_simulations=n_simulations,
            n_fixations=n_fixations,
            population_size=self.N,
            mean_fixation_time=mean_time,
        )

    def fixation_probability_exact(self, initial_A: int = 1) -> float:
        """
        Compute exact fixation probability for 2-strategy Moran process.

        For constant fitness ratio r = f_A / f_B:
            rho = (1 - 1/r) / (1 - 1/r^N)

        For frequency-dependent selection, we use the recursive formula:
            rho_k = sum_{j=1}^{k-1} prod_{i=1}^{j} (gamma_i)
                    / (1 + sum_{j=1}^{N-1} prod_{i=1}^{j} gamma_i)

        where gamma_i = P(i -> i-1) / P(i -> i+1).
        """
        # Compute gamma_i for each state
        gammas = np.zeros(self.N)
        for i in range(1, self.N):
            f_A, f_B = self._fitness(i)
            n_B = self.N - i

            p_up = (f_A * i / (f_A * i + f_B * n_B)) * (n_B / self.N)
            p_down = (f_B * n_B / (f_A * i + f_B * n_B)) * (i / self.N)

            if p_up < 1e-15:
                gammas[i] = 1e15
            else:
                gammas[i] = p_down / p_up

        # Compute products
        products = np.zeros(self.N)
        products[1] = gammas[1]
        for j in range(2, self.N):
            products[j] = products[j - 1] * gammas[j]

        denominator = 1.0 + np.sum(products[1:])

        # rho_k = (1 + sum_{j=1}^{k-1} products[j]) / denominator
        numerator = 1.0 + np.sum(products[1:initial_A])
        return numerator / denominator

    def simulate_with_trajectory(
        self, initial_A: int = 1, max_steps: int = 100_000
    ) -> tuple[bool, list[int]]:
        """Run single simulation and return full trajectory."""
        fixated, _, trajectory = self.simulate_once(
            initial_A=initial_A,
            max_steps=max_steps,
            record_trajectory=True,
        )
        return fixated, trajectory or []
