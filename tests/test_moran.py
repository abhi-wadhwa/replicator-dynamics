"""
Tests for the Moran process simulation.

Key test: Under neutral drift (w=0), fixation probability = 1/N.
"""

import numpy as np
import pytest

from src.core.moran import MoranProcess


class TestMoranProcess:
    """Tests for the finite-population Moran process."""

    def test_neutral_drift_fixation_probability(self):
        """
        Under neutral drift (w=0), fixation probability of a single
        mutant should be approximately 1/N.

        Use large N and many simulations for a tight estimate.
        """
        N = 20
        # Neutral game: all payoffs equal
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        mp = MoranProcess(A, population_size=N, intensity=0.0)

        result = mp.fixation_probability(
            initial_A=1,
            n_simulations=20000,
            max_steps=500_000,
        )

        expected = 1.0 / N
        # Allow 40% relative tolerance for stochastic test
        assert abs(result.fixation_probability - expected) < 0.02, (
            f"Fixation prob {result.fixation_probability:.4f} too far from "
            f"expected {expected:.4f}"
        )

    def test_exact_vs_monte_carlo(self):
        """
        Exact fixation probability should be close to Monte Carlo estimate.
        """
        A = np.array([[2.0, 1.0], [1.0, 2.0]])  # Coordination
        N = 15
        mp = MoranProcess(A, population_size=N, intensity=1.0)

        exact = mp.fixation_probability_exact(initial_A=1)
        mc = mp.fixation_probability(initial_A=1, n_simulations=15000)

        assert abs(exact - mc.fixation_probability) < 0.03, (
            f"Exact={exact:.4f}, MC={mc.fixation_probability:.4f}"
        )

    def test_dominant_strategy_high_fixation(self):
        """
        When strategy A strictly dominates, fixation probability should
        be well above 1/N.
        """
        # A dominates: payoff of A is always higher
        A = np.array([[3.0, 3.0], [1.0, 1.0]])
        N = 20
        mp = MoranProcess(A, population_size=N, intensity=1.0)

        result = mp.fixation_probability(initial_A=1, n_simulations=5000)
        assert result.fixation_probability > 1.0 / N

    def test_dominated_strategy_low_fixation(self):
        """
        When strategy A is dominated, fixation probability should be
        well below 1/N.
        """
        A = np.array([[1.0, 1.0], [3.0, 3.0]])
        N = 20
        mp = MoranProcess(A, population_size=N, intensity=1.0)

        result = mp.fixation_probability(initial_A=1, n_simulations=5000)
        assert result.fixation_probability < 1.0 / N

    def test_trajectory_recording(self):
        """Trajectory should start at initial_A and end at 0 or N."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        N = 10
        mp = MoranProcess(A, population_size=N, intensity=0.0)

        fixated, trajectory = mp.simulate_with_trajectory(initial_A=3)

        assert trajectory[0] == 3
        assert trajectory[-1] == 0 or trajectory[-1] == N
        if fixated:
            assert trajectory[-1] == N
        else:
            assert trajectory[-1] == 0

    def test_exact_neutral_drift(self):
        """Exact formula should give 1/N for neutral case."""
        N = 50
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        mp = MoranProcess(A, population_size=N, intensity=0.0)

        exact = mp.fixation_probability_exact(initial_A=1)
        assert abs(exact - 1.0 / N) < 1e-6

    def test_exact_initial_k(self):
        """Exact fixation prob for k initial mutants should be k/N under neutrality."""
        N = 30
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        mp = MoranProcess(A, population_size=N, intensity=0.0)

        for k in [1, 5, 10, 15]:
            exact = mp.fixation_probability_exact(initial_A=k)
            expected = k / N
            assert abs(exact - expected) < 1e-4, (
                f"k={k}: exact={exact:.6f}, expected={expected:.6f}"
            )

    def test_invalid_payoff_shape(self):
        """Non-2x2 payoff matrix should raise ValueError."""
        with pytest.raises(ValueError):
            MoranProcess(np.eye(3), population_size=10)

    def test_fitness_computation(self):
        """Verify fitness values for known configuration."""
        A = np.array([[3.0, 0.0], [5.0, 1.0]])  # PD
        mp = MoranProcess(A, population_size=10, intensity=1.0)

        # 5 of type A, 5 of type B
        f_A, f_B = mp._fitness(5)

        # f_A = (3*4 + 0*5) / 9 = 12/9 = 1.333...
        # fitness_A = 1 - 1 + 1 * 1.333 = 1.333
        expected_payoff_A = (3.0 * 4 + 0.0 * 5) / 9
        expected_fitness_A = 1.0 - 1.0 + 1.0 * expected_payoff_A
        assert abs(f_A - expected_fitness_A) < 1e-10
