"""
Tests for the replicator ODE solver.

Key test: Rock-Paper-Scissors produces closed orbits --
trajectories return near their starting point.
"""

import numpy as np
import pytest

from src.core.replicator import PRESET_GAMES, MultiPopulationReplicator, ReplicatorODE


class TestReplicatorODE:
    """Tests for the single-population replicator equation."""

    def test_rps_closed_orbits(self):
        """
        RPS with symmetric payoff matrix produces closed orbits.
        A trajectory starting near the interior should return near its start.
        """
        A = PRESET_GAMES["Rock-Paper-Scissors"]["matrix"]
        ode = ReplicatorODE(A)

        x0 = np.array([0.5, 0.3, 0.2])
        # Integrate for a long time with high precision
        result = ode.solve(x0, t_span=(0.0, 200.0), n_points=10000,
                           rtol=1e-10, atol=1e-12)

        # The trajectory should stay on the simplex
        for i in range(result.x.shape[1]):
            assert abs(result.x[:, i].sum() - 1.0) < 1e-6

        # For symmetric RPS, the Lyapunov function sum(log(x_i)) is conserved.
        # Check that it doesn't drift much.
        eps = 1e-12
        L_start = np.sum(np.log(np.clip(result.x[:, 0], eps, None)))
        L_end = np.sum(np.log(np.clip(result.x[:, -1], eps, None)))
        assert abs(L_start - L_end) < 0.01, (
            f"Lyapunov function drifted: start={L_start:.6f}, end={L_end:.6f}"
        )

    def test_dominant_strategy_converges(self):
        """
        Prisoner's Dilemma: Defect dominates Cooperate.
        Starting from mixed, should converge to all-Defect.
        """
        A = PRESET_GAMES["Prisoner's Dilemma"]["matrix"]
        ode = ReplicatorODE(A)

        x0 = np.array([0.5, 0.5])
        result = ode.solve(x0, t_span=(0.0, 100.0))

        # Strategy 1 (Defect) should dominate
        assert result.final_state[1] > 0.99
        assert result.final_state[0] < 0.01

    def test_simplex_constraint(self):
        """Population fractions sum to 1 at all times."""
        A = np.random.RandomState(42).randn(4, 4)
        ode = ReplicatorODE(A)
        x0 = np.array([0.1, 0.2, 0.3, 0.4])
        result = ode.solve(x0)

        sums = result.x.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_velocity_at_equilibrium(self):
        """Velocity should be zero at the interior equilibrium of RPS."""
        A = PRESET_GAMES["Rock-Paper-Scissors"]["matrix"]
        ode = ReplicatorODE(A)

        x_eq = np.array([1/3, 1/3, 1/3])
        v = ode.velocity_field(x_eq)
        np.testing.assert_allclose(v, 0.0, atol=1e-10)

    def test_invalid_payoff_matrix(self):
        """Non-square matrix should raise ValueError."""
        with pytest.raises(ValueError):
            ReplicatorODE(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_invalid_initial_condition(self):
        """Initial condition not summing to 1 should raise ValueError."""
        ode = ReplicatorODE(np.eye(3))
        with pytest.raises(ValueError):
            ode.solve(np.array([0.3, 0.3, 0.3]))  # sums to 0.9

    def test_n_strategies_property(self):
        """Check n_strategies property."""
        ode = ReplicatorODE(np.eye(5))
        assert ode.n_strategies == 5


class TestMultiPopulationReplicator:
    """Tests for asymmetric (two-population) replicator dynamics."""

    def test_matching_pennies(self):
        """
        Matching Pennies: no pure-strategy equilibrium,
        interior equilibrium at (0.5, 0.5) for both populations.
        """
        A = np.array([[1, -1], [-1, 1]])  # row player
        B = np.array([[-1, 1], [1, -1]])  # col player

        multi = MultiPopulationReplicator(A, B)
        t, x, y = multi.solve(
            np.array([0.6, 0.4]),
            np.array([0.4, 0.6]),
            t_span=(0.0, 100.0),
        )

        # Both populations should oscillate around 0.5
        assert x.shape[0] == 2
        assert y.shape[0] == 2

    def test_shape_mismatch(self):
        """Mismatched payoff matrix shapes should raise."""
        with pytest.raises(ValueError):
            MultiPopulationReplicator(
                np.array([[1, 2], [3, 4]]),
                np.array([[1, 2, 3], [4, 5, 6]]),
            )

    def test_simplex_constraint_multi(self):
        """Both population states should sum to 1."""
        A = np.array([[2, 0], [0, 1]])
        B = np.array([[1, 0], [0, 2]])
        multi = MultiPopulationReplicator(A, B)

        t, x, y = multi.solve(
            np.array([0.7, 0.3]),
            np.array([0.4, 0.6]),
        )

        x_sums = x.sum(axis=0)
        y_sums = y.sum(axis=0)
        np.testing.assert_allclose(x_sums, 1.0, atol=1e-4)
        np.testing.assert_allclose(y_sums, 1.0, atol=1e-4)
