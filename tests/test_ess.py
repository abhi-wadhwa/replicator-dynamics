"""
Tests for ESS analysis.

Key test: Hawk-Dove ESS matches p* = V/C analytically.
"""

import numpy as np
import pytest

from src.core.ess import ESSChecker


class TestESSChecker:
    """Tests for evolutionarily stable strategy analysis."""

    def test_hawk_dove_ess_mixed(self):
        """
        Hawk-Dove with V < C: ESS is mixed with p*(Hawk) = V/C.

        V=2, C=5 => p* = 2/5 = 0.4
        """
        V, C = 2.0, 5.0
        checker = ESSChecker(np.zeros((2, 2)))  # placeholder
        result = checker.hawk_dove_ess(V, C)

        assert result["type"] == "mixed"
        assert abs(result["p_hawk"] - V / C) < 1e-10

    def test_hawk_dove_ess_pure(self):
        """
        Hawk-Dove with V >= C: ESS is pure Hawk.
        """
        V, C = 5.0, 3.0
        checker = ESSChecker(np.zeros((2, 2)))
        result = checker.hawk_dove_ess(V, C)

        assert result["type"] == "pure"
        assert result["p_hawk"] == 1.0

    def test_hawk_dove_ess_parametric(self):
        """Test several V/C ratios."""
        test_cases = [
            (1.0, 2.0, 0.5),
            (1.0, 4.0, 0.25),
            (3.0, 10.0, 0.3),
            (1.0, 1.0, 1.0),  # V = C => pure Hawk
        ]
        for V, C, expected_p in test_cases:
            checker = ESSChecker(np.zeros((2, 2)))
            result = checker.hawk_dove_ess(V, C)
            assert abs(result["p_hawk"] - expected_p) < 1e-10, (
                f"V={V}, C={C}: expected p*={expected_p}, got {result['p_hawk']}"
            )

    def test_prisoners_dilemma_pure_ess(self):
        """
        PD: Defect is a strict Nash equilibrium => ESS.
        Cooperate is not Nash => not ESS.
        """
        A = np.array([
            [3, 0],
            [5, 1],
        ])
        checker = ESSChecker(A)

        # Cooperate (strategy 0) is NOT Nash
        r0 = checker.check_pure(0)
        assert not r0.is_nash
        assert not r0.is_ess

        # Defect (strategy 1) IS ESS (strict Nash)
        r1 = checker.check_pure(1)
        assert r1.is_nash
        assert r1.is_ess

    def test_coordination_game_two_ess(self):
        """
        Coordination: both pure strategies are ESS.
        A = [[2, 0], [0, 1]]
        """
        A = np.array([[2, 0], [0, 1]])
        checker = ESSChecker(A)

        r0 = checker.check_pure(0)
        assert r0.is_ess, "Strategy 0 should be ESS in coordination game"

        r1 = checker.check_pure(1)
        assert r1.is_ess, "Strategy 1 should be ESS in coordination game"

    def test_rps_no_pure_ess(self):
        """RPS: no pure strategy is Nash or ESS."""
        A = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ])
        checker = ESSChecker(A)

        for i in range(3):
            r = checker.check_pure(i)
            assert not r.is_nash
            assert not r.is_ess

    def test_rps_interior_not_ess(self):
        """
        RPS interior equilibrium (1/3, 1/3, 1/3) is Nash but NOT ESS.
        The tangent-space condition fails (eigenvalues are zero).
        """
        A = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ])
        checker = ESSChecker(A)

        interior = checker.find_interior_equilibrium()
        assert interior is not None
        np.testing.assert_allclose(interior, [1/3, 1/3, 1/3], atol=1e-8)

        result = checker.check_mixed(interior)
        assert result.is_nash
        # RPS interior is Nash but not ESS (center, not asymptotically stable)
        assert not result.is_ess

    def test_find_interior_equilibrium(self):
        """Interior equilibrium computation for various games."""
        # RPS: interior at (1/3, 1/3, 1/3)
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        checker = ESSChecker(A)
        eq = checker.find_interior_equilibrium()
        assert eq is not None
        np.testing.assert_allclose(eq, [1/3, 1/3, 1/3], atol=1e-8)

    def test_analyze_all(self):
        """analyze_all returns results for all pure strategies + interior."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        checker = ESSChecker(A)
        results = checker.analyze_all()

        # 3 pure + 1 interior = 4 results
        assert len(results) == 4

    def test_payoff_function(self):
        """Test the payoff computation u(p, q) = p^T A q."""
        A = np.array([[3, 0], [5, 1]])
        checker = ESSChecker(A)

        p = np.array([1.0, 0.0])  # Cooperate
        q = np.array([0.0, 1.0])  # Defect

        assert checker.payoff(p, q) == 0.0  # C vs D = 0
        assert checker.payoff(q, p) == 5.0  # D vs C = 5
        assert checker.payoff(p, p) == 3.0  # C vs C = 3
        assert checker.payoff(q, q) == 1.0  # D vs D = 1
