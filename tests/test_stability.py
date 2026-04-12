"""
Tests for Jacobian analysis and fixed point classification.

Key test: Eigenvalues correctly classify RPS interior as center,
coordination game vertices as stable nodes, etc.
"""

import numpy as np
import pytest

from src.core.jacobian import JacobianAnalyzer, StabilityType


class TestJacobianAnalyzer:
    """Tests for Jacobian-based stability analysis."""

    def test_rps_interior_is_center(self):
        """
        RPS interior equilibrium (1/3, 1/3, 1/3) should be a center:
        purely imaginary eigenvalues.
        """
        A = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ])
        analyzer = JacobianAnalyzer(A)
        interior = np.array([1/3, 1/3, 1/3])

        result = analyzer.analyze_fixed_point(interior, label="RPS interior")

        # Eigenvalues should be purely imaginary
        for ev in result.eigenvalues:
            assert abs(ev.real) < 1e-4, (
                f"Expected purely imaginary eigenvalue, got {ev}"
            )

        assert result.stability == StabilityType.CENTER

    def test_coordination_vertex_stable(self):
        """
        Coordination game: vertices (1,0) and (0,1) should be
        stable nodes (negative eigenvalue).
        """
        A = np.array([[2, 0], [0, 1]])
        analyzer = JacobianAnalyzer(A)

        # Vertex e_0 = (1, 0): dominant strategy A
        e0 = np.array([1.0, 0.0])
        result = analyzer.analyze_fixed_point(e0, label="All-A")

        # Should have negative eigenvalue
        assert all(ev.real < 0 for ev in result.eigenvalues), (
            f"Expected stable, got eigenvalues {result.eigenvalues}"
        )
        assert result.stability in (StabilityType.STABLE_NODE, StabilityType.STABLE_FOCUS)

    def test_coordination_interior_saddle(self):
        """
        Coordination game interior equilibrium should be a saddle
        (or unstable).
        """
        A = np.array([[2, 0], [0, 1]])
        analyzer = JacobianAnalyzer(A)

        # Interior equilibrium: p* = (A[1,1] - A[0,1]) / (A[0,0] - A[0,1] - A[1,0] + A[1,1])
        # = 1 / 3
        interior = np.array([1/3, 2/3])
        result = analyzer.analyze_fixed_point(interior)

        # Should be unstable (positive eigenvalue)
        assert any(ev.real > 0 for ev in result.eigenvalues)

    def test_find_fixed_points_rps(self):
        """RPS should have 3 vertices + 1 interior = 4 fixed points."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        analyzer = JacobianAnalyzer(A)

        fps = analyzer.find_fixed_points()

        # 3 vertices + possible edge equilibria + 1 interior
        # Vertices are always fixed points
        assert len(fps) >= 4  # At least 3 vertices + 1 interior

        # Check interior is found
        has_interior = False
        for fp in fps:
            if np.all(fp > 0.01):
                has_interior = True
                np.testing.assert_allclose(fp, [1/3, 1/3, 1/3], atol=1e-6)
        assert has_interior

    def test_find_edge_equilibrium(self):
        """Coordination game should have an edge equilibrium."""
        A = np.array([[2, 0], [0, 1]])
        analyzer = JacobianAnalyzer(A)

        eq = analyzer._find_edge_equilibrium(0, 1)
        assert eq is not None
        # Interior of 2-strategy game: p* = (1-0)/(2-0-0+1) = 1/3
        assert abs(eq[0] - 1/3) < 1e-6

    def test_classify_eigenvalues(self):
        """Test eigenvalue classification for known cases."""
        analyzer = JacobianAnalyzer(np.eye(2))

        # Stable node: all real negative
        assert analyzer.classify_eigenvalues(
            np.array([-1.0 + 0j, -2.0 + 0j])
        ) == StabilityType.STABLE_NODE

        # Unstable node: all real positive
        assert analyzer.classify_eigenvalues(
            np.array([1.0 + 0j, 2.0 + 0j])
        ) == StabilityType.UNSTABLE_NODE

        # Saddle: mixed signs
        assert analyzer.classify_eigenvalues(
            np.array([1.0 + 0j, -2.0 + 0j])
        ) == StabilityType.SADDLE

        # Stable focus: complex with negative real part
        assert analyzer.classify_eigenvalues(
            np.array([-0.5 + 1j, -0.5 - 1j])
        ) == StabilityType.STABLE_FOCUS

        # Unstable focus: complex with positive real part
        assert analyzer.classify_eigenvalues(
            np.array([0.5 + 1j, 0.5 - 1j])
        ) == StabilityType.UNSTABLE_FOCUS

        # Center: purely imaginary
        assert analyzer.classify_eigenvalues(
            np.array([0.0 + 1j, 0.0 - 1j])
        ) == StabilityType.CENTER

    def test_full_analysis_returns_results(self):
        """full_analysis should return FixedPointAnalysis for each FP."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        analyzer = JacobianAnalyzer(A)

        results = analyzer.full_analysis()
        assert len(results) >= 4  # 3 vertices + 1 interior

        for r in results:
            assert r.point is not None
            assert r.jacobian is not None
            assert r.eigenvalues is not None
            assert r.stability is not None
            assert r.label

    def test_pd_defect_vertex_stable(self):
        """
        Prisoner's Dilemma: Defect vertex should be stable,
        Cooperate vertex should be unstable.
        """
        A = np.array([[3, 0], [5, 1]])
        analyzer = JacobianAnalyzer(A)

        # Defect is strategy 1
        e1 = np.array([0.0, 1.0])
        r1 = analyzer.analyze_fixed_point(e1, "Defect")
        assert all(ev.real < 0 for ev in r1.eigenvalues)

        # Cooperate is strategy 0
        e0 = np.array([1.0, 0.0])
        r0 = analyzer.analyze_fixed_point(e0, "Cooperate")
        assert any(ev.real > 0 for ev in r0.eigenvalues)
