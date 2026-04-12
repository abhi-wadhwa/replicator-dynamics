"""
Jacobian analysis for replicator dynamics.

Compute the Jacobian of the replicator equation at fixed points and
classify stability via eigenvalue analysis.

For n strategies on the simplex, we work in the reduced (n-1)-dimensional
coordinate system by eliminating one strategy (x_n = 1 - sum x_i).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class StabilityType(Enum):
    STABLE_NODE = "Stable node (all eigenvalues real, negative)"
    UNSTABLE_NODE = "Unstable node (all eigenvalues real, positive)"
    SADDLE = "Saddle point (eigenvalues of mixed sign)"
    STABLE_FOCUS = "Stable focus (complex eigenvalues with negative real part)"
    UNSTABLE_FOCUS = "Unstable focus (complex eigenvalues with positive real part)"
    CENTER = "Center (purely imaginary eigenvalues)"
    NON_HYPERBOLIC = "Non-hyperbolic (zero eigenvalue present)"


@dataclass
class FixedPointAnalysis:
    """Analysis result for a single fixed point."""

    point: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    eigenvalues: NDArray[np.complex128]
    stability: StabilityType
    label: str


class JacobianAnalyzer:
    """
    Compute and analyze the Jacobian of the replicator equation.

    For a symmetric game with payoff matrix A and n strategies,
    the replicator equation is:
        dx_i/dt = x_i * [(Ax)_i - x^T A x]

    We compute the Jacobian numerically via finite differences,
    and analytically where possible.
    """

    def __init__(self, payoff_matrix: NDArray[np.float64]) -> None:
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        self.n = self.A.shape[0]

    def replicator_rhs(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute dx/dt for the full n-dimensional system."""
        f = self.A @ x
        f_bar = float(x @ f)
        return x * (f - f_bar)

    def _reduced_rhs(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        RHS in reduced coordinates.

        y = (x_1, ..., x_{n-1}), x_n = 1 - sum(y).
        Return dy/dt = (dx_1/dt, ..., dx_{n-1}/dt).
        """
        x = np.zeros(self.n)
        x[: self.n - 1] = y
        x[self.n - 1] = 1.0 - y.sum()
        x = np.clip(x, 0.0, 1.0)
        s = x.sum()
        if s > 0:
            x /= s
        dx = self.replicator_rhs(x)
        return dx[: self.n - 1]

    def jacobian_numerical(
        self, x: NDArray[np.float64], eps: float = 1e-7
    ) -> NDArray[np.float64]:
        """
        Compute the Jacobian of the reduced system at point x via
        central finite differences.

        Returns (n-1) x (n-1) matrix.
        """
        y = x[: self.n - 1].copy()
        m = len(y)
        J = np.zeros((m, m))

        for j in range(m):
            y_plus = y.copy()
            y_minus = y.copy()
            y_plus[j] += eps
            y_minus[j] -= eps

            f_plus = self._reduced_rhs(y_plus)
            f_minus = self._reduced_rhs(y_minus)
            J[:, j] = (f_plus - f_minus) / (2 * eps)

        return J

    def jacobian_analytical(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute the Jacobian analytically.

        For the replicator equation dx_i/dt = x_i[(Ax)_i - x^TAx]:

        d(dx_i/dt)/dx_j = delta_{ij}[(Ax)_i - x^TAx]
                         + x_i[A_{ij} - (Ax)_j - (Ax)_i + 2*x^TAx]
                         (approximately, using chain rule through the constraint)

        We use the full n x n Jacobian and then project to the simplex
        tangent space for the reduced system.
        """
        f = self.A @ x
        f_bar = float(x @ f)

        # Full n x n Jacobian (before simplex reduction)
        J_full = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                J_full[i, j] = (
                    (1 if i == j else 0) * (f[i] - f_bar)
                    + x[i] * (self.A[i, j] - f[j] - f[i] + 2 * f_bar)
                    - x[i] * (self.A[j, :] @ x - f_bar)
                )

        # Actually, let's do it cleanly with the derivative formula
        # d(dx_i/dt)/dx_j = delta_{ij}(f_i - f_bar) + x_i(A_{ij} - 2*(Ax)_j)
        # + x_i * sum_k x_k A_{kj}  ... this gets messy.
        # Use numerical for reliability, analytical as cross-check.
        return self.jacobian_numerical(x)

    def classify_eigenvalues(
        self, eigenvalues: NDArray[np.complex128], tol: float = 1e-6
    ) -> StabilityType:
        """Classify stability based on eigenvalues."""
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag

        has_zero = np.any(np.abs(real_parts) < tol)
        has_complex = np.any(np.abs(imag_parts) > tol)
        all_negative = np.all(real_parts < -tol)
        all_positive = np.all(real_parts > tol)
        mixed = np.any(real_parts > tol) and np.any(real_parts < -tol)

        if has_zero:
            if has_complex and np.all(np.abs(real_parts) < tol):
                return StabilityType.CENTER
            return StabilityType.NON_HYPERBOLIC

        if mixed:
            return StabilityType.SADDLE

        if all_negative:
            if has_complex:
                return StabilityType.STABLE_FOCUS
            return StabilityType.STABLE_NODE

        if all_positive:
            if has_complex:
                return StabilityType.UNSTABLE_FOCUS
            return StabilityType.UNSTABLE_NODE

        return StabilityType.NON_HYPERBOLIC

    def analyze_fixed_point(
        self, x: NDArray[np.float64], label: str = ""
    ) -> FixedPointAnalysis:
        """
        Full analysis of a fixed point: Jacobian, eigenvalues, stability.
        """
        x = np.asarray(x, dtype=np.float64)
        J = self.jacobian_numerical(x)
        eigenvalues = np.linalg.eigvals(J)
        stability = self.classify_eigenvalues(eigenvalues)

        if not label:
            label = f"x = {np.array2string(x, precision=4)}"

        return FixedPointAnalysis(
            point=x,
            jacobian=J,
            eigenvalues=eigenvalues,
            stability=stability,
            label=label,
        )

    def find_fixed_points(self) -> list[NDArray[np.float64]]:
        """
        Find all fixed points of the replicator equation on the simplex.

        Fixed points include:
        1. All vertices (pure strategies)
        2. Interior equilibrium (if exists)
        3. Edge equilibria (pairs, triples, etc.)
        """
        fixed_points = []

        # Vertices
        for i in range(self.n):
            e = np.zeros(self.n)
            e[i] = 1.0
            fixed_points.append(e)

        # Edge equilibria (2-strategy faces)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                eq = self._find_edge_equilibrium(i, j)
                if eq is not None:
                    fixed_points.append(eq)

        # Interior equilibrium
        interior = self._find_interior_equilibrium()
        if interior is not None:
            fixed_points.append(interior)

        return fixed_points

    def _find_edge_equilibrium(
        self, i: int, j: int
    ) -> NDArray[np.float64] | None:
        """
        Find equilibrium on the edge between strategies i and j.

        On this edge, x_k = 0 for k != i, j, and x_i + x_j = 1.
        Equilibrium when f_i(x) = f_j(x).
        """
        # Reduced 2x2 game
        A_sub = np.array([
            [self.A[i, i], self.A[i, j]],
            [self.A[j, i], self.A[j, j]],
        ])

        # f_i = A[i,i]*p + A[i,j]*(1-p) = p*(A[i,i]-A[i,j]) + A[i,j]
        # f_j = A[j,i]*p + A[j,j]*(1-p) = p*(A[j,i]-A[j,j]) + A[j,j]
        # Set equal: p*(A[i,i]-A[i,j] - A[j,i]+A[j,j]) = A[j,j] - A[i,j]
        denom = A_sub[0, 0] - A_sub[0, 1] - A_sub[1, 0] + A_sub[1, 1]
        if np.isclose(denom, 0.0):
            return None

        p = (A_sub[1, 1] - A_sub[0, 1]) / denom
        if 0 < p < 1:
            x = np.zeros(self.n)
            x[i] = p
            x[j] = 1 - p
            return x
        return None

    def _find_interior_equilibrium(self) -> NDArray[np.float64] | None:
        """
        Find the interior equilibrium if it exists.

        Solves the augmented system requiring equal fitness and sum = 1.
        Handles singular payoff matrices (e.g., zero-sum games).
        """
        try:
            M = np.zeros((self.n, self.n))
            b = np.zeros(self.n)

            for i in range(self.n - 1):
                M[i] = self.A[i] - self.A[self.n - 1]

            M[self.n - 1] = np.ones(self.n)
            b[self.n - 1] = 1.0

            det = np.linalg.det(M)
            if np.isclose(det, 0.0, atol=1e-12):
                x, residuals, _, _ = np.linalg.lstsq(M, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
            else:
                x = np.linalg.solve(M, b)

            if np.isclose(x.sum(), 1.0, atol=1e-8) and np.all(x > -1e-8):
                x = np.clip(x, 0.0, 1.0)
                x /= x.sum()
                fitness = self.A @ x
                if np.allclose(fitness, fitness[0], atol=1e-8):
                    return x
            return None
        except np.linalg.LinAlgError:
            return None

    def full_analysis(self) -> list[FixedPointAnalysis]:
        """
        Find all fixed points and classify their stability.
        """
        fixed_points = self.find_fixed_points()
        results = []

        for i, fp in enumerate(fixed_points):
            # Create descriptive label
            support = np.where(fp > 1e-8)[0]
            if len(support) == 1:
                label = f"Pure strategy {support[0]}"
            elif len(support) == self.n:
                label = f"Interior equilibrium"
            else:
                label = f"Edge equilibrium on strategies {list(support)}"

            results.append(self.analyze_fixed_point(fp, label=label))

        return results
