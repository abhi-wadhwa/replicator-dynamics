"""
Evolutionarily Stable Strategy (ESS) analysis.

A strategy p* is an ESS if for every mutant strategy q != p*:
  1. (Equilibrium) p* is a Nash Equilibrium: u(p*, p*) >= u(q, p*)
  2. (Stability) If u(p*, p*) = u(q, p*), then u(p*, q) > u(q, q)

For a symmetric game with payoff matrix A:
  - u(p, q) = p^T A q
  - Pure strategy e_i is ESS if A[i,i] > A[j,i] for all j != i,
    or A[i,i] = A[j,i] and A[i,j] > A[j,j] for all j != i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ESSResult:
    """Result of ESS analysis for a single candidate strategy."""

    strategy: NDArray[np.float64]
    is_nash: bool
    is_ess: bool
    label: str
    details: str


class ESSChecker:
    """
    Check whether strategies are Evolutionarily Stable.

    Works for symmetric games with payoff matrix A.
    """

    def __init__(self, payoff_matrix: NDArray[np.float64]) -> None:
        self.A = np.asarray(payoff_matrix, dtype=np.float64)
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError(f"Payoff matrix must be square, got {self.A.shape}")
        self.n = n

    def payoff(
        self, p: NDArray[np.float64], q: NDArray[np.float64]
    ) -> float:
        """Compute u(p, q) = p^T A q."""
        return float(p @ self.A @ q)

    def check_pure(self, i: int) -> ESSResult:
        """
        Check if pure strategy e_i is an ESS.

        Condition 1 (Nash): A[i,i] >= A[j,i] for all j
        Condition 2 (Stability): if A[i,i] = A[j,i], then A[i,j] > A[j,j]
        """
        e_i = np.zeros(self.n)
        e_i[i] = 1.0

        nash_violations = []
        stability_violations = []
        is_nash = True
        is_ess = True

        for j in range(self.n):
            if j == i:
                continue
            if self.A[i, i] < self.A[j, i]:
                is_nash = False
                is_ess = False
                nash_violations.append(
                    f"Strategy {j} gets higher payoff vs e_{i}: "
                    f"A[{j},{i}]={self.A[j, i]:.4f} > A[{i},{i}]={self.A[i, i]:.4f}"
                )
            elif np.isclose(self.A[i, i], self.A[j, i]):
                # Need strict stability condition
                if self.A[i, j] <= self.A[j, j]:
                    is_ess = False
                    stability_violations.append(
                        f"Against mutant {j}: u(e_{i}, e_{j})={self.A[i, j]:.4f} "
                        f"<= u(e_{j}, e_{j})={self.A[j, j]:.4f}"
                    )

        details_parts = []
        if nash_violations:
            details_parts.append("Nash violations: " + "; ".join(nash_violations))
        if stability_violations:
            details_parts.append("Stability violations: " + "; ".join(stability_violations))
        if not details_parts:
            details_parts.append("All ESS conditions satisfied.")

        return ESSResult(
            strategy=e_i,
            is_nash=is_nash,
            is_ess=is_ess,
            label=f"Pure strategy e_{i}",
            details=" | ".join(details_parts),
        )

    def check_mixed(
        self, p: NDArray[np.float64], tol: float = 1e-8
    ) -> ESSResult:
        """
        Check if mixed strategy p is an ESS.

        For a mixed ESS, the support must be indifferent and the
        stability condition must hold for all alternative best replies.

        We check:
        1. All strategies in the support get equal (maximal) payoff against p.
        2. For any q in the best-reply set, u(p, q) > u(q, q).
        """
        p = np.asarray(p, dtype=np.float64)
        if not np.isclose(p.sum(), 1.0):
            raise ValueError(f"Strategy must sum to 1, got {p.sum():.6f}")

        # Fitness of each pure strategy against p
        fitness = self.A @ p
        max_fitness = fitness.max()
        support = np.where(p > tol)[0]

        # Nash check: all strategies in support must be best replies
        is_nash = True
        nash_details = []
        for i in support:
            if not np.isclose(fitness[i], max_fitness, atol=tol):
                is_nash = False
                nash_details.append(
                    f"Strategy {i} in support has fitness {fitness[i]:.6f} "
                    f"< max {max_fitness:.6f}"
                )

        # Also check no strategy outside support does strictly better
        for i in range(self.n):
            if i not in support and fitness[i] > max_fitness + tol:
                is_nash = False
                nash_details.append(
                    f"Strategy {i} outside support has fitness {fitness[i]:.6f} "
                    f"> max {max_fitness:.6f}"
                )

        if not is_nash:
            return ESSResult(
                strategy=p,
                is_nash=False,
                is_ess=False,
                label=f"Mixed strategy {np.array2string(p, precision=4)}",
                details="Not Nash: " + "; ".join(nash_details),
            )

        # Stability check: for every pure best reply q, u(p, q) > u(q, q)
        best_replies = np.where(np.isclose(fitness, max_fitness, atol=tol))[0]
        is_ess = True
        stab_details = []

        for j in best_replies:
            e_j = np.zeros(self.n)
            e_j[j] = 1.0
            u_p_q = self.payoff(p, e_j)
            u_q_q = self.payoff(e_j, e_j)
            if u_p_q <= u_q_q + tol:
                # Need to check the full stability condition via the Hessian
                # For interior ESS, use Bishop-Cannings: check if -A_support is
                # negative definite on the tangent space
                pass

        # For a fully supported mixed strategy, check negative definiteness
        # of A restricted to tangent space of simplex at p
        if len(support) == self.n:
            # Interior equilibrium: ESS iff x^T A x < 0 for all x in
            # tangent space (sum x_i = 0, x != 0)
            is_ess = self._check_interior_ess(p, tol)
            if is_ess:
                stab_details.append("Interior ESS: A restricted to tangent space is neg-def.")
            else:
                stab_details.append("Interior equilibrium but NOT ESS: tangent-space condition fails.")
        else:
            # Boundary mixed strategy: check reduced game
            A_sub = self.A[np.ix_(support, support)]
            is_ess = self._check_interior_ess_subgame(p[support], A_sub, tol)
            if is_ess:
                # Also verify no outside strategy can invade
                for j in range(self.n):
                    if j in support:
                        continue
                    if fitness[j] > max_fitness - tol:
                        is_ess = False
                        stab_details.append(f"Outside strategy {j} can invade.")
                        break
                if is_ess:
                    stab_details.append("Boundary ESS: reduced game stable, no outside invaders.")
            else:
                stab_details.append("Reduced game is not stable.")

        return ESSResult(
            strategy=p,
            is_nash=is_nash,
            is_ess=is_ess,
            label=f"Mixed strategy {np.array2string(p, precision=4)}",
            details=" | ".join(stab_details) if stab_details else "ESS analysis complete.",
        )

    def _check_interior_ess(
        self, p: NDArray[np.float64], tol: float
    ) -> bool:
        """
        Check ESS condition for a fully interior equilibrium.

        An interior equilibrium p* is ESS iff for all z in the tangent space
        of the simplex (sum z_i = 0, z != 0): z^T A z < 0.

        Equivalent: the matrix A restricted to tangent space has all
        negative eigenvalues.
        """
        return self._check_interior_ess_subgame(p, self.A, tol)

    @staticmethod
    def _check_interior_ess_subgame(
        p: NDArray[np.float64],
        A: NDArray[np.float64],
        tol: float,
    ) -> bool:
        """
        Check if a fully-supported equilibrium in a sub-game is ESS.

        Construct the tangent-space projection and check neg-definiteness.
        """
        n = len(p)
        if n <= 1:
            return True

        # Basis for tangent space of (n-1)-simplex: v_i = e_i - e_n, i=1..n-1
        V = np.zeros((n, n - 1))
        for i in range(n - 1):
            V[i, i] = 1.0
            V[n - 1, i] = -1.0

        # Project A onto tangent space
        A_tan = V.T @ A @ V  # (n-1) x (n-1)

        # Check if all eigenvalues are strictly negative
        eigenvalues = np.linalg.eigvalsh(
            (A_tan + A_tan.T) / 2  # symmetrize for real eigenvalues
        )
        return bool(np.all(eigenvalues < -tol))

    def find_interior_equilibrium(self) -> NDArray[np.float64] | None:
        """
        Find the interior equilibrium (if it exists).

        At an interior equilibrium, all strategies have equal fitness:
            (Ap*)_i = c  for all i, and sum(p*) = 1.

        We solve the augmented system:
            [ A_1 - A_n ]       [ 0 ]
            [ ...       ] p* =  [ . ]
            [ A_{n-1}-A_n]      [ 0 ]
            [ 1 ... 1   ]       [ 1 ]

        where A_i is the i-th row of A, giving n equations in n unknowns.
        This handles both invertible and singular payoff matrices.
        """
        try:
            # Build augmented system: first n-1 rows enforce equal fitness,
            # last row enforces sum = 1
            M = np.zeros((self.n, self.n))
            b = np.zeros(self.n)

            for i in range(self.n - 1):
                M[i] = self.A[i] - self.A[self.n - 1]

            M[self.n - 1] = np.ones(self.n)
            b[self.n - 1] = 1.0

            # Check if system is solvable
            det = np.linalg.det(M)
            if np.isclose(det, 0.0, atol=1e-12):
                # Try least-squares as fallback
                p, residuals, _, _ = np.linalg.lstsq(M, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
            else:
                p = np.linalg.solve(M, b)

            # Verify it's a valid probability vector
            if np.isclose(p.sum(), 1.0, atol=1e-8) and np.all(p > -1e-8):
                p = np.clip(p, 0.0, 1.0)
                p /= p.sum()

                # Verify equal fitness condition
                fitness = self.A @ p
                if np.allclose(fitness, fitness[0], atol=1e-8):
                    return p
            return None
        except np.linalg.LinAlgError:
            return None

    def analyze_all(self) -> list[ESSResult]:
        """
        Analyze all pure strategies and the interior equilibrium (if any).
        """
        results = []
        for i in range(self.n):
            results.append(self.check_pure(i))

        interior = self.find_interior_equilibrium()
        if interior is not None:
            results.append(self.check_mixed(interior))

        return results

    def hawk_dove_ess(self, V: float, C: float) -> dict:
        """
        Analytic ESS for Hawk-Dove game with value V and cost C.

        Payoff matrix:
            H vs H: (V-C)/2
            H vs D: V
            D vs H: 0
            D vs D: V/2

        ESS:
            If V >= C: pure Hawk
            If V < C:  mixed with p* = V/C (probability of Hawk)
        """
        A = np.array([
            [(V - C) / 2, V],
            [0, V / 2],
        ])

        if V >= C:
            return {
                "type": "pure",
                "strategy": "Hawk",
                "p_hawk": 1.0,
                "payoff_matrix": A,
            }
        else:
            p_star = V / C
            return {
                "type": "mixed",
                "strategy": f"Hawk with probability {p_star:.4f}",
                "p_hawk": p_star,
                "payoff_matrix": A,
            }
