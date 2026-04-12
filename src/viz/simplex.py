"""
Simplex phase portrait visualization for 3-strategy games.

The 2-simplex (triangle) is the space of probability distributions over
3 strategies. We convert barycentric coordinates to 2D Cartesian
coordinates for plotting.

Key features:
  - Vector field: arrows showing direction and magnitude of dx/dt
  - Trajectories: ODE solutions from selected initial conditions
  - Fixed points: marked and color-coded by stability
  - Basins of attraction: shaded regions
"""

from __future__ import annotations

from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

from src.core.replicator import ReplicatorODE
from src.core.jacobian import JacobianAnalyzer, StabilityType


# ── Coordinate transforms ────────────────────────────────────────────

def bary_to_cart(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert barycentric coordinates to 2D Cartesian.

    For a 3-simplex with vertices at:
      v0 = (0, 0)       [strategy 0, bottom-left]
      v1 = (1, 0)       [strategy 1, bottom-right]
      v2 = (0.5, sqrt(3)/2) [strategy 2, top]

    Point p = x0*v0 + x1*v1 + x2*v2.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.array([
            x[1] + 0.5 * x[2],
            x[2] * np.sqrt(3) / 2,
        ])
    # Batch: x has shape (..., 3)
    cart_x = x[..., 1] + 0.5 * x[..., 2]
    cart_y = x[..., 2] * np.sqrt(3) / 2
    return np.stack([cart_x, cart_y], axis=-1)


def cart_to_bary(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert 2D Cartesian back to barycentric coordinates.
    """
    p = np.asarray(p, dtype=np.float64)
    h = np.sqrt(3) / 2

    x2 = p[..., 1] / h
    x1 = p[..., 0] - 0.5 * x2
    x0 = 1.0 - x1 - x2
    return np.stack([x0, x1, x2], axis=-1)


def simplex_grid(
    resolution: int = 30,
) -> NDArray[np.float64]:
    """
    Generate a grid of points on the 2-simplex interior.

    Returns array of shape (n_points, 3) with barycentric coordinates.
    """
    points = []
    eps = 0.02  # Stay away from boundary
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            x = np.array([i, j, k], dtype=np.float64) / resolution
            if np.all(x > eps):
                points.append(x)
    return np.array(points)


# ── Plotting ─────────────────────────────────────────────────────────

class SimplexPlotter:
    """
    Create simplex phase portraits for 3-strategy replicator dynamics.
    """

    def __init__(
        self,
        payoff_matrix: NDArray[np.float64],
        strategy_labels: list[str] | None = None,
    ) -> None:
        if payoff_matrix.shape != (3, 3):
            raise ValueError(
                f"SimplexPlotter requires 3x3 payoff matrix, got {payoff_matrix.shape}"
            )
        self.A = payoff_matrix
        self.ode = ReplicatorODE(payoff_matrix)
        self.labels = strategy_labels or ["S0", "S1", "S2"]

    def draw_simplex_boundary(self, ax: Axes) -> None:
        """Draw the simplex triangle boundary."""
        vertices = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3) / 2],
            [0, 0],  # close the triangle
        ])
        ax.plot(vertices[:, 0], vertices[:, 1], "k-", linewidth=1.5)

        # Labels
        offset = 0.06
        ax.text(-offset, -offset, self.labels[0],
                fontsize=12, fontweight="bold", ha="center")
        ax.text(1 + offset, -offset, self.labels[1],
                fontsize=12, fontweight="bold", ha="center")
        ax.text(0.5, np.sqrt(3) / 2 + offset, self.labels[2],
                fontsize=12, fontweight="bold", ha="center")

    def plot_vector_field(
        self,
        ax: Axes,
        resolution: int = 20,
        color: str = "steelblue",
        alpha: float = 0.6,
        scale: float | None = None,
    ) -> None:
        """
        Draw velocity arrows on the simplex.

        Computes dx/dt at grid points in barycentric coordinates,
        converts to Cartesian, and plots as a quiver field.
        """
        grid = simplex_grid(resolution)
        cart_points = bary_to_cart(grid)

        # Compute velocities in barycentric coordinates
        velocities_bary = np.array([
            self.ode.velocity_field(x) for x in grid
        ])

        # Convert velocity to Cartesian
        # v_cart = J * v_bary where J is the Jacobian of the transform
        # For our mapping: cart_x = x1 + 0.5*x2, cart_y = x2*sqrt(3)/2
        # dv_cart_x = dv_x1 + 0.5 * dv_x2
        # dv_cart_y = dv_x2 * sqrt(3)/2
        vx = velocities_bary[:, 1] + 0.5 * velocities_bary[:, 2]
        vy = velocities_bary[:, 2] * np.sqrt(3) / 2

        # Normalize arrow lengths for visibility
        magnitudes = np.sqrt(vx**2 + vy**2)
        max_mag = magnitudes.max()
        if max_mag > 0:
            if scale is None:
                scale = max_mag * 20

        ax.quiver(
            cart_points[:, 0], cart_points[:, 1],
            vx, vy,
            magnitudes,
            cmap="Blues",
            alpha=alpha,
            scale=scale,
            width=0.004,
            headwidth=4,
            headlength=5,
        )

    def plot_trajectory(
        self,
        ax: Axes,
        x0: NDArray[np.float64],
        t_span: tuple[float, float] = (0.0, 50.0),
        n_points: int = 2000,
        color: str = "red",
        linewidth: float = 1.2,
        alpha: float = 0.8,
        show_start: bool = True,
    ) -> None:
        """
        Integrate and plot a single trajectory on the simplex.
        """
        result = self.ode.solve(x0, t_span=t_span, n_points=n_points)
        cart = bary_to_cart(result.x.T)

        ax.plot(cart[:, 0], cart[:, 1], color=color,
                linewidth=linewidth, alpha=alpha)

        if show_start:
            ax.plot(cart[0, 0], cart[0, 1], "o", color=color,
                    markersize=5, zorder=5)

    def plot_fixed_points(self, ax: Axes) -> None:
        """
        Find and plot fixed points, colored by stability type.
        """
        analyzer = JacobianAnalyzer(self.A)
        results = analyzer.full_analysis()

        stability_colors = {
            StabilityType.STABLE_NODE: "green",
            StabilityType.STABLE_FOCUS: "green",
            StabilityType.UNSTABLE_NODE: "red",
            StabilityType.UNSTABLE_FOCUS: "red",
            StabilityType.SADDLE: "orange",
            StabilityType.CENTER: "blue",
            StabilityType.NON_HYPERBOLIC: "gray",
        }

        stability_markers = {
            StabilityType.STABLE_NODE: "o",
            StabilityType.STABLE_FOCUS: "o",
            StabilityType.UNSTABLE_NODE: "^",
            StabilityType.UNSTABLE_FOCUS: "^",
            StabilityType.SADDLE: "s",
            StabilityType.CENTER: "D",
            StabilityType.NON_HYPERBOLIC: "x",
        }

        for fp_result in results:
            cart = bary_to_cart(fp_result.point)
            color = stability_colors.get(fp_result.stability, "gray")
            marker = stability_markers.get(fp_result.stability, "o")

            ax.plot(
                cart[0], cart[1],
                marker=marker,
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )

    def plot_basins_of_attraction(
        self,
        ax: Axes,
        resolution: int = 50,
        t_end: float = 100.0,
        alpha: float = 0.15,
    ) -> None:
        """
        Shade the simplex by basin of attraction.

        For each grid point, integrate forward and see which vertex
        (or interior) the trajectory converges to.
        """
        grid = simplex_grid(resolution)
        cart_points = bary_to_cart(grid)

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]  # for each attractor

        basin_labels = np.zeros(len(grid), dtype=int)
        for idx, x0 in enumerate(grid):
            result = self.ode.solve(x0, t_span=(0, t_end), n_points=500)
            final = result.final_state
            # Classify by dominant strategy
            basin_labels[idx] = int(np.argmax(final))

        for label in range(3):
            mask = basin_labels == label
            if mask.any():
                ax.scatter(
                    cart_points[mask, 0],
                    cart_points[mask, 1],
                    c=colors[label],
                    alpha=alpha,
                    s=30,
                    marker="s",
                    edgecolors="none",
                )

    def phase_portrait(
        self,
        trajectories: list[NDArray[np.float64]] | None = None,
        show_vector_field: bool = True,
        show_fixed_points: bool = True,
        show_basins: bool = False,
        vector_resolution: int = 20,
        title: str = "Simplex Phase Portrait",
        figsize: tuple[float, float] = (8, 7),
        trajectory_colors: list[str] | None = None,
    ) -> Figure:
        """
        Create a complete simplex phase portrait.

        Parameters
        ----------
        trajectories : list of arrays
            Initial conditions for trajectories to plot.
        show_vector_field : bool
            Whether to show velocity arrows.
        show_fixed_points : bool
            Whether to mark and classify fixed points.
        show_basins : bool
            Whether to shade basins of attraction.
        vector_resolution : int
            Grid density for vector field.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        trajectory_colors : list of str
            Colors for each trajectory.

        Returns
        -------
        matplotlib Figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw basins first (background)
        if show_basins:
            self.plot_basins_of_attraction(ax)

        # Simplex boundary
        self.draw_simplex_boundary(ax)

        # Vector field
        if show_vector_field:
            self.plot_vector_field(ax, resolution=vector_resolution)

        # Trajectories
        if trajectories is not None:
            if trajectory_colors is None:
                cmap = plt.cm.Set1
                trajectory_colors = [
                    cmap(i / max(len(trajectories) - 1, 1))
                    for i in range(len(trajectories))
                ]
            for i, x0 in enumerate(trajectories):
                color = trajectory_colors[i] if i < len(trajectory_colors) else "red"
                self.plot_trajectory(ax, np.asarray(x0), color=color)

        # Fixed points
        if show_fixed_points:
            self.plot_fixed_points(ax)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        fig.tight_layout()
        return fig


def plot_time_series(
    payoff_matrix: NDArray[np.float64],
    x0: NDArray[np.float64],
    t_span: tuple[float, float] = (0.0, 50.0),
    strategy_labels: list[str] | None = None,
    title: str = "Population Dynamics",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """
    Plot population fractions over time.
    """
    ode = ReplicatorODE(payoff_matrix)
    result = ode.solve(np.asarray(x0), t_span=t_span)

    n = result.n_strategies
    labels = strategy_labels or [f"Strategy {i}" for i in range(n)]
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        ax.plot(result.t, result.x[i], label=labels[i],
                color=colors[i], linewidth=2)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Population Fraction", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_bifurcation(
    payoff_matrix: NDArray[np.float64],
    mu_values: NDArray[np.float64] | None = None,
    strategy_labels: list[str] | None = None,
    title: str = "Bifurcation Diagram (Mutation Rate)",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """
    Plot equilibrium strategy frequencies as a function of mutation rate.
    """
    from src.core.mutations import ReplicatorMutator

    if mu_values is None:
        mu_values = np.linspace(0.0, 0.3, 60)

    rm = ReplicatorMutator(payoff_matrix, mutation_rate=0.0)
    mus, equilibria = rm.bifurcation_data(mu_values)

    n = payoff_matrix.shape[0]
    labels = strategy_labels or [f"Strategy {i}" for i in range(n)]
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        ax.plot(mus, equilibria[:, i], label=labels[i],
                color=colors[i], linewidth=2)

    ax.set_xlabel("Mutation Rate (mu)", fontsize=12)
    ax.set_ylabel("Equilibrium Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_moran_trajectory(
    trajectory: list[int],
    population_size: int,
    title: str = "Moran Process Trajectory",
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Plot a single Moran process trajectory."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(trajectory, color="steelblue", linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4, label="Extinction")
    ax.axhline(y=population_size, color="green", linestyle="--",
               alpha=0.4, label="Fixation")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Number of Type A", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-1, population_size + 1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
