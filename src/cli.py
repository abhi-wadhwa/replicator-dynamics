"""
Command-line interface for replicator dynamics.

Usage:
    replicator simulate --game hawk-dove --x0 0.5,0.5
    replicator ess --game rps
    replicator moran --N 100 --w 1.0 --sims 10000
    replicator app  (launches Streamlit)
"""

from __future__ import annotations

import json
import sys

import click
import numpy as np

from src.core.ess import ESSChecker
from src.core.jacobian import JacobianAnalyzer
from src.core.moran import MoranProcess
from src.core.replicator import PRESET_GAMES, ReplicatorODE

GAME_ALIASES = {
    "rps": "Rock-Paper-Scissors",
    "hawk-dove": "Hawk-Dove",
    "coordination": "Coordination",
    "snowdrift": "Snowdrift",
    "pd": "Prisoner's Dilemma",
    "rps-asym": "RPS-Asymmetric",
}


def resolve_game(name: str) -> tuple[np.ndarray, list[str]]:
    """Resolve game name (alias or full) to payoff matrix and labels."""
    full_name = GAME_ALIASES.get(name.lower(), name)
    if full_name in PRESET_GAMES:
        game = PRESET_GAMES[full_name]
        return game["matrix"], game["strategies"]
    raise click.BadParameter(
        f"Unknown game '{name}'. Available: {list(PRESET_GAMES.keys())} "
        f"or aliases: {list(GAME_ALIASES.keys())}"
    )


@click.group()
def main() -> None:
    """Evolutionary game dynamics simulator."""
    pass


@main.command()
@click.option("--game", "-g", default="rps", help="Preset game name or alias")
@click.option("--x0", default=None, help="Initial condition (comma-separated)")
@click.option("--t-end", default=50.0, help="Integration time")
@click.option("--output", "-o", default=None, help="Output file (JSON)")
def simulate(game: str, x0: str | None, t_end: float, output: str | None) -> None:
    """Simulate replicator dynamics."""
    matrix, labels = resolve_game(game)
    n = matrix.shape[0]

    if x0 is None:
        x0_arr = np.ones(n) / n
    else:
        x0_arr = np.array([float(v) for v in x0.split(",")])

    click.echo(f"Game: {game} ({n} strategies: {labels})")
    click.echo(f"Payoff matrix:\n{matrix}")
    click.echo(f"Initial condition: {x0_arr}")

    ode = ReplicatorODE(matrix)
    result = ode.solve(x0_arr, t_span=(0.0, t_end))

    click.echo(f"\nFinal state (t={t_end}):")
    for i, label in enumerate(labels):
        click.echo(f"  {label}: {result.final_state[i]:.6f}")

    if output:
        data = {
            "game": game,
            "payoff_matrix": matrix.tolist(),
            "x0": x0_arr.tolist(),
            "t_end": t_end,
            "final_state": result.final_state.tolist(),
            "t": result.t.tolist(),
            "x": result.x.tolist(),
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"Results saved to {output}")


@main.command()
@click.option("--game", "-g", default="rps", help="Preset game name or alias")
def ess(game: str) -> None:
    """Analyze evolutionarily stable strategies."""
    matrix, labels = resolve_game(game)

    click.echo(f"ESS Analysis: {game}")
    click.echo(f"Payoff matrix:\n{matrix}\n")

    checker = ESSChecker(matrix)
    results = checker.analyze_all()

    for r in results:
        status = "ESS" if r.is_ess else ("Nash" if r.is_nash else "---")
        click.echo(f"[{status:>4}] {r.label}")
        click.echo(f"       {r.details}")


@main.command()
@click.option("--game", "-g", default="hawk-dove", help="Preset game (2-strategy)")
@click.option("-N", "--pop-size", default=50, help="Population size")
@click.option("-w", "--intensity", default=1.0, help="Selection intensity")
@click.option("--sims", default=10000, help="Number of Monte Carlo simulations")
@click.option("--initial", default=1, help="Initial number of type A")
def moran(game: str, pop_size: int, intensity: float, sims: int, initial: int) -> None:
    """Run Moran process simulation."""
    matrix, labels = resolve_game(game)

    if matrix.shape != (2, 2):
        click.echo("Error: Moran process requires a 2-strategy game.", err=True)
        sys.exit(1)

    click.echo(f"Moran Process: {game}, N={pop_size}, w={intensity}")
    click.echo(f"Running {sims} simulations...")

    mp = MoranProcess(matrix, population_size=pop_size, intensity=intensity)
    result = mp.fixation_probability(initial_A=initial, n_simulations=sims)

    exact = mp.fixation_probability_exact(initial)

    click.echo("\nResults:")
    click.echo(f"  Fixation probability (MC):    {result.fixation_probability:.4f}")
    click.echo(f"  Fixation probability (exact): {exact:.4f}")
    click.echo(f"  Neutral drift (1/N):          {1/pop_size:.4f}")
    click.echo(f"  Fixations: {result.n_fixations}/{result.n_simulations}")
    click.echo(f"  Mean fixation time: {result.mean_fixation_time:.0f} steps")


@main.command()
@click.option("--game", "-g", default="rps", help="Preset game name or alias")
def stability(game: str) -> None:
    """Analyze fixed point stability via Jacobian eigenvalues."""
    matrix, labels = resolve_game(game)

    click.echo(f"Stability Analysis: {game}")
    click.echo(f"Payoff matrix:\n{matrix}\n")

    analyzer = JacobianAnalyzer(matrix)
    results = analyzer.full_analysis()

    for fp in results:
        click.echo(f"Fixed point: {fp.label}")
        click.echo(f"  x = {np.array2string(fp.point, precision=4)}")
        eigstr = ", ".join([f"{e:.4f}" for e in fp.eigenvalues])
        click.echo(f"  Eigenvalues: {eigstr}")
        click.echo(f"  Classification: {fp.stability.value}")
        click.echo()


@main.command()
@click.option("--port", default=8501, help="Streamlit port")
def app(port: int) -> None:
    """Launch the Streamlit web application."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/viz/app.py",
        "--server.port", str(port),
    ])


if __name__ == "__main__":
    main()
