# Replicator Dynamics

Simulate evolutionary game dynamics -- replicator equations, ESS analysis, Moran process, and interactive simplex phase portraits.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Overview

This project provides a complete toolkit for studying **evolutionary game theory** through computational methods:

- **Replicator Equations**: Continuous-time ODE model of strategy frequency evolution
- **ESS Analysis**: Automated verification of evolutionarily stable strategies
- **Moran Process**: Finite-population stochastic dynamics with fixation probabilities
- **Jacobian Stability**: Eigenvalue classification of fixed points
- **Mutation-Selection**: Replicator-mutator equation with bifurcation analysis
- **Simplex Phase Portraits**: Interactive 2D visualization of 3-strategy dynamics

## Theory

### The Replicator Equation

The replicator equation describes how the frequency of strategy $i$ changes over time in an infinite, well-mixed population:

$$\dot{x}_i = x_i \left[ (Ax)_i - x^\top A x \right]$$

where:
- $x_i$ is the fraction of the population playing strategy $i$
- $A$ is the payoff matrix
- $(Ax)_i = \sum_j A_{ij} x_j$ is the expected payoff of strategy $i$
- $\bar{f} = x^\top A x$ is the mean fitness of the population

The population state $x$ lives on the **simplex** $\Delta^{n-1} = \{x \in \mathbb{R}^n : x_i \geq 0, \sum_i x_i = 1\}$.

### Evolutionarily Stable Strategies (ESS)

A strategy $p^*$ is an ESS if for every mutant strategy $q \neq p^*$:

1. **Nash condition**: $u(p^*, p^*) \geq u(q, p^*)$
2. **Stability condition**: If $u(p^*, p^*) = u(q, p^*)$, then $u(p^*, q) > u(q, q)$

where $u(p, q) = p^\top A q$.

### Moran Process

The Moran process models evolution in a **finite population** of $N$ individuals:

1. Select an individual for reproduction proportional to fitness
2. The offspring replaces a uniformly random individual

For neutral drift (all fitnesses equal), the fixation probability of a single mutant is $\rho = 1/N$. Under frequency-dependent selection with fitness ratio $r$:

$$\rho = \frac{1 - 1/r}{1 - 1/r^N}$$

### Replicator-Mutator Equation

Adding mutation to selection:

$$\dot{x}_i = \sum_j x_j f_j(x) Q_{ji} - x_i \bar{f}(x)$$

where $Q_{ji}$ is the probability that an offspring of type $j$ becomes type $i$.

## Architecture

```
src/
 ├── core/
 │    ├── replicator.py    # ReplicatorODE, MultiPopulationReplicator, preset games
 │    ├── ess.py           # ESSChecker: Nash + stability conditions
 │    ├── moran.py         # MoranProcess: Monte Carlo + exact fixation prob
 │    ├── jacobian.py      # JacobianAnalyzer: eigenvalue stability classification
 │    └── mutations.py     # ReplicatorMutator: mutation-selection dynamics
 ├── viz/
 │    ├── simplex.py       # SimplexPlotter: barycentric coords, vector field, trajectories
 │    └── app.py           # Streamlit interactive dashboard
 └── cli.py               # Click-based CLI
```

### Key Design Decisions

| Component | Method | Why |
|---|---|---|
| ODE integration | `scipy.integrate.solve_ivp` (RK45) | Adaptive step size, well-tested |
| Simplex coordinates | Barycentric to Cartesian | Natural for 3-strategy visualization |
| Moran process | Monte Carlo + exact recursion | Handles frequency-dependent selection |
| ESS verification | Tangent-space eigenvalue check | Works for mixed strategies |
| Stability | Numerical Jacobian via central differences | Robust for arbitrary payoff matrices |

## Quickstart

### Install

```bash
git clone https://github.com/abhi-wadhwa/replicator-dynamics.git
cd replicator-dynamics
pip install -e ".[dev]"
```

### Run the Streamlit App

```bash
streamlit run src/viz/app.py
```

### CLI Usage

```bash
# Simulate Rock-Paper-Scissors
replicator simulate --game rps --x0 0.6,0.2,0.2

# ESS analysis
replicator ess --game hawk-dove

# Moran process
replicator moran --game hawk-dove -N 100 -w 1.0 --sims 10000

# Stability analysis
replicator stability --game rps
```

### Python API

```python
import numpy as np
from src.core.replicator import ReplicatorODE, PRESET_GAMES
from src.core.ess import ESSChecker
from src.core.moran import MoranProcess
from src.viz.simplex import SimplexPlotter

# Rock-Paper-Scissors dynamics
A = PRESET_GAMES["Rock-Paper-Scissors"]["matrix"]
ode = ReplicatorODE(A)
result = ode.solve(np.array([0.6, 0.2, 0.2]), t_span=(0, 100))

# ESS check
checker = ESSChecker(A)
for r in checker.analyze_all():
    print(f"{r.label}: ESS={r.is_ess}")

# Moran process (2-strategy game)
mp = MoranProcess(np.array([[3, 0], [5, 1]]), population_size=50)
result = mp.fixation_probability(n_simulations=10000)
print(f"Fixation prob: {result.fixation_probability:.4f}")

# Simplex phase portrait
plotter = SimplexPlotter(A, ["Rock", "Paper", "Scissors"])
fig = plotter.phase_portrait(
    trajectories=[np.array([0.6, 0.2, 0.2])],
    show_vector_field=True,
    show_fixed_points=True,
)
fig.savefig("phase_portrait.png")
```

## Preset Games

| Game | Strategies | Key Property |
|---|---|---|
| Rock-Paper-Scissors | R, P, S | Cyclic dominance, interior center |
| Hawk-Dove | H, D | Mixed ESS at p* = V/C |
| Coordination | A, B | Two pure ESS, bistability |
| Snowdrift | C, D | Coexistence equilibrium |
| Prisoner's Dilemma | C, D | Defect dominates |
| RPS-Asymmetric | R, P, S | Outward spiral |

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

Test highlights:
- **RPS closed orbits**: Lyapunov function conservation verifies trajectories don't spiral
- **Hawk-Dove ESS**: Analytical p* = V/C matches numerical result
- **Moran neutral drift**: Fixation probability converges to 1/N
- **Jacobian classification**: Eigenvalues correctly identify centers, saddles, stable nodes

## Docker

```bash
docker build -t replicator-dynamics .
docker run -p 8501:8501 replicator-dynamics
```

## License

MIT
