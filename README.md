# replicator-dynamics

evolutionary game dynamics on the simplex. watch populations of strategies compete, coexist, or drive each other to extinction.

## what this is

- **replicator equations** — the ODE system where strategies grow in proportion to their fitness advantage. the core model of evolutionary game theory
- **ESS analysis** — find evolutionarily stable strategies. ESS implies asymptotic stability under replicator dynamics (not always the converse)
- **moran process** — the finite-population stochastic version. fixation probabilities, selection intensity
- **simplex phase portraits** — visualize 3-strategy dynamics as vector fields on the triangle

## running it

```bash
pip install -r requirements.txt
python main.py
```

## the math

the replicator equation: dx_i/dt = x_i(f_i(x) - f_bar(x)), where f_i is the fitness of strategy i and f_bar is the population average. strategies that do better than average grow; those that do worse shrink.

the connection to game theory: rest points of the replicator dynamics are nash equilibria. ESS are asymptotically stable rest points. the simplex portraits for rock-paper-scissors show the famous cycling behavior — no equilibrium is reached, populations orbit forever.
