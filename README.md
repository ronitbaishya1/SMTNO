# SMTNO — Scale–Memory–Time Neural Operator (Research Development)

Scientific-machine-learning research exploring the transfer of effective coarse PDE dynamics, unresolved memory, and eventually local/global time evolution across physical scales.

## Code layout

- `src/burgers_1d/` — original FreeFEM++ Burgers numerical experiments (Step 1).
- `src/burgers_pilots/` — earlier Burgers, scale-switching, residual and memory pilots (Steps 2A–2D.4D).
- `src/advection_diffusion/` — FreeFEM++ benchmarks, spatial filtering, weak-form closure, boundary/time audits and FNO memory controls (Steps 3A–3C.2).
- `colab/` — cleaned historical Colab notebook and an optional source-only training launcher.
- `docs/` — experiment and numerical caveats.

## Status

The advection–diffusion benchmark is a numerical negative control for memory. The mobile–immobile transport model is the first physically motivated memory benchmark. Joint learned scale–memory transfer, local/global asynchronous time integration, and optimal control are **research objectives**, not implemented results.

All generated datasets, plots, metrics, model checkpoints and run logs are excluded. Full experimentation requires generating or supplying data separately. FreeFEM++ is required for the Step 3A and 3B.3 `.edp` solvers; it is not a Python package.

The `Project_Execution_legacy.ipynb` reflects the user's earlier Drive layout and may reference scripts or outputs no longer stored in those exact locations. Use the `.py` source scripts for the reproducible pipeline.
