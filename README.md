# SMTNO — Scale–Memory–Time Neural Operator

Research code for multiscale numerical PDEs, effective coarse-scale closure, history-dependent learning and future local/global time integration.

## Repository layout

- `src/burgers_pilots/` — earlier FNO, coarse closure, scale-switching and memory experiments (Steps 2A–2D.4).
- `src/advection_diffusion/` — FreeFEM++ and Python benchmarks, filtering, closure and numerical audits (Steps 3A–3C.1).
- `colab/` — GPU training launcher; source files remain ordinary Python programs.
- `docs/` — Mac setup, Colab workflow and research status.

**No simulation datasets, checkpoints, output images, results directories or logs are tracked.** Code needed for Step 3C.2 is in the accompanying source package but its four individual scripts are not yet committed.

## Research status

Advection–diffusion provides a useful numerical negative control for memory, while multiple-rate mobile–immobile transport is being used as a physically motivated memory benchmark. A learned joint scale–memory transformation, asynchronous local/global time stepping, and optimal control remain planned work, not validated results.

FreeFEM++ is needed for the `.edp` solvers. Python dependencies are in `requirements.txt`.
