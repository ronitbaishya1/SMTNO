# SMTNO experiment status (2026-11-07)

- **Steps 2A–2D.4D**: Legacy Burgers / scale-switch / memory pilots. Preserved for research history, not the final proposed architecture.
- **Steps 3A–3A.5**: Published advection–diffusion-reaction FEM verification. Smooth benchmark converged; the very sharp-layer `epsilon=1e-8` example was not resolved on a 64-by-64 grid.
- **Steps 3B.1–3B.4**: Dirichlet-compatible spatial filtering, P1 projection and 30-trajectory source-free weak-form closure dataset.
- **Steps 3B.5–3C.0**: Numerical audits found boundary-localized closure and significant sensitivity of residual labels to saved snapshot cadence; refined observation intervals improved consistency.
- **Step 3C.1**: Same-capacity current-state versus four-frame history FNOs on advection–diffusion. No strong evidence for necessary physical memory.
- **Step 3C.2A**: Three training seeds, 70 epochs; at unseen `N=64` history gains averaged around 1.5–2.2% across horizons, with seed variation. This is not proof of physical memory.
- **Step 3C.2B**: Paired mobile–immobile test uses established MRMT exchange equations with illustrative, dimensionless parameters. One reported seed showed history gains around 63–77% across resolutions/horizons. Requires replication and stronger full-state numerical baselines before architectural claims.

## Next research direction

Audit and identify known physical memory kernels; test cross-filter memory transfer, not only history-input FNOs. Later evaluate local/global time stepping and eventually PDE-constrained optimal control using state/adjoint optimization.

## Numerical cautions

`3B.4` closure includes boundary, projection and temporal-differentiation contributions. The original `step3a_cdr2d.edp` is corrected for earlier parser issues but its full FreeFEM++ SUPG/PINN-paper method is *not* implemented. The published PDF benchmark is not reproduced exactly by the simplified solver. Do not interpret current prediction gains as proof of a novel architecture.
