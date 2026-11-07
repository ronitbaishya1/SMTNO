"""SMTNO Stage 1: validate and plot 1-D periodic Burgers FEM snapshots.

Place this file in: SMTNO/FEM/burgers_1d/validate_burgers.py
Run: python FEM/burgers_1d/validate_burgers.py
Requires: numpy, pandas, matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
CSV = PROJECT / 'datasets' / 'raw' / 'burgers_1d.csv'
OUT = PROJECT / 'results' / 'burgers_1d'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
assert set(df.columns) == {'time', 'x', 'u'}
assert not df.isna().any().any(), 'CSV contains missing values'

rows = []
fig, ax = plt.subplots(figsize=(9, 5))
times = np.sort(df.time.unique())
for t in times:
    s = df[np.isclose(df.time, t)].sort_values('x')
    x, u = s.x.to_numpy(), s.u.to_numpy()
    assert len(x) == len(df[df.time == times[0]]), 'Unexpected mesh-size change'
    assert np.all(np.diff(x) > 0), 'Non-increasing x coordinates'
    mass = np.trapezoid(u, x)
    energy = 0.5 * np.trapezoid(u**2, x)
    rows.append({'time': t, 'mass_from_saved_csv': mass,
                 'energy': energy, 'endpoint_mismatch': abs(u[0]-u[-1])})
    if np.isclose(t, times[::2]).any():
        ax.plot(x, u, label=f't={t:.2f}')
ax.set(xlabel='x', ylabel='u(x,t)', title='1D viscous Burgers: FEM evolution')
ax.grid(alpha=0.2); ax.legend(ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'burgers_evolution.png', dpi=170)
plt.close(fig)

m = pd.DataFrame(rows)
m.to_csv(OUT / 'validation_metrics.csv', index=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(m.time, m.energy, 'o-')
ax.set(xlabel='Time', ylabel='Energy = 1/2 integral(u^2 dx)',
       title='Burgers energy decay')
ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(OUT / 'burgers_energy.png', dpi=170)
plt.close(fig)

print(f'Loaded {len(times)} snapshots from: {CSV}')
print(f'Max periodic endpoint mismatch: {m.endpoint_mismatch.max():.3e}')
print(f'Max CSV-integrated |mass|: {m.mass_from_saved_csv.abs().max():.3e}')
print(f'Energy: {m.energy.iloc[0]:.8f} -> {m.energy.iloc[-1]:.8f}')
print(f'Energy decreases at each saved time: {(m.energy.diff().dropna() < 0).all()}')
print(f'Plots and metrics saved to: {OUT}')
