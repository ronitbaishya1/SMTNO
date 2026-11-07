#!/usr/bin/env python3
"""SMTNO Step 2D.4C: audit causal derivative shortcuts and resolution scaling.

Requires step2d4a_generate.py and step2d4b_compare.py beside this file.
Reads the dataset and saved test_metrics.csv; does not train models or change old files.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from step2d4a_generate import transport_matrix
from step2d4b_compare import split_base_ids


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--min-history', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    output = args.results_root / 'audit_2D4C'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as obj:
        d = {key: obj[key] for key in obj.files}
    times = np.asarray(d['times'], dtype=np.float64)
    dt = float(times[1] - times[0])
    if not np.allclose(np.diff(times), dt, rtol=1e-5, atol=1e-7):
        raise ValueError('Data must have uniformly sampled observation times.')
    tstart = max(int(args.min_history) - 1, 2)
    if tstart >= len(times):
        raise ValueError('Not enough time snapshots for the requested min-history.')
    test_ids = split_base_ids(d['base_id'], seed=args.seed)['test']
    trial_mask = np.isin(d['base_id'], test_ids)
    if not np.any(trial_mask):
        raise ValueError('No held-out trajectories matched the split.')
    regimes = [str(r) for r in d['regimes']]
    n_values = sorted(map(int, d['n_values']))
    metrics, scaling, per_case = [], [], []
    ell_values = np.asarray(d['ell'], dtype=float)
    for regime in regimes:
        for n in n_values:
            u = np.asarray(d[f'{regime}_u_N{n}'][trial_mask], dtype=np.float64)
            residual = np.asarray(d[f'{regime}_r_N{n}'][trial_mask], dtype=np.float64)
            # Match Step 2D.4A's coarse reference operator exactly.
            diff_hom = .003 * np.sqrt(3.0)
            operator = transport_matrix(n, np.full(n, diff_hom), .35)
            spatial_rhs = (operator @ u.reshape(-1, n).T).T.reshape(u.shape)
            reference = residual[:, :, tstart:, :]
            bdf1 = (u[:, :, tstart:, :] - u[:, :, tstart-1:-1, :]) / dt - spatial_rhs[:, :, tstart:, :]
            bdf2 = (3*u[:, :, tstart:, :] - 4*u[:, :, tstart-1:-1, :] + u[:, :, tstart-2:-2, :]) / (2*dt) - spatial_rhs[:, :, tstart:, :]
            candidates = {'zero': np.zeros_like(reference), 'bdf1': bdf1, 'bdf2': bdf2}
            selected_base_ids = d['base_id'][trial_mask]
            for j, ell in enumerate(ell_values):
                y = reference[:, j]
                y_rms = float(np.sqrt(np.mean(y**2)))
                scaling.append({'regime': regime, 'N': n, 'ell': float(ell), 'heldout_target_rms': y_rms,
                                'eligible_time_start': float(times[tstart]), 'num_base_ids': int(len(test_ids))})
                for method, val in candidates.items():
                    err = y - val[:, j]
                    row = {'regime': regime, 'N': n, 'ell': float(ell), 'method': method,
                           'rmse': float(np.sqrt(np.mean(err**2))),
                           'relative_l2': float(np.linalg.norm(err) / max(np.linalg.norm(y), 1e-12)),
                           'samples': int(y.shape[0]*y.shape[1])}
                    metrics.append(row)
                    for bid in test_ids:
                        ix = selected_base_ids == bid
                        target_case = y[ix]
                        error_case = err[ix]
                        per_case.append({'regime': regime, 'N': n, 'ell': float(ell), 'method': method,
                                         'base_id': int(bid), 'rmse': float(np.sqrt(np.mean(error_case**2))),
                                         'reference_rms': float(np.sqrt(np.mean(target_case**2)))})
    save_csv(output/'causal_derivative_metrics.csv', metrics)
    save_csv(output/'cross_resolution_target_rms.csv', scaling)
    save_csv(output/'causal_derivative_by_case.csv', per_case)

    # Figure 1: three physically distinct regimes, one chart per mesh N.
    for n in n_values:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        labels = [f'{regime}\nell={ell:.2f}' for regime in regimes for ell in ell_values]
        x = np.arange(len(labels), dtype=float)
        for j, method in enumerate(['zero', 'bdf1', 'bdf2']):
            yvals = [next(row['relative_l2'] for row in metrics if row['regime']==r and row['N']==n and
                          abs(row['ell']-ell)<1e-6 and row['method']==method)
                     for r in regimes for ell in ell_values]
            ax.bar(x+(j-1)*.25, yvals, width=.25, label=method)
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.set(ylabel='Relative L2 residual error', title=f'Causal derivative baselines, held-out N={n}')
        ax.grid(axis='y', alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/f'causal_derivative_comparison_N{n}.png', dpi=180)
        plt.close(fig)

    # Figure 2: inspect how the physical residual target changes with mesh resolution.
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    for regime in regimes:
        vals = []
        for ell in ell_values:
            s = [next(row['heldout_target_rms'] for row in scaling if row['regime']==regime and
                      row['N']==n and abs(row['ell']-ell)<1e-6) for n in n_values]
            vals.append(s[-1]/max(s[0], 1e-12))
        ax.plot(ell_values, vals, marker='o', label=regime)
    ax.axhline(1.0, linestyle='--', alpha=.6)
    ax.set(xlabel='Physical filter width ell', ylabel=f'Target RMS ratio: N={n_values[-1]} / N={n_values[0]}',
           title='Resolution dependence of the closure target')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'cross_resolution_target_scaling.png', dpi=180)
    plt.close(fig)

    # Figure 3: compare the causal derivative baseline to the saved H=4 neural results.
    h4_file = args.results_root/'training'/'multirate'/'h4'/'test_metrics.csv'
    if h4_file.is_file():
        with h4_file.open(newline='') as f:
            previous = list(csv.DictReader(f))
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for n in n_values:
            neural = [next(float(row['relative_l2']) for row in previous
                           if row['model']=='history_fno' and int(row['N'])==n and abs(float(row['ell'])-ell)<1e-6)
                      for ell in ell_values]
            causal = [next(row['relative_l2'] for row in metrics if row['regime']=='multirate' and
                           row['method']=='bdf2' and row['N']==n and abs(row['ell']-ell)<1e-6)
                      for ell in ell_values]
            ax.plot(ell_values, neural, marker='o', label=f'History FNO, N={n}')
            ax.plot(ell_values, causal, marker='s', label=f'Causal BDF2, N={n}')
        ax.set(xlabel='Physical filter width ell', ylabel='Relative L2 forcing-residual error',
               title='Multirate: history FNO vs simple numerical differentiation')
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/'multirate_history_vs_bdf2.png', dpi=180)
        plt.close(fig)
    print('Done:', output)
    print('Held-out base IDs:', list(map(int, test_ids)))
    print('First eligible time:', float(times[tstart]))

if __name__ == '__main__':
    main()