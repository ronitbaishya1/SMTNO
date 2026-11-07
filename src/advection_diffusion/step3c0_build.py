from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuses the exact filtering, projection, and FEM operators from the previous steps.
from step3b1_filter_project import dirichlet_gaussian, prepare_projection
from step3b2_weak_closure import fem_operators, mass_norm
from step3b4_build import calculate_coarse
from step3b6b_cadence import load_three


def parse_values(raw, typ):
    vals = [typ(s.strip()) for s in raw.split(',') if s.strip()]
    if not vals or len(vals) != len(set(vals)):
        raise ValueError('Expected a nonempty list of unique comma-separated values.')
    return vals


def weighted_norm(v, weights, mask=None):
    w = weights if mask is None else weights * mask
    return float(np.sqrt(max(float(np.sum(w * np.square(v))), 0.0)))


def masks(interior, n, width):
    xx = (interior % (n + 1)) / n
    yy = (interior // (n + 1)) / n
    dist = np.minimum.reduce([xx, 1 - xx, yy, 1 - yy])
    boundary = dist <= width + 1e-12
    top = yy >= 1 - width - 1e-12
    return {'boundary': boundary, 'interior': ~boundary, 'top': top}


def restore(a, interior, n):
    out = np.zeros((a.shape[0], (n + 1)**2), dtype=np.float32)
    out[:, interior] = a.astype(np.float32)
    out[:2] = np.nan  # No causal BDF2 label exists for the first two snapshots.
    return out.reshape(a.shape[0], n + 1, n + 1)


def load_splits(folder, cases):
    path = folder / 'manifest.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing original split manifest: {path}')
    m = pd.read_csv(path)
    if not {'case_id', 'split'}.issubset(m.columns):
        raise ValueError('The Step 3B3 manifest needs case_id and split columns.')
    if m.case_id.duplicated().any():
        raise ValueError('Duplicated case IDs in the original split manifest.')
    split = dict(zip(m.case_id.astype(int), m.split.astype(str)))
    if any(i not in split for i in cases):
        raise ValueError('Requested case missing from original split manifest.')
    if any(split[i] not in {'train', 'val', 'test'} for i in cases):
        raise ValueError('Unexpected data split name.')
    return split


def main():
    p = argparse.ArgumentParser(description='Step 3C.0: refined causal closure labels and boundary/interior audit')
    p.add_argument('--baseline', type=Path, default=Path('results/step3B3'))
    p.add_argument('--saved005', type=Path, default=Path('results/step3B6/FEM_save4'))
    p.add_argument('--saved0025', type=Path, default=Path('results/step3B6/FEM_save2'))
    p.add_argument('--outdir', type=Path, default=Path('results/step3C0'))
    p.add_argument('--cases', type=int, default=2)
    p.add_argument('--coarse', default='32,64')
    p.add_argument('--ells', default='0.01,0.02,0.04')
    p.add_argument('--boundary-width', type=float, default=0.1)
    args = p.parse_args()

    ns, ells = parse_values(args.coarse, int), parse_values(args.ells, float)
    if args.cases < 1 or min(ns) < 4 or min(ells) <= 0 or not 0 < args.boundary_width < 0.5:
        p.error('Invalid cases, coarse mesh, filter width or boundary width.')
    folders = [x.expanduser().resolve() for x in (args.baseline, args.saved005, args.saved0025)]
    out = args.outdir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(folders[0], range(args.cases))
    compare_rows, region_rows, summary_rows = [], [], []

    for case_id in range(args.cases):
        runs = load_three(folders, case_id)  # Also checks matching physical FEM states at shared times.
        nf = runs[0]['u'].shape[1] - 1
        if any(n >= nf or nf % n for n in ns):
            raise ValueError(f'Coarse meshes must divide the fine grid N={nf}.')
        if not np.allclose([r['step'] for r in runs], [0.01, 0.005, 0.0025], rtol=0, atol=1e-8):
            raise ValueError('Expected output intervals of 0.01, 0.005 and 0.0025.')
        case_out = out / f'case_{case_id:03d}'
        case_out.mkdir(exist_ok=True)
        print(f'Case {case_id:03d}: filtering {sum(len(r["t"]) for r in runs)} snapshots per scale', flush=True)
        # Filter once per (saved cadence, physical scale), reusing at both coarse meshes.
        for ell in ells:
            filtered = [np.stack([dirichlet_gaussian(u, ell) for u in r['u']]) for r in runs]
            for n in ns:
                proj = prepare_projection(n)
                ops = fem_operators(n, runs[0]['eps'], proj)
                M, A, interior, lu, *_ = ops
                Mii = M[interior][:, interior].tocsr()
                weights = np.asarray(Mii.sum(axis=1)).reshape(-1)
                if np.any(weights <= 0):
                    raise ValueError('Nonpositive lumped FEM mass weight.')
                reg = masks(interior, n, args.boundary_width)
                computed = []
                for run, f in zip(runs, filtered):
                    u_coarse, cbdf, ccenter, weak, details, I = calculate_coarse(
                        f, run['t'], n, run['eps'], proj, ops)
                    if not np.array_equal(I, interior):
                        raise ValueError('Inconsistent FEM interior DOF ordering.')
                    ui = u_coarse.reshape(len(run['t']), -1)[:, interior]
                    dt = run['step']
                    du = np.full_like(ui, np.nan)
                    du[2:] = (3*ui[2:] - 4*ui[1:-1] + ui[:-2])/(2*dt)
                    computed.append((u_coarse, cbdf, ccenter, du))

                t_f = runs[-1]['t']
                u_f, c_f, c_center_f, du_f = computed[-1]
                np.savez_compressed(
                    case_out / f'coarse_n{n}_ell{ell:g}_refined.npz',
                    u_coarse=u_f.astype(np.float32),
                    closure_rate_bdf2_CAUSAL=restore(c_f, interior, n),
                    closure_rate_centered_OFFLINE_ONLY=restore(c_center_f, interior, n),
                    t=t_f, x=np.linspace(0, 1, n+1), y=np.linspace(0, 1, n+1),
                    n_coarse=n, ell=ell, case_id=case_id, split=splits[case_id],
                    epsilon=runs[-1]['eps'], saved_dt=runs[-1]['step'],
                    valid_bdf2=np.arange(len(t_f)) >= 2,
                    note=('Source-free coarse Galerkin weak residual mass-inverted to a rate. '
                          'Causal BDF2 labels using 0.0025 saved time interval. '
                          'Not an exact physical memory term. Centered labels are offline-only.'))

                # Boundary localization at each valid time, using positive diagonal-lumped mass weights.
                for i in range(2, len(t_f)):
                    c, d = c_f[i], du_f[i]
                    total = weighted_norm(c, weights)**2
                    row = dict(case_id=case_id, split=splits[case_id], n_coarse=n, ell=ell,
                               time=float(t_f[i]), snapshot_dt=float(runs[-1]['step']),
                               closure_mass_norm=mass_norm(c, Mii),
                               tendency_mass_norm=mass_norm(d, Mii),
                               closure_over_tendency=mass_norm(c, Mii)/(mass_norm(d, Mii)+1e-12),
                               bdf_vs_centered_rel=mass_norm(c-c_center_f[i], Mii)/
                                   (mass_norm(c_center_f[i], Mii)+1e-12))
                    for name, mask in reg.items():
                        row[name+'_fraction_lumped'] = (weighted_norm(c, weights, mask)**2)/(total+1e-24)
                    region_rows.append(row)

                # Compare BDF2 labels at the same physical times. Reference: finest *saved* interval.
                # Start at t=0.02 so all three have a defined causal BDF2 stencil.
                for i0, t in enumerate(runs[0]['t']):
                    if i0 < 2:
                        continue
                    js = [int(round((t-r['t'][0])/r['step'])) for r in runs]
                    if any(j < 2 or j >= len(r['t']) or abs(r['t'][j]-t) > 1e-8 for j, r in zip(js, runs)):
                        raise ValueError(f'Cannot align causal BDF2 labels at t={t:g}.')
                    ref, refdu = computed[-1][1][js[-1]], computed[-1][3][js[-1]]
                    for tag, run, data, j in zip(['saved010','saved005','saved0025'], runs, computed, js):
                        c = data[1][j]
                        row = dict(case_id=case_id, split=splits[case_id], n_coarse=n, ell=ell,
                                   time=float(t), source=tag, snapshot_dt=float(run['step']),
                                   label_difference_mass_norm=mass_norm(c-ref, Mii),
                                   label_diff_over_reference_closure=mass_norm(c-ref, Mii)/
                                       (mass_norm(ref, Mii)+1e-12),
                                   label_diff_over_reference_tendency=mass_norm(c-ref, Mii)/
                                       (mass_norm(refdu, Mii)+1e-12))
                        for name, mask in reg.items():
                            row[name+'_diff_over_reference_closure_lumped'] = weighted_norm(
                                c-ref, weights, mask)/(weighted_norm(ref, weights, mask)+1e-12)
                        compare_rows.append(row)
                print(f'  N={n}, ell={ell:g}: refined labels saved', flush=True)

    df = pd.DataFrame(compare_rows)
    rf = pd.DataFrame(region_rows)
    df.to_csv(out/'cadence_comparison_by_time.csv', index=False)
    rf.to_csv(out/'refined_closure_by_time.csv', index=False)
    agg = df.groupby(['case_id', 'split', 'n_coarse', 'ell', 'source', 'snapshot_dt'], as_index=False).agg(
        mean_diff_over_closure=('label_diff_over_reference_closure','mean'),
        mean_diff_over_tendency=('label_diff_over_reference_tendency','mean'),
        mean_boundary_diff_over_closure=('boundary_diff_over_reference_closure_lumped','mean'),
        mean_interior_diff_over_closure=('interior_diff_over_reference_closure_lumped','mean'))
    agg.to_csv(out/'cadence_comparison_by_case.csv', index=False)
    reg_agg = rf.groupby(['case_id', 'split', 'n_coarse', 'ell'], as_index=False).agg(
        mean_closure_mass_norm=('closure_mass_norm','mean'),
        mean_boundary_fraction=('boundary_fraction_lumped','mean'),
        final_boundary_fraction=('boundary_fraction_lumped','last'),
        final_top_fraction=('top_fraction_lumped','last'),
        mean_bdf_vs_centered_rel=('bdf_vs_centered_rel','mean'))
    reg_agg.to_csv(out/'refined_closure_by_case.csv', index=False)
    # This is a screening diagnostic, NOT a pass/fail statement about physical memory.
    mid = agg[agg.source=='saved005']
    summary = {
        'cases_processed': args.cases,
        'coarse_meshes': ns,
        'filter_widths': ells,
        'saved_intervals': [float(r['step']) for r in runs],
        'source_free': True,
        'boundary_width': args.boundary_width,
        'boundary_energy_definition': 'fraction of squared mass-lumped nodal closure rates',
        'mean_saved005_vs_saved0025_rel_closure': float(mid.mean_diff_over_closure.mean()),
        'mean_saved005_vs_saved0025_rel_tendency': float(mid.mean_diff_over_tendency.mean()),
        'mean_final_boundary_fraction': float(reg_agg.final_boundary_fraction.mean()),
        'diagnostic_only': True,
        'warning': ('Saved 0.0025 BDF2 is a refined numerical label, not a certified exact '
                    'semidiscrete closure. This is a linear constant-coefficient problem and '
                    'provides no proof of a physical memory kernel.'),
        'next_decision': 'Check boundary/interior and label error before proceeding to a genuine memory benchmark.'
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Done: {out}', flush=True)
    print('Mean 0.005 vs 0.0025 closure difference:',
          f'{summary["mean_saved005_vs_saved0025_rel_closure"]:.3%}', flush=True)

if __name__ == '__main__':
    main()