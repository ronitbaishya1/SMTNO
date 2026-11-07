from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import the very same operators used in Steps 3B.1 and 3B.2.
from step3b1_filter_project import (
    dirichlet_gaussian, prepare_projection, project_field_from_fine,
    p1_reconstruction_at_fine, quad_integral, relative_rms,
)
from step3b2_weak_closure import fem_operators, mass_norm

VERSION = '3B4_source_free_v2'


def parse_csv_numbers(raw, caster):
    vals = [caster(v.strip()) for v in raw.split(',') if v.strip()]
    if not vals or len(set(vals)) != len(vals):
        raise ValueError('Expected a nonempty list without repeated values')
    return vals


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_case(path, case_id):
    if not path.is_file():
        raise FileNotFoundError(f'Missing fine FEM case {case_id:03d}: {path}')
    with np.load(path, allow_pickle=False) as z:
        required = {'u', 't', 'x', 'y', 'eps', 'source_zero', 'case_id'}
        if not required.issubset(set(z.files)):
            raise ValueError(f'Case {case_id:03d} missing keys: {required-set(z.files)}')
        u = np.asarray(z['u'], dtype=np.float64)
        t = np.asarray(z['t'], dtype=np.float64)
        x, y = np.asarray(z['x'], dtype=np.float64), np.asarray(z['y'], dtype=np.float64)
        eps = float(z['eps'])
        source_zero = bool(z['source_zero'])
        saved_id = int(z['case_id'])
    if saved_id != case_id or not source_zero:
        raise ValueError(f'case_{case_id:03d}: wrong case_id or not a source-free ensemble')
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError(f'Invalid epsilon in case_{case_id:03d}')
    if u.ndim != 3 or u.shape[1] != u.shape[2] or u.shape[0] != len(t):
        raise ValueError(f'Invalid [time, y, x] shape in case_{case_id:03d}')
    if not np.all(np.isfinite(u)):
        raise ValueError(f'Nonfinite solution in case_{case_id:03d}')
    nf = u.shape[1] - 1
    if len(x) != nf + 1 or len(y) != nf + 1:
        raise ValueError(f'Coordinate shape mismatch in case_{case_id:03d}')
    reference_grid = np.linspace(0, 1, nf+1)
    if not np.allclose(x, reference_grid, atol=1e-9) or not np.allclose(y, reference_grid, atol=1e-9):
        raise ValueError('Source should use a uniform [0,1]^2 grid')
    if len(t) < 5 or not np.allclose(np.diff(t), t[1]-t[0], rtol=1e-6, atol=1e-8):
        raise ValueError('Need >=5 equally spaced snapshots for BDF2 diagnosis')
    if np.max(np.abs(np.concatenate((u[:, 0, :].ravel(), u[:, -1, :].ravel(),
                                     u[:, :, 0].ravel(), u[:, :, -1].ravel())))) > 5e-5:
        raise ValueError('Zero Dirichlet boundary assumption violated')
    return u, t, x, eps


def extend_interior(vals, interior, n):
    # vals shape [time, n_internal], edges remain zero.
    result = np.zeros((len(vals), (n+1)**2), dtype=np.float32)
    result[:, interior] = vals.astype(np.float32)
    # Invalid startup entries must remain invalid even at boundary nodes.
    invalid = np.all(np.isnan(vals), axis=1)
    result[invalid, :] = np.nan
    return result.reshape(len(vals), n+1, n+1)


def calculate_coarse(filtered, times, n_coarse, eps, projection, operators):
    M, A, interior, lu, *_ = operators
    Mii = M[interior][:, interior].tocsr()
    Aii = A[interior][:, interior].tocsr()
    projected = np.stack([project_field_from_fine(frame, projection)
                          for frame in filtered], axis=0)
    u_interior = projected.reshape(len(times), -1)[:, interior]
    dt = float(times[1]-times[0])

    # Causal BDF2: only t >= 2*dt has a valid 3-point backward derivative.
    du_bdf = np.full(u_interior.shape, np.nan, dtype=np.float64)
    du_bdf[2:] = (3*u_interior[2:] - 4*u_interior[1:-1] + u_interior[:-2])/(2*dt)
    # Noncausal centered differences solely for independent offline diagnostics.
    du_ctr = np.gradient(u_interior, dt, axis=0, edge_order=2)

    closure_bdf = np.full_like(du_bdf, np.nan)
    weak_bdf = np.full_like(du_bdf, np.nan)
    for i in range(2, len(times)):
        weak_bdf[i] = Mii @ du_bdf[i] + Aii @ u_interior[i]
        closure_bdf[i] = lu.solve(weak_bdf[i])
    closure_ctr = np.empty_like(du_ctr)
    for i in range(len(times)):
        closure_ctr[i] = lu.solve(Mii @ du_ctr[i] + Aii @ u_interior[i])

    details = []
    nf = filtered.shape[1]-1
    for i, ti in enumerate(times):
        recon = p1_reconstruction_at_fine(projected[i], nf)
        row = dict(time=float(ti), valid_bdf2=bool(i>=2),
                   projection_error_rel=relative_rms(recon, filtered[i]),
                   filtered_integral=quad_integral(filtered[i]),
                   projected_integral=quad_integral(projected[i]))
        if i >= 2:
            c_norm = mass_norm(closure_bdf[i], Mii)
            d_norm = mass_norm(du_bdf[i], Mii)
            row.update(closure_mass_norm=c_norm,
                       relative_unclosed_tendency_error=c_norm/(d_norm+1e-12),
                       bdf2_centered_closure_discrepancy_rel=
                           mass_norm(closure_bdf[i]-closure_ctr[i], Mii)/(mass_norm(closure_ctr[i], Mii)+1e-12),
                       projected_state_mass_norm=mass_norm(u_interior[i], Mii),
                       tendency_mass_norm=d_norm)
        else:
            row.update(closure_mass_norm=np.nan,
                       relative_unclosed_tendency_error=np.nan,
                       bdf2_centered_closure_discrepancy_rel=np.nan,
                       projected_state_mass_norm=np.nan,
                       tendency_mass_norm=np.nan)
        details.append(row)
    return projected, closure_bdf, closure_ctr, weak_bdf, details, interior


def main():
    parser = argparse.ArgumentParser(description='Step 3B.4: build source-free coarse FEM weak closure ensemble')
    parser.add_argument('--indir', type=Path, default=Path('results/step3B3'))
    parser.add_argument('--outdir', type=Path, default=Path('results/step3B4'))
    parser.add_argument('--cases', type=int, default=2,
                        help='Process first K cases, e.g. 2 for pilot, 30 for full build')
    parser.add_argument('--coarse', default='32,64')
    parser.add_argument('--ells', default='0.01,0.02,0.04')
    parser.add_argument('--force', action='store_true', help='Regenerate even when validated cached files exist')
    args = parser.parse_args()
    indir = args.indir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    manifest_path = indir/'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing manifest: {manifest_path}')
    original_manifest = pd.read_csv(manifest_path)
    if not {'case_id','split'}.issubset(original_manifest):
        raise ValueError('Expected case_id and split columns in Step 3B.3 manifest')
    if original_manifest.case_id.duplicated().any():
        raise ValueError('Duplicate case IDs in manifest')
    original_manifest = original_manifest.sort_values('case_id')
    total = len(original_manifest)
    if args.cases < 1 or args.cases > total:
        raise ValueError(f'--cases must be in 1..{total}')
    selected = original_manifest.iloc[:args.cases]
    ns = parse_csv_numbers(args.coarse, int)
    ells = parse_csv_numbers(args.ells, float)
    if min(ns) < 4 or min(ells) <= 0:
        raise ValueError('Invalid coarse resolution or filter width')
    outdir.mkdir(parents=True, exist_ok=True)

    # These are assembled once per coarse mesh and once per diffusivity.
    cached = {}
    summary_rows = []
    all_metrics = []
    for row in selected.itertuples(index=False):
        case_id, split = int(row.case_id), str(row.split)
        if split not in {'train','val','test'}:
            raise ValueError(f'Unexpected split {split!r} for case {case_id}')
        source = indir/f'case_{case_id:03d}'/'field_snapshots.npz'
        case_folder = outdir/f'case_{case_id:03d}'
        case_folder.mkdir(parents=True,exist_ok=True)
        source_hash = sha256_file(source)
        expected = [case_folder/f'coarse_n{n}_ell{ell:g}.npz' for n in ns for ell in ells]
        config_file = case_folder/'build_config.json'
        signature = dict(version=VERSION, case_id=case_id, source_sha256=source_hash,
                         source_zero=True, coarse=ns, ells=ells)
        cached_ok = (not args.force and config_file.is_file()
                     and json.loads(config_file.read_text()) == signature
                     and all(p.is_file() for p in expected)
                     and (case_folder/'case_metrics.csv').is_file())
        if cached_ok:
            print(f'Cache hit: case_{case_id:03d}',flush=True)
        else:
            fine, times, grid_x, eps = load_case(source, case_id)
            nf = len(grid_x)-1
            for n in ns:
                if n >= nf or nf % n:
                    raise ValueError(f'Fine N={nf} must be divisible by coarse N={n} and finer')
            for n in ns:
                key = (n, eps)
                if key not in cached:
                    projection = prepare_projection(n)
                    cached[key] = (projection, fem_operators(n, eps, projection))
            case_metrics = []
            filtered_dict = {ell: np.stack([dirichlet_gaussian(frame, ell) for frame in fine])
                             for ell in ells}
            for ell in ells:
                f = filtered_dict[ell]
                np.savez_compressed(case_folder/f'filtered_fine_ell{ell:g}.npz',
                                    u_filtered_fine=f.astype(np.float32), t=times, x=grid_x,
                                    y=grid_x, ell=ell, case_id=case_id,
                                    note='Diagnostic fine-grid filtered state, not a unique coarse-to-fine reconstruction')
            for n in ns:
                projection, operators = cached[(n,eps)]
                for ell in ells:
                    filtered = filtered_dict[ell]
                    p, cbdf, ccenter, weakbdf, metrics, interior = calculate_coarse(
                        filtered, times, n, eps, projection, operators)
                    final_path = case_folder/f'coarse_n{n}_ell{ell:g}.npz'
                    np.savez_compressed(final_path,
                        u_coarse=p.astype(np.float32),
                        closure_rate_bdf2_CAUSAL=extend_interior(cbdf, interior,n),
                        closure_rate_centered_OFFLINE_ONLY=extend_interior(ccenter,interior,n),
                        closure_weak_bdf2_CAUSAL=extend_interior(weakbdf,interior,n),
                        valid_bdf2=np.arange(len(times))>=2,
                        t=times, x=np.linspace(0,1,n+1), y=np.linspace(0,1,n+1),
                        ell=ell, n_coarse=n, n_fine=nf, epsilon=eps,
                        case_id=case_id, split=split, source_zero=True,
                        note=('C=M u_t + A u (f=0), P1 Galerkin coarse; mass-inverted closure '
                              'rate, causal BDF2 on coarse filtered states at saved dt. '
                              'First two time entries are NaN/invalid. Centered derivative OFFLINE ONLY. '
                              'Includes projection and numerical differentiation effects; NOT physical memory.'))
                    for m in metrics:
                        m.update(case_id=case_id, split=split, n_coarse=n,
                                 ell=ell, n_fine=nf, epsilon=eps,
                                 snapshot_dt=float(times[1]-times[0]))
                    case_metrics.extend(metrics)
                    print(f'  case_{case_id:03d} N={n} ell={ell:g}',flush=True)
            pd.DataFrame(case_metrics).to_csv(case_folder/'case_metrics.csv',index=False)
            config_file.write_text(json.dumps(signature,indent=2))
        metric_rows = pd.read_csv(case_folder/'case_metrics.csv')
        all_metrics.append(metric_rows)
        for n in ns:
            for ell in ells:
                file_path = case_folder/f'coarse_n{n}_ell{ell:g}.npz'
                subset = metric_rows[(metric_rows.n_coarse==n)&np.isclose(metric_rows.ell,ell)]
                valid = subset[subset.valid_bdf2.astype(bool)]
                if len(valid)==0:
                    raise ValueError(f'No valid snapshots in {file_path}')
                summary_rows.append(dict(case_id=case_id, split=split,
                    n_coarse=n, ell=ell, path=str(file_path.relative_to(outdir)),
                    valid_snapshots=len(valid),
                    mean_closure_mass_norm=float(valid.closure_mass_norm.mean()),
                    mean_unclosed_tendency_error_rel=float(valid.relative_unclosed_tendency_error.mean()),
                    mean_bdf2_centered_discrepancy_rel=
                        float(valid.bdf2_centered_closure_discrepancy_rel.mean()),
                    final_projection_error_rel=float(subset.projection_error_rel.iloc[-1]),
                    temporal_derivative_audit_gt_10pct=bool(
                        float(valid.bdf2_centered_closure_discrepancy_rel.mean()) > .10)))
        print(f'Completed case {case_id:03d} ({split})',flush=True)

    df = pd.DataFrame(summary_rows).sort_values(['case_id','n_coarse','ell'])
    df.to_csv(outdir/'dataset_manifest.csv',index=False)
    pd.concat(all_metrics,ignore_index=True).to_csv(outdir/'closure_metrics.csv',index=False)
    overview = dict(version=VERSION, requested_cases=args.cases,
                    completed_cases=int(df.case_id.nunique()),
                    total_source_cases=total, produced_combinations=len(df),
                    source='FreeFEM++ P1/SUPG trajectories (source-free); coarse P1 Galerkin closure',
                    temporal_derivative_sensitivity_warning_count=int(df.temporal_derivative_audit_gt_10pct.sum()),
                    reference_warning=('Numerical weak residual includes spatial filtering, projection, '
                                       'FEM mismatch and saved-snapshot derivative error; it is not '
                                       'identified Mori-Zwanzig memory.'),
                    first_two_snapshots_invalid=True,
                    field_shape='[time,y,x]',
                    split_counts=df.groupby('split').case_id.nunique().to_dict())
    (outdir/'summary.json').write_text(json.dumps(overview,indent=2))
    print(f'Finished {overview["completed_cases"]} cases, {len(df)} coarse/filter combinations.',flush=True)
    print('Outputs:',outdir,flush=True)


if __name__ == '__main__':
    main()