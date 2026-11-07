from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def boundary_masks(n: int, width: float):
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing='xy')
    d = np.minimum.reduce((X, 1-X, Y, 1-Y))
    return d <= width + 1e-12, Y >= 1 - width - 1e-12, d


def frac_energy(field: np.ndarray, mask: np.ndarray) -> float:
    f = np.asarray(field, dtype=np.float64)
    den = np.sum(f*f)
    if den <= 1e-30:
        return float('nan')
    return float(np.sum(f[mask]**2) / den)


def rnorm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64).ravel()))


def audit_one(path: Path, width: float):
    with np.load(path, allow_pickle=False) as z:
        needed = {'u_coarse', 'closure_rate_bdf2_CAUSAL',
                  'closure_rate_centered_OFFLINE_ONLY', 'valid_bdf2',
                  't', 'n_coarse', 'ell', 'case_id', 'split'}
        missing = needed - set(z.files)
        if missing:
            raise ValueError(f'{path}: missing fields {sorted(missing)}')
        u = np.asarray(z['u_coarse'], dtype=np.float64)
        cb = np.asarray(z['closure_rate_bdf2_CAUSAL'], dtype=np.float64)
        cc = np.asarray(z['closure_rate_centered_OFFLINE_ONLY'], dtype=np.float64)
        valid = np.asarray(z['valid_bdf2'], dtype=bool)
        t = np.asarray(z['t'], dtype=float)
        n = int(z['n_coarse'])
        ell = float(z['ell'])
        cid = int(z['case_id'])
        split = str(z['split'])
    if u.shape != cb.shape or u.shape != cc.shape or u.shape[1:] != (n+1,n+1):
        raise ValueError(f'{path}: field shapes disagree')
    if len(t) != len(valid) or not np.all(np.diff(t) > 0):
        raise ValueError(f'{path}: unexpected timestamps')
    shell, top, dist = boundary_masks(n, width)
    rows = []
    for k in np.flatnonzero(valid):
        a, b = cb[k], cc[k]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'{path}: nonfinite valid closure at time index {k}')
        idx = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        du_bdf = (3*u[k] - 4*u[k-1] + u[k-2])/(2*(t[k]-t[k-1]))
        # Use interior nodes only for derivative scale: imposed Dirichlet nodes remain zero.
        sl = np.s_[1:-1,1:-1]
        db = a[sl] - b[sl]
        rows.append(dict(
            case_id=cid, split=split, n_coarse=n, ell=ell,
            time=float(t[k]), valid_bdf2=True,
            boundary_shell_fraction_nodal_sq=frac_energy(a, shell),
            top_strip_fraction_nodal_sq=frac_energy(a, top),
            interior_fraction_nodal_sq=frac_energy(a, ~shell),
            closure_nodal_rms=float(np.sqrt(np.mean(a[sl]**2))),
            bdf_vs_centered_rel_nodal=rnorm(db)/(rnorm(b[sl])+1e-12),
            derivative_delta_over_tendency_nodal=rnorm(db)/(rnorm(du_bdf[sl])+1e-12),
            peak_abs_closure=float(np.max(np.abs(a))),
            peak_x=float(idx[1]/n), peak_y=float(idx[0]/n),
            state_max=float(np.max(u[k])),
        ))
    return rows


def summary_stats(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        'n_times':len(g),
        'boundary_fraction_mean':g.boundary_shell_fraction_nodal_sq.mean(),
        'boundary_fraction_final':g.boundary_shell_fraction_nodal_sq.iloc[-1],
        'top_fraction_mean':g.top_strip_fraction_nodal_sq.mean(),
        'top_fraction_final':g.top_strip_fraction_nodal_sq.iloc[-1],
        'interior_fraction_mean':g.interior_fraction_nodal_sq.mean(),
        'bdf_centered_discrepancy_mean':g.bdf_vs_centered_rel_nodal.mean(),
        'derivative_delta_over_tendency_mean':g.derivative_delta_over_tendency_nodal.mean(),
        'peak_abs_closure_final':g.peak_abs_closure.iloc[-1],
        'peak_x_final':g.peak_x.iloc[-1],
        'peak_y_final':g.peak_y.iloc[-1],
    })


def plot_by_filter(cases: pd.DataFrame, metric: str, ylabel: str, title: str, dst: Path):
    fig, ax = plt.subplots(figsize=(8.4,5.3))
    for (n,split), g in cases.groupby(['n_coarse','split'], sort=True):
        agg = g.groupby('ell')[metric].agg(['mean','std']).sort_index()
        ax.errorbar(agg.index.to_numpy(), agg['mean'].to_numpy(),
                    yerr=agg['std'].fillna(0).to_numpy(), capsize=3,
                    linewidth=1.8, marker='o', label=f'N={n}, {split}')
    ax.set(title=title, xlabel='Physical Gaussian filter width ell', ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(dst, dpi=180)
    plt.close(fig)


def plot_exemplar(input_root: Path, manifest: pd.DataFrame, out: Path, width: float):
    ex = manifest[(manifest.case_id==manifest.case_id.min())]
    if ex.empty:
        return
    row = ex.sort_values(['n_coarse','ell']).iloc[-1]
    with np.load(input_root/str(row.path), allow_pickle=False) as z:
        u = np.asarray(z['u_coarse'][-1],float)
        closure = np.asarray(z['closure_rate_bdf2_CAUSAL'][-1],float)
        n = int(z['n_coarse'])
        ell = float(z['ell'])
    shell, top, _ = boundary_masks(n, width)
    remainder = closure.copy(); remainder[shell] = np.nan
    v = max(float(np.max(np.abs(closure))), 1e-12)
    fig, axs = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    im1=axs[0].imshow(u,origin='lower',extent=(0,1,0,1),cmap='viridis')
    fig.colorbar(im1,ax=axs[0],shrink=.8)
    im2=axs[1].imshow(closure,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-v,vmax=v)
    fig.colorbar(im2,ax=axs[1],shrink=.8)
    # Interior uses same color range so the audience can see how much smaller it is.
    im3=axs[2].imshow(remainder,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-v,vmax=v)
    fig.colorbar(im3,ax=axs[2],shrink=.8)
    axs[0].set_title('Coarse state')
    axs[1].set_title('Full closure rate')
    axs[2].set_title(f'Closure outside boundary strip d <= {width:g}')
    for ax in axs: ax.set(xlabel='x',ylabel='y')
    fig.suptitle(f'Case {int(row.case_id):03d}, N={n}, ell={ell:g}, final snapshot')
    fig.savefig(out/'exemplar_closure_boundary_decomposition.png',dpi=170)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='SMTNO Step 3B.5: boundary localization and temporal-label audit')
    ap.add_argument('--indir',type=Path,default=Path('results/step3B4'))
    ap.add_argument('--outdir',type=Path,default=Path('results/step3B5'))
    ap.add_argument('--boundary-width',type=float,default=.10)
    ap.add_argument('--max-cases',type=int,default=None, help='Optional pilot: evaluate first K cases')
    args=ap.parse_args()
    root=args.indir.expanduser().resolve();out=args.outdir.expanduser().resolve()
    if not (0 < args.boundary_width < .5):
        raise ValueError('--boundary-width must be between 0 and .5')
    manifest_path=root/'dataset_manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing {manifest_path}; run full Step 3B.4 first')
    manifest=pd.read_csv(manifest_path)
    required={'case_id','split','n_coarse','ell','path'}
    if not required.issubset(manifest):
        raise ValueError(f'Missing manifest columns {required-set(manifest)}')
    if manifest.duplicated(subset=['case_id','n_coarse','ell']).any():
        raise ValueError('Duplicate case / resolution / filter combination in manifest')
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError('--max-cases must be positive')
        keep = set(sorted(manifest.case_id.unique())[:args.max_cases])
        manifest = manifest[manifest.case_id.isin(keep)].copy()
    all_rows=[]
    for i,row in enumerate(manifest.itertuples(index=False), start=1):
        p=root/str(row.path)
        if not p.is_file():
            raise FileNotFoundError(p)
        rows=audit_one(p,args.boundary_width)
        if (int(row.case_id),int(row.n_coarse))!=(rows[0]['case_id'],rows[0]['n_coarse']) or not np.isclose(float(row.ell),rows[0]['ell']):
            raise ValueError(f'Inconsistent manifest entry: {p}')
        all_rows.extend(rows)
        if i%20 == 0 or i == len(manifest):
            print(f'Inspected {i}/{len(manifest)} combinations',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    snap=pd.DataFrame(all_rows).sort_values(['case_id','n_coarse','ell','time'])
    snap.to_csv(out/'boundary_audit_by_time.csv',index=False)
    group_cols=['case_id','split','n_coarse','ell']
    cases=pd.DataFrame([dict(zip(group_cols,keys), **summary_stats(g).to_dict())
                        for keys,g in snap.groupby(group_cols,sort=True)])
    cases.to_csv(out/'boundary_audit_by_case.csv',index=False)
    group=(cases.groupby(['split','n_coarse','ell'])
           .agg(n_cases=('case_id','nunique'),
                shell_fraction_mean=('boundary_fraction_mean','mean'),
                shell_fraction_sd=('boundary_fraction_mean','std'),
                shell_fraction_final_mean=('boundary_fraction_final','mean'),
                top_fraction_final_mean=('top_fraction_final','mean'),
                temporal_discrepancy_mean=('bdf_centered_discrepancy_mean','mean'))
           .reset_index())
    group.to_csv(out/'boundary_audit_group_summary.csv',index=False)
    plot_by_filter(cases,'boundary_fraction_final',
                   'Fraction of nodal squared closure in boundary strip',
                   'Final-time closure localization versus filter width',
                   out/'boundary_fraction_by_filter.png')
    plot_by_filter(cases,'bdf_centered_discrepancy_mean',
                   'BDF2 versus centered closure discrepancy (nodal ratio)',
                   'Temporal-label sensitivity versus filter width',
                   out/'temporal_label_sensitivity.png')
    plot_exemplar(root,manifest,out,args.boundary_width)
    # Time evolution of the boundary contribution for the first case on largest mesh/ell
    cmin=int(snap.case_id.min());nmax=int(snap.n_coarse.max());emax=float(snap.ell.max())
    sub=snap[(snap.case_id==cmin)&(snap.n_coarse==nmax)&np.isclose(snap.ell,emax)]
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.plot(sub.time,sub.boundary_shell_fraction_nodal_sq,marker='.',label='All boundary sides')
    ax.plot(sub.time,sub.top_strip_fraction_nodal_sq,marker='.',label='Top boundary strip')
    ax.set(xlabel='Physical time',ylabel='Fraction of nodal squared closure',
           title=f'Case {cmin:03d}, N={nmax}, ell={emax:g}: localization over time',ylim=(0,1.03))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(out/'example_boundary_fraction_vs_time.png',dpi=180);plt.close(fig)
    print('Number of inspected cases:',cases.case_id.nunique())
    print('Number of inspected combinations:',len(cases))
    print('Boundary width:',args.boundary_width)
    print('Outputs:',out)
    print('Interpretation: nodal squared-amplitude localization is NOT FEM mass energy or identified Mori-Zwanzig memory.')


if __name__=='__main__':
    main()