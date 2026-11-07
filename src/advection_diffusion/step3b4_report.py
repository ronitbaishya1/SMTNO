from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_summary(df, column, title, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for (n, split), g in df.groupby(['n_coarse', 'split']):
        data = g.groupby('ell')[column].agg(['mean', 'std','count']).sort_index()
        x = data.index.to_numpy(dtype=float)
        means = data['mean'].to_numpy(dtype=float)
        stds = data['std'].fillna(0).to_numpy(dtype=float)
        ax.errorbar(x, means, yerr=stds, marker='o', capsize=3,
                    label=f'N={n}, {split} (mean ± case SD)')
    ax.set(xlabel='Physical Gaussian filter width ell', ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Step 3B.4: plot source-free closure ensemble diagnostics')
    ap.add_argument('--indir',type=Path,default=Path('results/step3B4'))
    args=ap.parse_args()
    path=args.indir.expanduser().resolve()
    mpath=path/'dataset_manifest.csv'
    metpath=path/'closure_metrics.csv'
    if not mpath.is_file() or not metpath.is_file():
        raise FileNotFoundError('First run step3b4_build.py, then rerun this report script')
    d=pd.read_csv(mpath)
    detailed=pd.read_csv(metpath)
    if len(d)==0 or not set(['case_id','split','n_coarse','ell','path']).issubset(d):
        raise ValueError('Empty or malformed dataset manifest')
    missing=[p for p in d.path if not (path/p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} combination files missing, example {missing[0]}')
    figs=path/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    for col,title,ylabel,filename in [
        ('mean_closure_mass_norm','Source-free weak-form closure versus filtering scale',
         'Mean FEM mass norm of closure rate','closure_vs_filter.png'),
        ('mean_unclosed_tendency_error_rel','Relative correction to the unclosed coarse tendency',
         'Mean ||closure||_M / ||du/dt||_M','unclosed_tendency_by_filter.png'),
        ('mean_bdf2_centered_discrepancy_rel','Temporal differentiation sensitivity',
         'Mean relative BDF2 vs centered closure difference','temporal_derivative_audit.png'),
        ('final_projection_error_rel','P1 projection accuracy at final time',
         'Relative nodal-grid projection error','projection_error_by_filter.png'),
    ]:
        plot_summary(d,col,title,ylabel,figs/filename)
    # Time-trace representative case, labels are always explicitly physical time.
    case=int(d.case_id.min())
    subset=detailed[(detailed.case_id==case)&(detailed.valid_bdf2.astype(bool))]
    fig,ax=plt.subplots(figsize=(7.6,4.6))
    for (n,ell),g in subset.groupby(['n_coarse','ell']):
        ax.plot(g.time,g.closure_mass_norm,label=f'N={n}, ell={ell:g}')
    ax.set(xlabel='Physical time',ylabel='FEM mass norm of closure rate',
           title=f'Case {case:03d}: closure versus time (causal BDF2)')
    ax.grid(alpha=.3);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(figs/'example_case_closure_time.png',dpi=160);plt.close(fig)
    # Spatial profiles of the closure at the last time and the largest available width.
    n=int(d.n_coarse.max());ell=float(d.ell.max())
    entry=d[(d.case_id==case)&(d.n_coarse==n)&np.isclose(d.ell,ell)].iloc[0]
    with np.load(path/entry.path,allow_pickle=False) as z:
        u=z['u_coarse'][-1]
        closure=z['closure_rate_bdf2_CAUSAL'][-1]
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    a=axes[0].imshow(u,origin='lower',extent=[0,1,0,1],aspect='equal')
    axes[0].set(title='Projected coarse state',xlabel='x',ylabel='y')
    fig.colorbar(a,ax=axes[0],shrink=.8)
    lim=max(float(np.max(np.abs(closure))),1e-12)
    b=axes[1].imshow(closure,origin='lower',extent=[0,1,0,1],aspect='equal',
                     cmap='RdBu_r',vmin=-lim,vmax=lim)
    axes[1].set(title='Source-free BDF2 closure rate',xlabel='x',ylabel='y')
    fig.colorbar(b,ax=axes[1],shrink=.8)
    fig.suptitle(f'Case {case:03d}: N={n}, ell={ell:g}, final time')
    fig.savefig(figs/'example_case_spatial.png',dpi=160);plt.close(fig)
    print('Figures saved:',figs)
    print('Number of completed cases:',d.case_id.nunique())
    print('Source-free combinations:',len(d))
    print('Cases per split:',d.groupby('split').case_id.nunique().to_dict())
    print('CAUTION: these are numerical closure labels, not an identified memory kernel.')


if __name__=='__main__':
    main()