from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description='Step 3C.0: plots and audit summary')
    p.add_argument('--indir', type=Path, default=Path('results/step3C0'))
    args = p.parse_args()
    folder = args.indir.expanduser().resolve()
    comp = pd.read_csv(folder/'cadence_comparison_by_case.csv')
    regions = pd.read_csv(folder/'refined_closure_by_time.csv')
    figs = folder/'figures'; figs.mkdir(exist_ok=True)

    # 1. Relative difference between saved BDF2 labels (identical FEM states at common times).
    subset = comp[comp.source != 'saved0025']
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5), sharey=True)
    for ax, n in zip(axes, sorted(subset.n_coarse.unique())):
        part = subset[subset.n_coarse==n]
        for tag, name in [('saved010','saved 0.010'),('saved005','saved 0.005')]:
            sub = part[part.source==tag]
            g = sub.groupby('ell', as_index=False).agg(y=('mean_diff_over_closure','mean'),
                                                        sd=('mean_diff_over_closure','std'))
            ax.errorbar(g.ell, g.y, yerr=g.sd.fillna(0), fmt='o-', capsize=3,label=name)
        ax.set(title=f'Coarse N={n}', xlabel='Physical filter width ell')
        ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('Label difference / finest closure (FEM mass norm)')
    fig.suptitle('Causal BDF2 sensitivity to saved snapshot interval')
    fig.tight_layout();fig.savefig(figs/'01_label_convergence.png',dpi=170);plt.close(fig)

    # 2. Distinguish relative error in small closure from error relative to full tendency.
    finer = comp[comp.source=='saved005']
    g = finer.groupby(['n_coarse','ell'],as_index=False).agg(
        closure=('mean_diff_over_closure','mean'),
        tendency=('mean_diff_over_tendency','mean'))
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for n, sub in g.groupby('n_coarse'):
        sub=sub.sort_values('ell')
        axes[0].plot(sub.ell,sub.closure,'o-',label=f'N={n}')
        axes[1].plot(sub.ell,sub.tendency,'o-',label=f'N={n}')
    axes[0].set(title='Relative to closure',ylabel='Relative label difference')
    axes[1].set(title='Relative to coarse-state tendency',ylabel='Relative tendency difference')
    for ax in axes:
        ax.set_xlabel('Physical filter width ell'); ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Saved 0.005 labels versus 0.0025 reference')
    fig.tight_layout();fig.savefig(figs/'02_closure_vs_tendency.png',dpi=170);plt.close(fig)

    # 3. Boundary vs interior localization of refined labels, with time.
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    available_n = sorted(regions.n_coarse.unique())
    plot_n = 64 if 64 in available_n else int(available_n[-1])
    case0 = regions[(regions.case_id==0) & (regions.n_coarse==plot_n)]
    for ax,col,title in [(axes[0],'boundary_fraction_lumped','All boundary strips'),
                         (axes[1],'top_fraction_lumped','Top boundary strip')]:
        for ell, sub in case0.groupby('ell'):
            sub=sub.sort_values('time')
            ax.plot(sub.time,100*sub[col],label=f'ell={ell:g}')
        ax.set(title=title,xlabel='Physical time',ylim=(0,105));ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('Share of squared lumped-mass closure amplitude (%)')
    fig.suptitle(f'Case 000, N={plot_n}: where is the refined closure concentrated?')
    fig.tight_layout();fig.savefig(figs/'03_boundary_over_time.png',dpi=170);plt.close(fig)

    # 4. Show concentration and refined closure maps at early, middle, late times.
    path=folder/'case_000'/'coarse_n64_ell0.04_refined.npz'
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            u=np.asarray(z['u_coarse'])
            c=np.asarray(z['closure_rate_bdf2_CAUSAL'])
            t=np.asarray(z['t'])
        times=[0.02,0.10,0.20]
        inds=[int(np.argmin(abs(t-v))) for v in times]
        limit = np.nanpercentile(np.abs(c[np.asarray(inds)]),99.5)
        limit=max(float(limit), 1e-10)
        fig,ax=plt.subplots(2,len(inds),figsize=(12,7),constrained_layout=True)
        for j,k in enumerate(inds):
            ax[0,j].imshow(u[k],origin='lower',extent=[0,1,0,1],cmap='viridis',vmin=0)
            ax[1,j].imshow(c[k],origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax[0,j].set_title(f'Coarse concentration, t={t[k]:g}')
            ax[1,j].set_title(f'Refined causal closure, t={t[k]:g}')
            for a in [ax[0,j],ax[1,j]]:
                a.axhline(.1,color='white',lw=.8,ls='--',alpha=.8)
                a.axhline(.9,color='white',lw=.8,ls='--',alpha=.8)
                a.axvline(.1,color='white',lw=.8,ls='--',alpha=.8)
                a.axvline(.9,color='white',lw=.8,ls='--',alpha=.8)
                a.set(xlabel='x',ylabel='y')
        fig.savefig(figs/'04_refined_closure_maps.png',dpi=160);plt.close(fig)
    print('Figures saved to:',figs)

if __name__=='__main__':
    main()