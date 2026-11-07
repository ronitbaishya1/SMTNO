from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description='Step 3C.1 memory necessity report')
    p.add_argument('--indir',default='results/step3C1')
    args=p.parse_args()
    root=Path(args.indir)
    df=pd.read_csv(root/'forecast_metrics.csv')
    history=pd.read_csv(root/'training_history.csv')
    figs=root/'figures';figs.mkdir(exist_ok=True,parents=True)
    group=df.groupby(['split','n','ell','horizon','model'],as_index=False).agg(
        rel_state_error=('rel_state_error','mean'),
        rel_increment_error=('rel_increment_error','mean'),
        error_case_std=('rel_state_error','std'),
        n_windows=('rel_state_error','size'))
    group.to_csv(root/'forecast_summary.csv',index=False)
    for split in ['val','test']:
        for n in sorted(df.n.unique()):
            tab=group[(group.split==split)&(group.n==n)&(group.horizon==1)]
            if tab.empty:continue
            fig,ax=plt.subplots(figsize=(8,4.5))
            for m in ['current','history','linear_extrapolation','coarse_FEM_BE']:
                part=tab[tab.model==m].sort_values('ell')
                ax.plot(part.ell,part.rel_increment_error,marker='o',label=m)
            ax.set(xlabel='Physical filtering width ell',ylabel='Relative one-step increment error',
                   title=f'{split} | N={n}, one-step prediction')
            ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
            fig.savefig(figs/f'one_step_increment_{split}_n{n}.png',dpi=160);plt.close(fig)
    tab=group[group.split=='test']
    for n in sorted(tab.n.unique()):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for m in ['current','history','linear_extrapolation','coarse_FEM_BE']:
            q=tab[(tab.n==n)&(tab.model==m)].groupby('horizon').rel_state_error.mean().sort_index()
            ax.plot(q.index,q.values,marker='o',label=m)
        ax.set(xlabel='Autonomous prediction horizon (saved steps)',ylabel='Relative state error',
               title=f'Independent test cases, N={n}')
        ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(figs/f'rollout_test_n{n}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for mode,d in history.groupby('model'):
        ax.semilogy(d.epoch,np.maximum(d.val_mse,1e-12),label=f'{mode}: validation')
    ax.set(xlabel='Epoch',ylabel='Normalized validation MSE',title='Held-out validation during training')
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(figs/'validation_curves.png',dpi=160);plt.close(fig)
    cols=['split','n','ell','horizon'];pivot=group.pivot_table(index=cols,columns='model',values='rel_state_error').reset_index()
    pivot['history_gain_pct']=100*(pivot.current-pivot.history)/np.maximum(pivot.current,1e-12)
    pivot['history_erasure_penalty_pct']=100*(pivot.history_erased-pivot.history)/np.maximum(pivot.history,1e-12)
    pivot.to_csv(root/'memory_comparison.csv',index=False)
    test=pivot[pivot.split=='test']
    fig,ax=plt.subplots(figsize=(8,4.5))
    for n,grp in test.groupby('n'):
        for h in sorted(grp.horizon.unique()):
            v=grp[grp.horizon==h].sort_values('ell')
            ax.plot(v.ell,v.history_gain_pct,marker='o',label=f'N={n}, h={h}')
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(xlabel='Physical filtering width ell',ylabel='History gain over current FNO (%)',
           title='Held-out test: positive means history helps')
    ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.3);fig.tight_layout()
    fig.savefig(figs/'history_gain_test.png',dpi=160);plt.close(fig)
    print('Results:',root)
    print('\nMean test errors across filtering widths (interpret with per-scale plots):')
    s=group[group.split=='test'].groupby(['n','horizon','model']).agg(
        mean_state_error=('rel_state_error','mean'),mean_increment_error=('rel_increment_error','mean'))
    print(s.round(4).to_string())
    print('\nHistory gain by scale, positive = improvement:')
    print(test[['n','ell','horizon','history_gain_pct','history_erasure_penalty_pct']].round(2).to_string(index=False))
    print('\nNOTE: this linear PDE is a NEGATIVE CONTROL. Improvement does not prove physical MZ memory.')

if __name__=='__main__':main()