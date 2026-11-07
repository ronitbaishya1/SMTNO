from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(root.glob('example*')):
        cfg_path, metrics_path = folder / 'run_config.json', folder / 'metrics.csv'
        if not (cfg_path.exists() and metrics_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        final = df.iloc[-1]
        case = int(cfg['case'])
        t_end = float(cfg['T'])
        if abs(float(final.time) - t_end) > 1e-8:
            raise ValueError(f'Incomplete run: {folder}, final t={final.time}, T={t_end}')
        rows.append({
            'folder': folder.name, 'case': case, 'method': str(cfg['method']),
            'n': int(cfg['n']), 'dt': float(final['dt']), 'epsilon': float(cfg['epsilon']),
            'T': t_end, 'saved_snapshots': len(df),
            'final_relative_L2': float(final.l2_relative) if case == 3 else np.nan,
            'worst_min': float(df.min_u.min()), 'worst_max': float(df.max_u.max()),
            'final_min': float(final.min_u), 'final_max': float(final.max_u),
            'initial_integral': float(df.iloc[0].integral_u),
            'final_integral': float(final.integral_u),
        })
    if not rows:
        raise FileNotFoundError(f'No valid runs found under {root}')
    return pd.DataFrame(rows)


def convergence(df: pd.DataFrame, out: Path, mode: str) -> None:
    chosen = df.loc[df.case.eq(3) & df.final_relative_L2.notna()]
    if mode == 'spatial':
        keys, x, label = ['method','epsilon','dt','T'], 'n', 'Mesh intervals per direction N'
    else:
        keys, x, label = ['method','epsilon','n','T'], 'dt', 'Time step Δt'
    records=[]
    for group, block in chosen.groupby(keys):
        block=block.sort_values(x).drop_duplicates(x)
        if len(block)<2:
            continue
        xx=block[x].to_numpy(float)
        ee=block.final_relative_L2.to_numpy(float)
        p=np.full(len(block),np.nan)
        if mode=='spatial':
            p[1:]=np.log(ee[:-1]/ee[1:])/np.log(xx[1:]/xx[:-1])
        else:
            p[1:]=np.log(ee[1:]/ee[:-1])/np.log(xx[1:]/xx[:-1])
        for i,row in enumerate(block.to_dict('records')):
            records.append({**{key:val for key,val in zip(keys,group)}, x:xx[i],
                            'error':ee[i],'observed_error_order_vs_previous':p[i]})
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.loglog(xx,ee,'o-',lw=2)
        ax.set_xlabel(label)
        ax.set_ylabel('Final relative L2 error vs analytical solution')
        ax.set_title(f'Example 3 {mode} refinement: ' + ', '.join(f'{k}={v}' for k,v in zip(keys,group)))
        ax.grid(True,which='both',alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / (f'{mode}_' + '_'.join(f'{k}{v}' for k,v in zip(keys,group)).replace('/','_') + '.png'), dpi=160)
        plt.close(fig)
    pd.DataFrame(records).to_csv(out/f'{mode}_refinement.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description='SMTNO 3A.5 read-only run audit')
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results'/'step3A')
    ap.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'results'/'step3A5')
    a=ap.parse_args()
    root=a.root.expanduser().resolve();out=a.outdir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    runs=collect_runs(root)
    runs.sort_values(['case','epsilon','method','dt','n']).to_csv(out/'all_runs.csv',index=False)
    for name in ['spatial','temporal']:
        convergence(runs,out,name)
    for case in [3,5]:
        subset=runs[runs.case==case]
        if subset.empty:continue
        fig,ax=plt.subplots(figsize=(9,5))
        metric='final_relative_L2' if case==3 else 'worst_min'
        for label,g in subset.groupby(['epsilon','method']):
            ax.scatter(g.n,g[metric],s=70,label=f'epsilon={label[0]:g}, {label[1]}')
        ax.set(xlabel='N (not all time steps equal)',ylabel=metric,title=f'Example {case}: diagnostic only — see matched-condition convergence plots')
        ax.grid(alpha=.3);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f'example{case}_run_diagnostics.png',dpi=170);plt.close(fig)
    sharp=runs[(runs.case==3)&(runs.epsilon<=1e-6)]
    if not sharp.empty:
        sharp[['folder','n','dt','final_relative_L2','worst_min','worst_max']].to_csv(out/'sharp_layer_warning.csv',index=False)
    print(runs.to_string(index=False))
    print('\nNote: errors at epsilon=1e-8 on coarse meshes are NOT validated fine references.')
    print('Results:',out)


if __name__=='__main__':main()