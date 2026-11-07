from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_ints(s):
    return [int(v) for v in s.split(',') if v.strip()]


def aggregate(root: Path, seeds: list[int]):
    frames = []
    for seed in seeds:
        path = root/f'seed_{seed}'/'forecast_metrics.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_csv(path)
        d['seed'] = seed
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(root/'all_forecasts.csv', index=False)
    # Average repeated windows *within each independent case* before comparison.
    per_case = df.groupby(['seed','split','n','ell','horizon','case_id','model'],as_index=False).agg(
        rel_state_error=('rel_state_error','mean'))
    per_case.to_csv(root/'per_case_errors.csv', index=False)
    pivot = per_case.pivot_table(index=['seed','split','n','ell','horizon','case_id'],
                                 columns='model',values='rel_state_error').reset_index()
    if not {'current','history'}.issubset(pivot.columns):
        raise ValueError('Expected current and history forecast metrics')
    pivot['history_gain_pct'] = 100*(pivot.current-pivot.history)/np.maximum(pivot.current,1e-12)
    pivot.to_csv(root/'paired_history_gain.csv',index=False)
    test = pivot[pivot.split=='test']
    # Equal weight per seed and per independent test case; ell retained as an experimental factor.
    seed_means = test.groupby(['seed','n','horizon'],as_index=False).agg(
        current_error=('current','mean'),history_error=('history','mean'),
        history_gain_pct=('history_gain_pct','mean'))
    seed_means.to_csv(root/'seed_summary.csv',index=False)
    group = seed_means.groupby(['n','horizon'],as_index=False).agg(
        mean_gain_pct=('history_gain_pct','mean'),sd_across_seeds_pct=('history_gain_pct','std'),
        mean_current_error=('current_error','mean'),mean_history_error=('history_error','mean'),
        n_seeds=('seed','count'))
    group.to_csv(root/'repeat_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for n,part in group.groupby('n'):
        s=part.sort_values('horizon')
        ax.errorbar(s.horizon,s.mean_gain_pct,yerr=s.sd_across_seeds_pct.fillna(0),
                    marker='o',capsize=3,label=f'N={n}')
    ax.axhline(0,color='gray',lw=1)
    ax.set(xlabel='Forecast horizon (saved steps)',ylabel='History gain (%)',
           title='Test trajectories: mean ± SD across training seeds')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(root/'repeat_history_gain.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for n,part in seed_means.groupby('n'):
        for seed,g in part.groupby('seed'):
            ax.plot(g.horizon,g.history_gain_pct,marker='o',alpha=.7,label=f'N={n}, seed={seed}')
    ax.axhline(0,color='gray',lw=1)
    ax.set(xlabel='Forecast horizon',ylabel='History gain (%)',title='Seed-by-seed test results')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(root/'per_seed_history_gain.png',dpi=180);plt.close(fig)
    cfg={'seeds':seeds,'note':'SD is variability over training seeds, not a confidence interval. Test trajectories are paired by case ID and filter width. No evidence of physical memory is implied by negative-control gains.'}
    (root/'aggregate_config.json').write_text(json.dumps(cfg,indent=2))
    print('\nRepeated-run summary:\n',group.to_string(index=False))


def main():
    ap=argparse.ArgumentParser(description='Repeat Step 3C.1 with multiple random seeds and longer horizons.')
    ap.add_argument('--trainer',default='step3c1_train.py')
    ap.add_argument('--indir',default='results/step3B4')
    ap.add_argument('--outdir',default='results/step3C2A')
    ap.add_argument('--seeds',default='123,456,789')
    ap.add_argument('--epochs',type=int,default=70)
    ap.add_argument('--horizons',default='1,3,5,10')
    ap.add_argument('--eval-stride',type=int,default=3)
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--max-train-cases',type=int,default=None)
    ap.add_argument('--summary-only',action='store_true')
    args=ap.parse_args();root=Path(args.outdir);root.mkdir(exist_ok=True,parents=True)
    seeds=parse_ints(args.seeds)
    if len(seeds)<1 or len(set(seeds))!=len(seeds):ap.error('Use nonempty distinct seeds')
    if not args.summary_only:
        trainer=Path(args.trainer)
        if not trainer.exists():raise FileNotFoundError(f'Copy existing Step 3C.1 trainer into this directory: {trainer}')
        for seed in seeds:
            run=root/f'seed_{seed}'
            if (run/'forecast_metrics.csv').exists() and (run/'run_config.json').exists():
                cfg=json.loads((run/'run_config.json').read_text())
                training_log=run/'training_history.csv'
                completed_epochs=0
                if training_log.exists():
                    log=pd.read_csv(training_log)
                    completed_epochs=int(log.groupby('model').epoch.max().min()) if not log.empty else 0
                if (cfg.get('seed')==seed and cfg.get('forecast_horizons')==parse_ints(args.horizons)
                        and cfg.get('train_resolution')==32 and cfg.get('eval_stride')==args.eval_stride
                        and completed_epochs==args.epochs):
                    print(f'Reusing completed seed {seed}',flush=True);continue
                raise RuntimeError(f'{run} already exists with different settings. Choose another --outdir.')
            cmd=[sys.executable,str(trainer),'--indir',args.indir,'--outdir',str(run),
                 '--train-res','32','--test-res','64','--ells','0.01,0.02,0.04',
                 '--history','4','--epochs',str(args.epochs),'--batch-size',str(args.batch_size),
                 '--horizons',args.horizons,'--eval-stride',str(args.eval_stride),
                 '--device',args.device,'--seed',str(seed)]
            if args.max_train_cases is not None:cmd.extend(['--max-train-cases',str(args.max_train_cases)])
            print('Running seed',seed,flush=True)
            subprocess.run(cmd,check=True)
    aggregate(root,seeds)

if __name__=='__main__':main()
