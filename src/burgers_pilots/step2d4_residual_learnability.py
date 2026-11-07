#!/usr/bin/env python3
"""SMTNO Step 2D.4 - Residual Learnability Test (teacher-forced diagnostic).

Question: does four-snapshot resolved-state history predict the residual weak
forcing beyond what is predictable from the current resolved state alone?

Residual at physical scale ell, in the normalization of the frozen Step 2A FNO:
    R = (N * S_tau / target_scales[ell]) - FNO(u(t),ell)

Inputs use ONLY projected coarse FEM states, not the oracle stress or residual.
The same trajectory IDs, target times, scale paths and model width are used for
both predictors. Train N=64; evaluate on held-out N=64 and N=128 (zero-shot).
This is NOT a rollout test and does NOT establish causal memory or novelty.

Requires sibling scripts:
  step2a_train_closure_fno.py
  step2b_closed_loop_rollout.py
Requires existing Drive files:
  datasets/burgers_step1F_ensemble.npz
  checkpoints/step2A/memoryless.pt

Example in Colab:
  python step2d4_residual_learnability.py --epochs 45 --batch-size 128
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from step2a_train_closure_fno import ClosureFNO
from step2b_closed_loop_rollout import load_checkpoint

PATHS = (
    ('fixed_001', 0, 0, False),
    ('fixed_002', 1, 1, False),
    ('fixed_004', 2, 2, False),
    ('switch_001_to_004', 0, 2, True),
    ('switch_002_to_004', 1, 2, True),
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path,
                   default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    p.add_argument('--dataset', type=Path, default=None)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--width', type=int, default=24)
    p.add_argument('--modes', type=int, default=10)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--history', type=int, default=4)
    p.add_argument('--switch-time', type=float, default=.10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--smoke-test', action='store_true',
                   help='Run a small wiring test, with separately labeled outputs')
    return p.parse_args()


def gaussian_transfer_np(u, old, new):
    """Gaussian scale transfer of a periodic field, preserving its mean."""
    if new + 1e-12 < old:
        raise ValueError('Only fine-to-coarse physical filter transfer is supported')
    n = u.shape[-1]
    wave = np.fft.rfftfreq(n, d=1.0/n)
    factor = np.exp(-.5 * (2*np.pi*wave)**2 * (new**2 - old**2))
    return np.fft.irfft(np.fft.rfft(u, axis=-1)*factor, n=n, axis=-1).astype('float32')


@torch.no_grad()
def frozen_fno_predictions(base, u, ells, xstd, device, batch=512):
    """u (cases, scales, times, N), result in normalized Step2A forcing units."""
    cases, ns, timesteps, n = u.shape
    pred = np.empty_like(u, dtype='float32')
    base.eval()
    for j in range(ns):
        source = u[:, j].reshape(-1, n)
        chunks = []
        for start in range(0, len(source), batch):
            v = torch.as_tensor(source[start:start+batch], device=device,
                                dtype=torch.float32)
            e = v.new_full((len(v), n, 1), float(ells[j]/np.max(ells)))
            x = torch.cat((v.unsqueeze(-1)/xstd, e), dim=-1)
            chunks.append(base(x).detach().cpu().numpy())
        pred[:, j] = np.concatenate(chunks, axis=0).reshape(cases, timesteps, n)
    return pred


def make_examples(u, oracle_normalized, baseline, ells, case_indices,
                  times, hist, switch_idx, xstd):
    """Create exactly matched eligible labels and switch paths for both models.

    Historical states before a scale change are Gaussian-transferred to the
    current scale; no future state or oracle residual is used as input.
    """
    n = u.shape[-1]
    transfers = {(old, new): gaussian_transfer_np(u[:, old], float(ells[old]),
                                                 float(ells[new]))
                 for old, new in ((0, 2), (1, 2))}
    xs, ys, labels, meta = [], [], [], []
    for c in case_indices:
        for path, old, new, do_switch in PATHS:
            for t in range(hist-1, len(times)):
                j = new if do_switch and t >= switch_idx else old
                v = np.empty((n, hist+1), dtype='float32')
                for k in range(hist):
                    past_t = t-k
                    past_j = old if do_switch and past_t < switch_idx else j
                    if past_j == j:
                        state = u[c, j, past_t]
                    else:
                        state = transfers[(past_j, j)][c, past_t]
                    v[:, k] = state / xstd
                v[:, hist] = float(ells[j]/np.max(ells))
                # Residual in EXACT SAME units as the Step 2A training target.
                resid = oracle_normalized[c, j, t] - baseline[c, j, t]
                xs.append(v)
                ys.append(resid.astype('float32'))
                labels.append(oracle_normalized[c, j, t].astype('float32'))
                if not do_switch:
                    period = 'fixed'
                elif t < switch_idx:
                    period = 'switch_pre'
                elif t < switch_idx + hist:
                    period = 'switch_boundary'
                else:
                    period = 'switch_late'
                meta.append((int(c),path,int(t),int(j),period,int(do_switch)))
    X=np.stack(xs)
    Y=np.stack(ys)
    O=np.stack(labels)
    # The oracle and frozen FNO are conservative weak forcings. Verify this.
    if float(np.max(np.abs(Y.mean(axis=-1)))) > 1e-4:
        raise ValueError('Residual forcing is unexpectedly nonconservative')
    return dict(X=X,Y=Y,oracle=O,meta=meta)


def trim_inputs(X, count):
    """Current-state: first state + ell. History: all states + ell."""
    return np.ascontiguousarray(np.concatenate((X[..., :count],X[..., -1:]),axis=-1))


def new_model(history, args, device):
    net = ClosureFNO(history+1,width=args.width,modes=args.modes,
                     layers=args.layers).to(device)
    # Both models start from exactly ZERO residual: frozen FNO unchanged.
    torch.nn.init.zeros_(net.head[-1].weight)
    torch.nn.init.zeros_(net.head[-1].bias)
    return net


@torch.no_grad()
def predict(net, X, device, batch_size):
    net.eval(); out=[]
    for start in range(0,len(X),batch_size):
        v=torch.as_tensor(X[start:start+batch_size],device=device,dtype=torch.float32)
        out.append(net(v).cpu().numpy())
    return np.concatenate(out, axis=0)


def normalized_mse(p,y):
    return float(np.mean((np.asarray(p,dtype=np.float64)-
                          np.asarray(y,dtype=np.float64))**2))


def fit_model(name, count, datasets, args, device, out, ckdir):
    X_train=trim_inputs(datasets['train']['X'],count)
    Y_train=datasets['train']['Y']
    X_val=trim_inputs(datasets['val']['X'],count)
    Y_val=datasets['val']['Y']
    torch.manual_seed(args.seed)
    net=new_model(count,args,device)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-6)
    best=normalized_mse(np.zeros_like(Y_val),Y_val)
    initial_val=best
    state=copy.deepcopy(net.state_dict())
    best_epoch=0; patience_left=args.patience; history=[]
    n=len(X_train)
    for ep in range(1,args.epochs+1):
        net.train();train_sum=0.;seen=0
        # Identical seed, sample order, labels and batch boundaries for both.
        permutation=np.random.default_rng(args.seed+ep).permutation(n)
        for st in range(0,n,args.batch_size):
            ind=permutation[st:st+args.batch_size]
            xb=torch.as_tensor(X_train[ind],device=device,dtype=torch.float32)
            yb=torch.as_tensor(Y_train[ind],device=device,dtype=torch.float32)
            pr=net(xb)
            loss=(pr-yb).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            train_sum+=float(loss.item())*len(ind); seen+=len(ind)
        pred=predict(net,X_val,device,args.eval_batch_size)
        val=normalized_mse(pred,Y_val)
        history.append(dict(model=name,epoch=ep,train_mse=train_sum/seen,
                            validation_mse=val))
        if val < best*(1-1e-5):
            best=val; best_epoch=ep; state=copy.deepcopy(net.state_dict())
            patience_left=args.patience
        else:
            patience_left-=1
        print(f'{name:>8s} epoch={ep:03d} train={train_sum/seen:.6e} '
              f'val={val:.6e} best_epoch={best_epoch}',flush=True)
        if patience_left==0:break
    net.load_state_dict(state)
    torch.save(dict(model=name,history=count,state_dict=state,best_epoch=best_epoch,
                    best_val_normalized_mse=best,initial_zero_val=initial_val,
                    frozen_base='step2A/memoryless.pt',
                    network=dict(width=args.width,modes=args.modes,layers=args.layers)),
               ckdir/f'residual_{name}.pt')
    return net, history,dict(best_epoch=best_epoch,best_validation_mse=best,
                             initial_zero_validation_mse=initial_val)


def shuffled_past(X, meta, count, seed):
    """Diagnostic: reassign past channels across cases at same path and time.

    The current state, scale, residual label and sample group never change.
    Only past channels k=1,...,count-1 are swapped. Not a causal test.
    """
    copied=trim_inputs(X,count)
    if count==1:return copied
    groups={}
    for i, row in enumerate(meta):
        _,path,t,_,_,_=row
        groups.setdefault((path,t),[]).append(i)
    rng=np.random.default_rng(seed)
    for group in groups.values():
        if len(group)<2:continue
        ids=np.asarray(group)
        donors=np.roll(ids,int(rng.integers(1,len(ids))))
        copied[ids, :, 1:count]=X[donors, :, 1:count]
    return copied


def summaries(data, preds, n, split, ells, hist, shuffled=None):
    """Aggregate MSE on identical sample masks; report zero-residual skill."""
    meta=data['meta'];Y=data['Y'];O=data['oracle']
    t=np.array([x[2] for x in meta]);j=np.array([x[3] for x in meta])
    phases=np.array([x[4] for x in meta]); paths=np.array([x[1] for x in meta])
    masks=[('all',np.ones(len(Y),dtype=bool)),
           ('fixed_all',phases=='fixed'),
           ('switch_all',phases!='fixed'),
           ('switch_pre',phases=='switch_pre'),
           ('switch_boundary',phases=='switch_boundary'),
           ('switch_late',phases=='switch_late'),
           ('switch_post',np.isin(phases,['switch_boundary','switch_late']))]
    for path in sorted(set(paths)):
        masks.append((path,paths==path))
    for k,ell in enumerate(ells):
        masks.append((f'ell_{ell:g}',j==k))
    out=[]
    for group,mask in masks:
        if not np.any(mask):continue
        z=normalized_mse(np.zeros_like(Y[mask]),Y[mask])
        c=normalized_mse(preds['current'][mask],Y[mask])
        h=normalized_mse(preds['history'][mask],Y[mask])
        # Physical S/h units, not normalized; scales are checkpoint-derived.
        yphys=Y[mask]*np.asarray(hist, 'float64')[j[mask],None]
        ophys=O[mask]*np.asarray(hist, 'float64')[j[mask],None]
        baseline_rms_ratio=float(np.sqrt(np.mean(yphys**2)) /
             max(np.sqrt(np.mean(ophys**2)),1e-14))
        out.append(dict(N=n,split=split,group=group,n_samples=int(np.sum(mask)),
            zero_residual_mse=z,current_mse=c,history_mse=h,
            current_skill_vs_zero_pct=100*(1-c/max(z,1e-20)),
            history_skill_vs_zero_pct=100*(1-h/max(z,1e-20)),
            history_gain_vs_current_pct=100*(1-h/max(c,1e-20)),
            frozen_fno_residual_rms_over_oracle=baseline_rms_ratio,
            shuffled_history_mse=(normalized_mse(shuffled[mask],Y[mask])
                                   if shuffled is not None else float('nan'))))
    return out


def per_case_comparison(data, preds, n, split):
    Y=data['Y'];meta=data['meta']
    ids=np.array([m[0] for m in meta]);periods=np.array([m[4] for m in meta])
    out=[]
    for cid in sorted(set(ids)):
        for group,mask in [('all',ids==cid),
                           ('switch_post',(ids==cid)&np.isin(periods,['switch_boundary','switch_late']))]:
            if not mask.any():continue
            cc=normalized_mse(preds['current'][mask],Y[mask]); hh=normalized_mse(preds['history'][mask],Y[mask])
            out.append(dict(N=n,split=split,case_index=int(cid),group=group,
                            current_mse=cc,history_mse=hh,
                            history_gain_pct=100*(1-hh/max(cc,1e-20)),
                            history_wins=bool(hh<cc)))
    return out


def write_csv(file,rows):
    if not rows: return
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plots(out, learning, summary, data64, preds64, times, switch_t):
    fig,ax=plt.subplots(figsize=(8,4.6))
    for key in ('current','history'):
        rec=[z for z in learning if z['model']==key]
        ax.semilogy([0]+[r['epoch'] for r in rec],
                    [next(x['zero_residual_mse'] for x in summary if x['N']==64 and
                          x['split']=='val' and x['group']=='all')]+[r['validation_mse'] for r in rec],
                    marker='.',label=key)
    ax.set(xlabel='Epoch',ylabel='Validation MSE (normalized residual)',
           title='2D.4: paired residual learnability')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'residual_validation_curves.png',dpi=170);plt.close(fig)

    rows=[r for r in summary if r['split']=='test' and r['group'] in ('all','switch_boundary')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,n in zip(axs,(64,128)):
        for group,style in [('all','-'),('switch_boundary','--')]:
            r=next((z for z in rows if z['N']==n and z['group']==group),None)
            if r is None:continue
            ax.plot(['zero residual','current','history'],
                    [r['zero_residual_mse'],r['current_mse'],r['history_mse']],
                    style+'o',label=group)
        ax.set(title=f'held-out N={n}',ylabel='Normalized residual MSE')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Residual prediction: lower is better')
    fig.tight_layout();fig.savefig(out/'residual_comparison.png',dpi=170);plt.close(fig)

    dd=data64;rr=dd['Y'];mm=dd['meta']
    for group in ['switch_001_to_004']:
        ind=np.flatnonzero(np.array([z[1]==group for z in mm]))
        if not len(ind):continue
        cases=np.array([mm[k][0] for k in ind]);c=cases[0];ind=ind[cases==c]
        t=np.array([times[mm[k][2]] for k in ind]);ordr=np.argsort(t)
        fig,ax=plt.subplots(figsize=(9,4.7))
        for label,pred in [('zero residual',np.zeros_like(rr)),
                           ('current',preds64['current']),('history',preds64['history'])]:
            mse=np.mean((pred[ind]-rr[ind])**2,axis=-1)[ordr]
            ax.semilogy(t[ordr],np.maximum(mse,1e-14),label=label)
        ax.axvline(switch_t,color='k',ls=':',lw=1,label='scale switch')
        ax.set(xlabel='Physical time',ylabel='Normalized residual MSE',
               title=f'Switch case {c}: ell 0.01 to 0.04')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/'example_switch_residual_error.png',dpi=170);plt.close(fig)


def main():
    a=args_parser();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
    device=torch.device('cpu' if a.cpu else ('cuda' if torch.cuda.is_available() else
        ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')))
    if a.history<2:raise ValueError('History must be >=2')
    p=a.project_dir;datapath=a.dataset or p/'datasets'/'burgers_step1F_ensemble.npz'
    if not datapath.is_file():raise FileNotFoundError(datapath)
    with np.load(datapath,allow_pickle=False) as z:
        d={k:z[k] for k in ('ell','times','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128')}
    ell=np.asarray(d['ell'],dtype=float);times=np.asarray(d['times'],dtype=float)
    if not np.allclose(ell,[.01,.02,.04]):raise ValueError('Unexpected physical filter scales')
    if not np.allclose(np.diff(times),np.diff(times)[0]):raise ValueError('Saved snapshots are not equally spaced')
    where=np.flatnonzero(np.isclose(times,a.switch_time,atol=1e-9))
    if not len(where):raise ValueError('Switch time is absent from saved times')
    switch_idx=int(where[0])
    if switch_idx<a.history:raise ValueError('Switch occurs before enough history is available')
    split=d['split'].astype(str)
    ids={s:np.flatnonzero(split==s) for s in ('train','val','test')}
    if tuple(len(ids[s]) for s in ('train','val','test'))!=(20,5,5):
        raise ValueError('Expected 20/5/5 complete-trajectory split')
    if a.smoke_test:
        ids={s:v[:(2 if s=='train' else 1)] for s,v in ids.items()}
        a.epochs=1
    base,xstd,ystd,base_hist=load_checkpoint('memoryless',p/'checkpoints'/'step2A',device,1,ell)
    if base_hist!=1:raise ValueError('Frozen baseline must be memoryless')
    if np.any(np.asarray(ystd)<=0):raise ValueError('Invalid target scales')
    base.eval()
    for param in base.parameters():param.requires_grad_(False)
    out=p/'results'/('step2D4_smoke' if a.smoke_test else 'step2D4')
    ck=p/'checkpoints'/('step2D4_smoke' if a.smoke_test else 'step2D4')
    out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    print(f'CONFIG device={device} train={len(ids["train"])} val={len(ids["val"])} '
          f'test={len(ids["test"])} switch_t={a.switch_time}',flush=True)
    by_n={}
    for n in (64,128):
        u=np.asarray(d[f'coarse_N{n}'],dtype='float32')
        S=np.asarray(d[f'S_tau_N{n}'],dtype='float32')
        oracle=(S*n/np.asarray(ystd,dtype='float32')[None,:,None,None])
        frozen=frozen_fno_predictions(base,u,ell,xstd,device,a.eval_batch_size)
        by_n[n]={s:make_examples(u,oracle,frozen,ell,ids[s],times,a.history,
                                 switch_idx,xstd) for s in ('train','val','test') if
                 n==64 or s in ('val','test')}
        if n==64:
            print('N64 aligned samples:',{k:len(v['Y']) for k,v in by_n[n].items()},flush=True)
    models={};learning=[];configs={}
    for name,h in (('current',1),('history',a.history)):
        net,logs,info=fit_model(name,h,by_n[64],a,device,out,ck)
        models[name]=net;learning+=logs;configs[name]=info
    write_csv(out/'training_curves.csv',learning)
    summaries_all=[];cases_all=[]
    preds64=None
    for n in (64,128):
        for split_name in ('val','test'):
            dset=by_n[n][split_name]
            preds={name:predict(net,trim_inputs(dset['X'],1 if name=='current' else a.history),
                                 device,a.eval_batch_size) for name,net in models.items()}
            shuffled=predict(models['history'],shuffled_past(dset['X'],dset['meta'],a.history,a.seed+19),
                             device,a.eval_batch_size)
            summaries_all.extend(summaries(dset,preds,n,split_name,ell,ystd,shuffled))
            if split_name=='test':
                cases_all.extend(per_case_comparison(dset,preds,n,split_name))
            if n==64 and split_name=='test':preds64=preds
    write_csv(out/'residual_metrics.csv',summaries_all)
    write_csv(out/'case_level_gains.csv',cases_all)
    plots(out,learning,summaries_all,by_n[64]['test'],preds64,times,a.switch_time)
    gates=[]
    for n in (64,128):
        r=next(z for z in summaries_all if z['N']==n and z['split']=='test' and z['group']=='switch_post')
        cc=[z for z in cases_all if z['N']==n and z['group']=='switch_post']
        gates.append(dict(N=n,history_gain_pct=r['history_gain_vs_current_pct'],
                          history_win_cases=sum(z['history_wins'] for z in cc),
                          n_cases=len(cc)))
    (out/'run_config.json').write_text(json.dumps({
        'purpose':'Teacher-forced learnability test, NOT scale-switch autonomous rollout',
        'model_base':'Frozen Step 2A memoryless FNO',
        'dataset':str(datapath),'device':str(device),
        'normalization':'(N * S_tau / target_scale) - frozen_FNO_normalized',
        'reference_training_N':64,'heldout_resolutions':[64,128],
        'filter_widths':ell.tolist(),'switch_time':a.switch_time,
        'history_snapshots':a.history,'paths':[z[0] for z in PATHS],
        'paired_sample_windows':True,'trajectory_split':{k:len(v) for k,v in ids.items()},
        'models':configs,'switch_post_test_gains':gates,
        'smoke_test':bool(a.smoke_test),
        'interpretation_warning':'This is a one-step, teacher-forced predictor diagnostic. Do not claim learned Mori-Zwanzig memory, improved rollout, or architectural novelty.'
    },indent=2))
    print('\nHELD-OUT PERFORMANCE (normalized residual MSE):',flush=True)
    for n in (64,128):
        for group in ('all','switch_boundary','switch_post'):
            r=next(z for z in summaries_all if z['N']==n and z['split']=='test' and z['group']==group)
            print(f'N={n} {group:15s} zero={r["zero_residual_mse"]:.6e} '
                  f'current={r["current_mse"]:.6e} history={r["history_mse"]:.6e} '
                  f'hist_gain={r["history_gain_vs_current_pct"]:.2f}%',flush=True)
    print('Case-level post-switch history wins:',gates,flush=True)
    print('Saved results to:',out,flush=True)


if __name__=='__main__':main()