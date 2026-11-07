#!/usr/bin/env python3
"""SMTNO 2D.4D: causal FUTURE residual forecasting, not current-time reconstruction.

Only u(t), u(t-dt), ... at/before cutoff t are model inputs. The target
r(t+h) is never available to the model. Identical split of full base families
(20/5/5), common cutoff times for all horizons, and zero-shot N=128 evaluation.
This is a teacher-observed-history forecasting test, NOT autonomous rollout,
FEM/SUPG simulation, or validation of a learned Mori-Zwanzig kernel.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODELS = ('markov_fno', 'history_fno', 'lstm', 'fno_lstm')
BASELINES = ('zero', 'bdf2_hold', 'bdf2_linear')


class SpectralConv1D(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / (width**2))

    def forward(self, z):
        b, c, n = z.shape
        zft = torch.fft.rfft(z, dim=-1)
        nuse = min(self.modes, zft.shape[-1])
        yft = torch.zeros_like(zft)
        yft[:, :, :nuse] = torch.einsum('bim,iom->bom', zft[:, :, :nuse], self.weights[:, :, :nuse])
        return torch.fft.irfft(yft, n=n, dim=-1)


class FNO1D(nn.Module):
    def __init__(self, channels, width, modes, out_channels=1):
        super().__init__()
        self.lift = nn.Conv1d(channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv1D(width, modes) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(2)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, out_channels, 1))

    def forward(self, features):
        z = self.lift(features.transpose(1, 2))
        for sp, lc in zip(self.spec, self.local):
            z = torch.nn.functional.gelu(sp(z) + lc(z))
        return self.head(z).transpose(1, 2)


class ForecastNet(nn.Module):
    def __init__(self, name, history, width, modes):
        super().__init__()
        self.name = name
        if name == 'markov_fno':
            self.fno = FNO1D(2, width, modes)
        elif name == 'history_fno':
            self.fno = FNO1D(history + 1, width, modes)
        elif name == 'lstm':
            self.lstm = nn.LSTM(input_size=4, hidden_size=width, num_layers=1, batch_first=True)
            self.head = nn.Linear(width, 1)
        elif name == 'fno_lstm':
            self.encoder = FNO1D(2, width, modes, out_channels=width)
            self.lstm = nn.LSTM(input_size=width, hidden_size=width, num_layers=1, batch_first=True)
            self.head = nn.Linear(width, 1)
        else:
            raise ValueError(name)

    def forward(self, x, scale):
        # x: (batch, oldest-to-current history, spatial cells); scale: (batch,)
        b, h, n = x.shape
        if self.name == 'markov_fno':
            features = torch.stack((x[:, -1], scale[:, None].expand(b, n)), dim=-1)
            return self.fno(features).squeeze(-1)
        if self.name == 'history_fno':
            features = torch.cat((x.transpose(1, 2), scale[:, None, None].expand(b, n, 1)), dim=-1)
            return self.fno(features).squeeze(-1)
        if self.name == 'lstm':
            us = x.transpose(1, 2)
            left = torch.roll(us, 1, dims=1)
            right = torch.roll(us, -1, dims=1)
            feats = torch.stack((left, us, right, scale[:, None, None].expand(b, n, h)), dim=-1)
            seq = feats.reshape(b*n, h, 4)
            state, _ = self.lstm(seq)
            return self.head(state[:, -1]).reshape(b, n)
        if self.name == 'fno_lstm':
            z = x.reshape(b*h, n, 1)
            sc = scale[:, None].expand(b, h).reshape(b*h, 1, 1).expand(-1, n, 1)
            embeddings = self.encoder(torch.cat((z, sc), dim=-1))
            seq = embeddings.reshape(b, h, n, -1).permute(0, 2, 1, 3).reshape(b*n, h, -1)
            state, _ = self.lstm(seq)
            return self.head(state[:, -1]).reshape(b, n)
        raise ValueError(self.name)


def split_base_ids(ids, seed):
    bases = np.unique(ids)
    if len(bases) != 30:
        raise ValueError(f'Expected 30 base families; got {len(bases)}')
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    return {'train': bases[:20], 'val': bases[20:25], 'test': bases[25:]}


def make_examples(d, regime, N, selected_ids, history, cutoff_start, cutoff_end, horizon, xstd, ystd):
    # Select BOTH hidden-state variants for each base, never split sibling variants.
    ids = d['base_id']
    trial_indices = np.flatnonzero(np.isin(ids, selected_ids))
    u = d[f'{regime}_u_N{N}']
    r = d[f'{regime}_r_N{N}']
    ell = d['ell']
    X, Y, E, meta = [], [], [], []
    for ci in trial_indices:
        for sj in range(len(ell)):
            for t in range(cutoff_start, cutoff_end+1):
                # All inputs are at or BEFORE t. No future state in X.
                X.append(u[ci, sj, t-history+1:t+1]/xstd)
                Y.append(r[ci, sj, t+horizon]/ystd[sj])
                E.append(float(ell[sj]/np.max(ell)))
                meta.append((int(ids[ci]), int(d['variant'][ci]), sj, t, t+horizon))
    return (np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32),
            np.asarray(E, dtype=np.float32), np.asarray(meta, dtype=np.int32))


def reference_rhs(u, N):
    # Same homogeneous coarse operator as in Step 2D.4A; periodic upwind + diffusion.
    dx = 1.0/N
    velocity = .35
    d_hom = .003*np.sqrt(3.0)
    return (velocity/dx)*(np.roll(u, 1, axis=-1)-u) + (d_hom/dx**2)*(
        np.roll(u, 1, axis=-1)-2.0*u+np.roll(u, -1, axis=-1))


def numeric_baselines(d, regime, N, meta, horizon):
    # Compute at the cutoff only. NEVER compute a derivative at t+horizon.
    u = d[f'{regime}_u_N{N}']
    dt = float(d['times'][1]-d['times'][0])
    ids = d['base_id']; variants = d['variant']
    case_lookup = {(int(b), int(v)): i for i, (b, v) in enumerate(zip(ids, variants))}
    cutoff_idx = np.asarray([case_lookup[(int(b),int(v))] for b,v in meta[:, :2]], dtype=np.int32)
    j = meta[:,2]; t = meta[:,3]
    now = u[cutoff_idx,j,t].astype(np.float64)
    last = u[cutoff_idx,j,t-1].astype(np.float64)
    second = u[cutoff_idx,j,t-2].astype(np.float64)
    third = u[cutoff_idx,j,t-3].astype(np.float64)
    current_bdf = (3*now-4*last+second)/(2*dt)-reference_rhs(now,N)
    prev_bdf = (3*last-4*second+third)/(2*dt)-reference_rhs(last,N)
    return {
        'zero':np.zeros_like(current_bdf, dtype=np.float32),
        'bdf2_hold':current_bdf.astype(np.float32),
        'bdf2_linear':(current_bdf+horizon*(current_bdf-prev_bdf)).astype(np.float32),
    }


def make_loader(samples, batch_size, shuffle, seed):
    x, y, e, _ = samples
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(e)),
                      batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def epoch_mse(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total=0.0; count=0
    for X, Y, E in loader:
        X=X.to(device);Y=Y.to(device);E=E.to(device)
        with torch.set_grad_enabled(training):
            prediction=model(X,E)
            loss=torch.mean((prediction-Y)**2)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total+=float(loss.item())*Y.numel()
        count+=Y.numel()
    return total/max(count, 1)


def network_predict(model, sample, ystd, device, batch_size, replace_history=False):
    x, _, scale, meta = sample
    out=[]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb=torch.from_numpy(x[i:i+batch_size]).to(device)
            if replace_history:
                # Negative control: current state remains unchanged, earlier states erased.
                xb=xb.clone()
                xb[:,:-1,:]=xb[:,-1:,:]
            sb=torch.from_numpy(scale[i:i+batch_size]).to(device)
            out.append(model(xb,sb).cpu().numpy())
    return np.concatenate(out,axis=0)*ystd[meta[:,2],None]


def append_metrics(rows, cases, regime, N, horizon, method, reference, pred, meta, ell_values, cutoff_dt):
    for j, ell in enumerate(ell_values):
        isscale=meta[:,2]==j
        truth=reference[isscale]; forecast=pred[isscale]
        err=forecast-truth
        rmse=float(np.sqrt(np.mean(err**2)))
        rms=float(np.sqrt(np.mean(truth**2)))
        rows.append({'regime':regime,'N':N,'horizon_steps':horizon,
                     'horizon_time':horizon*cutoff_dt,'method':method,'ell':float(ell),
                     'relative_l2':float(np.linalg.norm(err)/max(np.linalg.norm(truth),1e-12)),
                     'rmse':rmse, 'target_rms':rms,
                     'improvement_over_zero_pct':100*(1-rmse/max(rms,1e-12)),
                     'samples':int(np.sum(isscale))})
        base_ids=np.unique(meta[isscale,0])
        for bid in base_ids:
            mask=isscale & (meta[:,0]==bid)
            rr=reference[mask]; pp=pred[mask]
            cases.append({'regime':regime,'N':N,'horizon_steps':horizon,'method':method,
                          'ell':float(ell),'base_id':int(bid),'n_variants':int(len(np.unique(meta[mask,1]))),
                          'rmse':float(np.sqrt(np.mean((pp-rr)**2))),
                          'reference_rms':float(np.sqrt(np.mean(rr**2)))})


def save_csv(path, rows):
    if not rows: return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def create_plots(folder, metrics, curves, ell_values, horizons, models, n_values):
    names=list(BASELINES)+list(models)
    for N in n_values:
        fig,axes=plt.subplots(1,len(ell_values),figsize=(4.8*len(ell_values),4.5),sharey=True)
        for j,(ax,ell) in enumerate(zip(axes,ell_values)):
            for model in names:
                y=[]
                for h in horizons:
                    row=next(z for z in metrics if z['N']==N and z['method']==model and
                             z['horizon_steps']==h and abs(z['ell']-float(ell))<1e-6)
                    y.append(row['relative_l2'])
                ax.plot(horizons,y,marker='o',label=model)
            ax.set(title=f'ell={ell:.3g}',xlabel='Forecast horizon (saved steps)')
            ax.grid(alpha=.25)
            if j==0:ax.set_ylabel('Future residual relative L2 error')
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Causal FUTURE residual forecasts — N={N}',fontsize=13)
        fig.tight_layout();fig.savefig(folder/f'future_error_by_horizon_N{N}.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,len(horizons),figsize=(5*len(horizons),4.5),sharey=True)
    axs=np.atleast_1d(axs)
    for ax,h in zip(axs,horizons):
        for model in models:
            rows=[v for v in curves if v['method']==model and v['horizon_steps']==h]
            ax.semilogy([v['epoch'] for v in rows],[v['validation_mse_normalized'] for v in rows],label=model)
        ax.set(xlabel='Epoch',title=f'h={h} saved steps')
        ax.grid(alpha=.25)
    axs[0].set_ylabel('Validation normalized MSE')
    axs[0].legend(fontsize=8)
    fig.suptitle('Held-out validation curves')
    fig.tight_layout();fig.savefig(folder/'validation_curves.png',dpi=170);plt.close(fig)
    for N in n_values:
        fig,ax=plt.subplots(figsize=(8.5,5))
        for h in horizons:
            mark=next(z for z in metrics if z['N']==N and z['method']=='markov_fno' and z['horizon_steps']==h and z['ell']==float(ell_values[-1])) if 'markov_fno' in models else None
            his=next((z for z in metrics if z['N']==N and z['method']=='history_fno' and z['horizon_steps']==h and z['ell']==float(ell_values[-1])),None)
            if mark and his:
                improvement=100*(1-his['rmse']/max(mark['rmse'],1e-12))
                ax.bar(str(h),improvement,label=f'h={h}')
        ax.axhline(0,color='black',lw=.8)
        ax.set(xlabel='Forecast horizon (saved steps)',ylabel='History FNO error reduction vs current-state FNO (%)',
               title=f'History benefit at ell={float(ell_values[-1]):g}, N={N}')
        ax.grid(axis='y',alpha=.25)
        fig.tight_layout();fig.savefig(folder/f'history_gain_N{N}.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--regime',choices=('control','heterogeneous','multirate'),default='multirate')
    parser.add_argument('--history',type=int,default=4)
    parser.add_argument('--min-history',type=int,default=16)
    parser.add_argument('--horizons',default='1,5,10')
    parser.add_argument('--models',default=','.join(MODELS))
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--width',type=int,default=24)
    parser.add_argument('--modes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    horizons=sorted(set(int(x.strip()) for x in args.horizons.split(',') if x.strip()))
    models=[x.strip() for x in args.models.split(',') if x.strip()]
    if not horizons or min(horizons)<1:raise ValueError('Horizon must be positive integer(s)')
    if not models or any(m not in MODELS for m in models):raise ValueError('Invalid --models option')
    if args.history<4:raise ValueError('History must be >=4 (BDF2-linear baseline uses four snapshots)')
    if args.min_history<args.history:raise ValueError('min-history must be >= history')
    if args.epochs<1 or args.batch_size<1:raise ValueError('Epochs and batch size must be positive')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    np.random.seed(args.seed);torch.manual_seed(args.seed);random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu'))
    if not args.dataset.is_file():raise FileNotFoundError(args.dataset)
    with np.load(args.dataset,allow_pickle=False) as npz:
        d={k:npz[k] for k in npz.files}
    needed=('times','ell','base_id','variant','n_values',f'{args.regime}_u_N64',f'{args.regime}_r_N64',f'{args.regime}_u_N128',f'{args.regime}_r_N128')
    for k in needed:
        if k not in d:raise ValueError(f'Missing dataset key: {k}')
    times=np.asarray(d['times'],dtype=np.float64)
    dt=float(times[1]-times[0])
    if not np.allclose(np.diff(times),dt,rtol=1e-5,atol=1e-7):raise ValueError('Uniform saved times required')
    ns=sorted(map(int,d['n_values']))
    if ns!=[64,128]:raise ValueError(f'Expected N64 and N128, found {ns}')
    cutoff_start=max(args.min_history-1,3)
    cutoff_end=len(times)-1-max(horizons)
    if cutoff_end<cutoff_start:raise ValueError('Not enough saved snapshots for requested horizons/history')
    # Validate paired family organization; paired samples must never be split across train/test.
    for bid in np.unique(d['base_id']):
        ix=np.flatnonzero(d['base_id']==bid)
        if len(ix)!=2 or set(d['variant'][ix].tolist())!={-1,1}:
            raise ValueError('Expected one pair of (-1,+1) variants per base ID')
    splits=split_base_ids(d['base_id'],args.seed)
    trainmask=np.isin(d['base_id'],splits['train'])
    u_train=d[f'{args.regime}_u_N64'][trainmask]
    r_train=d[f'{args.regime}_r_N64'][trainmask]
    xstd=max(float(np.std(u_train)),1e-6)
    ystd=np.maximum(np.sqrt(np.mean(r_train**2,axis=(0,2,3))),1e-5).astype(np.float32)
    output=args.outdir/args.regime/f'h{args.history}'
    output.mkdir(parents=True,exist_ok=True)
    ckpt=output/'checkpoints';ckpt.mkdir(exist_ok=True)
    rows=[];case_rows=[];curves=[];ablations=[]
    chosen_ell=float(d['ell'][-1]); example_plot_data=None
    print(f'[CONFIG] device={device} regime={args.regime} history={args.history} horizons={horizons}',flush=True)
    print(f'[SPLIT] base families train/val/test={list(map(len,splits.values()))}; cutoff t={times[cutoff_start]:.3f} to {times[cutoff_end]:.3f}',flush=True)
    for horizon in horizons:
        print(f'\n[HORIZON] {horizon} steps = {horizon*dt:.3f} physical time',flush=True)
        samples={(N,part):make_examples(d,args.regime,N,splits[part],args.history,cutoff_start,cutoff_end,
                                        horizon,xstd,ystd)
                 for N in ns for part in ('train','val','test')}
        for N in ns:
            sample=samples[(N,'test')]
            meta=sample[3]; ref=sample[1]*ystd[meta[:,2],None]
            for method, prediction in numeric_baselines(d,args.regime,N,meta,horizon).items():
                append_metrics(rows,case_rows,args.regime,N,horizon,method,ref,prediction,meta,d['ell'],dt)
        for mi,name in enumerate(models):
            torch.manual_seed(args.seed+100*mi+horizon)
            random.seed(args.seed+100*mi+horizon)
            model=ForecastNet(name,args.history,args.width,args.modes).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
            training=make_loader(samples[(64,'train')],args.batch_size,True,args.seed+mi+horizon)
            validation=make_loader(samples[(64,'val')],args.batch_size,False,args.seed)
            best=float('inf');best_epoch=0;best_state=copy.deepcopy(model.state_dict());stale=0
            for epoch in range(1,args.epochs+1):
                tr=epoch_mse(model,training,device,optimizer)
                val=epoch_mse(model,validation,device)
                curves.append({'regime':args.regime,'method':name,'horizon_steps':horizon,'epoch':epoch,
                               'train_mse_normalized':tr,'validation_mse_normalized':val})
                if epoch==1 or epoch%5==0:print(f'  {name:12s} epoch {epoch:03d} train={tr:.5e} val={val:.5e}',flush=True)
                if val<best-1e-7:
                    best=val;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
                else:
                    stale+=1
                    if stale>=args.patience:break
            model.load_state_dict(best_state)
            torch.save({'state_dict':model.state_dict(),'model':name,'history':args.history,
                        'horizon_steps':horizon,'horizon_time':horizon*dt,'regime':args.regime,
                        'train_N':64,'xstd':xstd,'ystd':ystd.tolist(),'best_epoch':best_epoch,
                        'best_validation_mse':best},ckpt/f'{name}_h{horizon}.pt')
            for N in ns:
                sample=samples[(N,'test')]
                meta=sample[3]
                ref=sample[1]*ystd[meta[:,2],None]
                prediction=network_predict(model,sample,ystd,device,args.batch_size)
                append_metrics(rows,case_rows,args.regime,N,horizon,name,ref,prediction,meta,d['ell'],dt)
                if name!='markov_fno':
                    erased=network_predict(model,sample,ystd,device,args.batch_size,replace_history=True)
                    append_metrics(rows,case_rows,args.regime,N,horizon,name+'_history_erased',ref,erased,meta,d['ell'],dt)
                    m=meta[:,2]==len(d['ell'])-1
                    full=np.sqrt(np.mean((prediction[m]-ref[m])**2))
                    erased_mse=np.sqrt(np.mean((erased[m]-ref[m])**2))
                    ablations.append({'N':N,'horizon_steps':horizon,'model':name,
                                      'ell':chosen_ell,'original_rmse':float(full),
                                      'erased_history_rmse':float(erased_mse),
                                      'erased_worsening_pct':float(100*(erased_mse/max(full,1e-12)-1))})
                if (name=='history_fno' and N==64 and horizon==max(horizons)):
                    jj=len(d['ell'])-1
                    selected=np.flatnonzero(meta[:,2]==jj)
                    if len(selected):
                        k=int(selected[len(selected)//2]);example_plot_data=(ref[k],prediction[k],sample[3][k])
        del samples
    save_csv(output/'future_test_metrics.csv',rows)
    save_csv(output/'future_case_metrics.csv',case_rows)
    save_csv(output/'future_train_curves.csv',curves)
    save_csv(output/'history_erasure.csv',ablations)
    create_plots(output,rows,curves,d['ell'],horizons,models,ns)
    if example_plot_data is not None:
        target, predicted, md=example_plot_data
        fig,ax=plt.subplots(figsize=(9,4.3))
        xx=(np.arange(len(target))+.5)/len(target)
        ax.plot(xx,target,label='True future residual',lw=2)
        ax.plot(xx,predicted,label='History FNO future forecast',lw=1.6)
        ax.set(xlabel='x',ylabel='future residual forcing',title=f'Example: cutoff index {md[3]}, forecast index {md[4]}')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'example_future_residual.png',dpi=180);plt.close(fig)
    cfg={'regime':args.regime,'device':str(device),'dataset':str(args.dataset),'training_resolution':64,
         'evaluation_resolutions':ns,'history':args.history,'min_history':args.min_history,
         'horizons_steps':horizons,'time_step':dt,'common_cutoff_index_window':[cutoff_start,cutoff_end],
         'common_cutoff_time_window':[float(times[cutoff_start]),float(times[cutoff_end])],
         'base_family_splits':{k:v.tolist() for k,v in splits.items()},'models':models,'baselines':list(BASELINES),
         'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
         'width':args.width,'modes':args.modes,'seed':args.seed,
         'notes':['No target state or residual after the cutoff is an input feature.',
                  'Output is future residual forcing under true past observations, not an autonomous state rollout.',
                  'History-erasure is a distribution-shift diagnostic, not standalone proof of physical memory.',
                  'BDF2 linear extrapolation is not always stable for distant forecast horizons.',
                  'Data source is the prior finite-volume pilot, not FreeFEM++/SUPG.']}
    (output/'run_config.json').write_text(json.dumps(cfg,indent=2))
    print('\n[DONE]',output,flush=True)
    print('Important: future residual regression is NOT an autonomous PDE rollout.',flush=True)


if __name__=='__main__':
    main()