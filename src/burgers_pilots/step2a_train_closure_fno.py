#!/usr/bin/env python3
"""SMTNO Step 2A — scale-conditioned closure learning, current-state vs short-history FNO.

Input: MyDrive/SMTNO/Colab Notebooks/datasets/burgers_step1F_ensemble.npz
Output: MyDrive/SMTNO/Colab Notebooks/results/step2A and checkpoints/step2A

Run from Colab after drive.mount('/content/drive'):
    !python "/content/drive/MyDrive/SMTNO/Colab Notebooks/step2a_train_closure_fno.py" --epochs 60

Trains on N=64 only, evaluates on N=64 held-out trajectories and N=128
zero-shot as a separate diagnostic. A learned weak closure must conserve mass;
we explicitly project predicted weak forcing onto the zero-sum subspace.

This is a pilot neural *operator* baseline, NOT the full SMTNO architecture.
No closure is used in a PDE rollout during this stage. No claim of memory
necessity follows from a single test split.
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
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--history', type=int, default=4, help='Current + previous snapshots, default 4')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--modes', type=int, default=12)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--patience', type=int, default=14)
    p.add_argument('--project-dir', type=Path, default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    p.add_argument('--dataset', type=Path, default=None)
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


class SpectralConv1d(nn.Module):
    def __init__(self, cin:int, cout:int, modes:int):
        super().__init__()
        self.cin,self.cout,self.modes=cin,cout,modes
        scale=1/(cin*cout)
        self.weights=nn.Parameter(scale*torch.randn(cin,cout,modes,dtype=torch.cfloat))

    def forward(self, x):
        # Input (batch, channels, periodic nodes)
        n=x.shape[-1]
        xft=torch.fft.rfft(x,dim=-1)
        nkeep=min(self.modes,xft.shape[-1])
        yft=torch.zeros(x.shape[0], self.cout, xft.shape[-1],
                        dtype=xft.dtype, device=xft.device)
        yft[:,:,:nkeep]=torch.einsum('bim,iom->bom',
                                  xft[:,:,:nkeep], self.weights[:,:,:nkeep])
        return torch.fft.irfft(yft,n=n,dim=-1)


class ClosureFNO(nn.Module):
    def __init__(self, in_channels:int, width:int=32, modes:int=12, layers:int=4):
        super().__init__()
        self.lift=nn.Linear(in_channels,width)
        self.spec=nn.ModuleList([SpectralConv1d(width,width,modes) for _ in range(layers)])
        self.local=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(layers)])
        self.head=nn.Sequential(nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,1,1))

    def forward(self,x):
        # x: (batch, periodic nodes, time-history + ell)
        z=self.lift(x).transpose(1,2)
        for s,l in zip(self.spec,self.local):
            z=torch.nn.functional.gelu(s(z)+l(z))
        f=self.head(z).squeeze(1)
        return f-f.mean(dim=-1,keepdim=True) # weak forcing has exactly zero sum


def make_examples(d, indices, N, history, x_std, target_std):
    """Same eligible t>=history-1 in both models, with no trajectory overlap.
    Inputs are resolved *projected FEM states* only, no oracle tau leakage.
    Labels: weak forcing S_i / h = N * S_i, standardized per physical ell.
    """
    u=d[f'coarse_N{N}']
    forcing=d[f'S_tau_N{N}']
    ell=d['ell']
    xout=[]; yout=[]; jlist=[]; cmap=[]; time_idx=[]
    T=u.shape[2]
    for case_idx in indices:
        for j in range(len(ell)):
            U=u[case_idx,j]
            S=forcing[case_idx,j]
            for t in range(history-1,T):
                history_states=np.stack([U[t-k] for k in range(history)],axis=-1)
                scale_channel=np.full((N,1),ell[j]/float(max(ell)),dtype=np.float32)
                X=np.concatenate([history_states/x_std,scale_channel],axis=-1).astype(np.float32)
                Y=(S[t]*N/target_std[j]).astype(np.float32)
                xout.append(X); yout.append(Y); jlist.append(j)
                cmap.append(int(d['case_ids'][case_idx]));time_idx.append(t)
    return (np.stack(xout),np.stack(yout),np.array(jlist),np.array(cmap),np.array(time_idx))


def baseline_gradient_from_X(X, ell, x_std, N):
    """Clark gradient approximation for 1/2(P[u²] - (P[u])²) on periodic grid:
       tau ≈ ell²/2 * (d ubar / dx)², weak force from edge flux differences.
       It is an existing physics baseline (not learned memory).
    """
    u=X[...,0]*x_std
    dx=(np.roll(u,-1,axis=-1)-np.roll(u,1,axis=-1))*(N/2)
    tau=0.5*(ell[:,None]**2)*(dx**2)
    edge=0.5*(tau+np.roll(tau,-1,axis=-1))
    S=np.roll(edge,1,axis=-1)-edge
    return N*S # return physical weak forcing / h


def make_loader(X,Y,batch_size,shuffle,seed):
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(Y))
    g=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,generator=g,num_workers=0)


def train(model,train_loader,val_loader,args,device):
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    best=np.inf; saved=None; stale=0; hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); trtot=0.; ntr=0
        for xb,yb in train_loader:
            xb=xb.to(device);yb=yb.to(device)
            opt_pred=model(xb)
            loss=torch.mean((opt_pred-yb)**2)
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step()
            trtot+=loss.item()*len(xb);ntr+=len(xb)
        model.eval();vtot=0.;nv=0
        with torch.no_grad():
            for xb,yb in val_loader:
                loss=torch.mean((model(xb.to(device))-yb.to(device))**2)
                vtot+=loss.item()*len(xb);nv+=len(xb)
        tr=trtot/ntr;val=vtot/nv
        hist.append({'epoch':epoch,'train_mse_normalized':tr,'val_mse_normalized':val})
        print(f'    epoch {epoch:03d}: train={tr:.6e}, val={val:.6e}',flush=True)
        if val<best-1e-8:
            best=val;saved=copy.deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=args.patience:
                print(f'    Early stopping at epoch {epoch}; best val={best:.6e}')
                break
    model.load_state_dict(saved)
    return hist,best


def predict(model,X,batch_size,device):
    model.eval();out=[]
    with torch.no_grad():
        for s in range(0,len(X),batch_size):
            xb=torch.as_tensor(X[s:s+batch_size],device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out,axis=0)


def rows_metrics(name, split, N, ystd, ypstd, jidx, ells):
    # ystd and ypstd are already physical S/h.
    out=[]
    for j,ell in enumerate(ells):
        ix=jidx==j
        ref=ystd[ix]; pred=ypstd[ix]
        err=ref-pred
        out.append({'model':name,'split':split,'mesh_N':N,'ell':float(ell),
                    'n_snapshots':int(len(ref)),
                    'rmse_forcing_per_h':float(np.sqrt(np.mean(err**2))),
                    'rel_l2':float(np.linalg.norm(err)/max(np.linalg.norm(ref),1e-12)),
                    'max_abs_sum_S':float(np.max(np.abs(np.sum(pred/N,axis=-1))))})
    return out


def workflow_figure(dst):
    txt=[
        '1G: Validated fine FEM data and oracle weak closure labels',
        'Transfer processed NPZ to Google Drive / Colab',
        'Split by complete trajectories (20 train / 5 val / 5 test)',
        'Train a scale-conditioned 1D FNO: current state only',
        'Train a matched 1D FNO: current + 3 past snapshots',
        'Compare against zero forcing and gradient/Clark closure',
        'Evaluate on held-out trajectories; probe N=128 zero-shot',
        'Next: couple predicted forcing to coarse FEM time rollouts'
    ]
    fig,ax=plt.subplots(figsize=(12,11));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,10)
    ys=np.linspace(9.15,.85,len(txt))
    ax.set_title('SMTNO Step 2A: controlled closure-learning experiment',pad=18)
    for j,(y,label) in enumerate(zip(ys,txt)):
        ax.add_patch(FancyBboxPatch((.7,y-.37),8.6,.74,facecolor='white',edgecolor='black',
                                   boxstyle='round,pad=.07',lw=1.2))
        ax.text(5,y,label,ha='center',va='center',fontsize=10)
        if j<len(ys)-1:
            ax.annotate('',xy=(5,ys[j+1]+.47),xytext=(5,y-.47),arrowprops={'arrowstyle':'->','lw':1.2})
    fig.savefig(dst,dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    args=parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu: device=torch.device('cuda')
    elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() and not args.cpu:
        device=torch.device('mps')
    else: device=torch.device('cpu')
    print(f'[CONFIG] device={device}, epochs={args.epochs}, history={args.history}',flush=True)
    project=args.project_dir
    file=args.dataset or (project/'datasets'/'burgers_step1F_ensemble.npz')
    if not file.is_file():
        raise FileNotFoundError(f'Missing {file}. Copy it from Mac Step 1F into this Drive datasets folder.')
    out=project/'results'/'step2A';ckpt=project/'checkpoints'/'step2A'
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    with np.load(file,allow_pickle=False) as f:
        d={key:f[key] for key in f.files if key in ('ell','times','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128')}
    needed=('ell','times','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128')
    missing=set(needed)-set(d)
    if missing:raise ValueError(f'Missing NPZ keys: {missing}')
    for n in (64,128):
        assert d[f'coarse_N{n}'].shape==d[f'S_tau_N{n}'].shape
    assert d['coarse_N64'].shape[2]>=args.history
    split=d['split'].astype(str)
    train_ids=np.flatnonzero(split=='train');val_ids=np.flatnonzero(split=='val');test_ids=np.flatnonzero(split=='test')
    assert tuple(map(len,(train_ids,val_ids,test_ids)))==(20,5,5), 'Expected 20/5/5 disjoint trajectory split'
    x_std=float(np.std(d['coarse_N64'][train_ids]))
    x_std=max(x_std,1e-7)
    label_scales=[]
    for j in range(len(d['ell'])):
        target=d['S_tau_N64'][train_ids,j]*64
        label_scales.append(max(float(np.sqrt(np.mean(target**2))),1e-7))
    label_scales=np.array(label_scales,dtype=np.float32)
    print(f'[DATA] x_std={x_std:.4g}; target scales per ell={label_scales}; splits 20/5/5',flush=True)

    # Both models use equal eligible times: current-history indices start at history-1.
    data={}
    for N in (64,128):
        for nam,ix in [('train',train_ids),('val',val_ids),('test',test_ids)]:
            data[(N,nam)]=make_examples(d,ix,N,args.history,x_std,label_scales)

    tracks={}; predictions={};metrics=[]
    ells=d['ell']
    for model_name, num_hist in [('memoryless',1),('short_history',args.history)]:
        print(f'\n[TRAIN] {model_name}',flush=True)
        # For fair time eligibility, crop the past channels but always keep ell column.
        def trim(a):
            return np.concatenate((a[...,:num_hist],a[...,-1:]),axis=-1)
        Xtr,Ytr,*_=data[(64,'train')];Xval,Yval,*_=data[(64,'val')]
        train_loader=make_loader(trim(Xtr),Ytr,args.batch_size,True,args.seed)
        val_loader=make_loader(trim(Xval),Yval,args.batch_size,False,args.seed)
        net=ClosureFNO(num_hist+1,width=args.width,modes=args.modes).to(device)
        his,best=train(net,train_loader,val_loader,args,device)
        tracks[model_name]=his
        torch.save({'state_dict':net.state_dict(),'model':model_name,'history':num_hist,
                    'x_std':x_std,'target_scales':label_scales.tolist(),'ell':ells.tolist(),
                    'best_validation':best,'training_resolution':64,
                    'note':'Predicts conservative FEM weak forcing divided by h; scale-wise target standardization.'},
                   ckpt/f'{model_name}.pt')
        for N in (64,128):
            X,Y,J,cases,tt=data[(N,'test')]
            pred_normalized=predict(net,trim(X),args.batch_size,device)
            ref_physical=Y*label_scales[J,None]
            pred_physical=pred_normalized*label_scales[J,None]
            predictions[(model_name,N)]=pred_physical
            metrics.extend(rows_metrics(model_name, 'test' if N==64 else 'test_zero_shot',N,
                                        ref_physical,pred_physical,J,ells))

    # Physical baselines at N64 and zero-shot N128.
    for N in (64,128):
        X,Y,J,_,_=data[(N,'test')]
        ref_physical=Y*label_scales[J,None]
        for name,pred in [('zero_closure',np.zeros_like(ref_physical)),
                          ('clark_gradient',baseline_gradient_from_X(X,ells[J],x_std,N))]:
            predictions[(name,N)]=pred
            metrics.extend(rows_metrics(name,'test' if N==64 else 'test_zero_shot',N,
                                        ref_physical,pred,J,ells))

    with (out/'closure_test_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=metrics[0].keys());writer.writeheader();writer.writerows(metrics)
    with (out/'training_losses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','epoch','train_mse_normalized','val_mse_normalized']);w.writeheader()
        for key,rows in tracks.items():
            for r in rows:w.writerow({'model':key,**r})

    fig,ax=plt.subplots(figsize=(8,5))
    for name,records in tracks.items():
        ax.semilogy([x['epoch'] for x in records],[x['val_mse_normalized'] for x in records],label=name)
    ax.set(xlabel='epoch',ylabel='validation MSE (normalized per scale)',title='Step 2A: FNO validation losses')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'validation_learning_curves.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,N in zip(axes,(64,128)):
        names=['zero_closure','clark_gradient','memoryless','short_history']
        x=np.arange(len(ells));width=.19
        for jj,name in enumerate(names):
            vals=[next(row['rel_l2'] for row in metrics if row['model']==name and row['mesh_N']==N and abs(row['ell']-e)<1e-8) for e in ells]
            ax.bar(x+(jj-1.5)*width,vals,width=width,label=name)
        ax.set_xticks(x,[f'{e:g}' for e in ells]);ax.set_title(f'N={N} '+('in-resolution' if N==64 else 'zero-shot resolution'))
        ax.set_xlabel('Physical filtering width ell');ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Relative L2 forcing error')
    axes[0].legend(fontsize=8)
    fig.suptitle('Step 2A: held-out trajectory forcing predictions')
    fig.tight_layout();fig.savefig(out/'test_closure_errors.png',dpi=180);plt.close(fig)

    # Example: first test case at ell=0.04 and final snapshot.
    N=64;X,Y,J,cases,tt=data[(N,'test')]
    chosen=np.flatnonzero((J==len(ells)-1)&(cases==cases[0])&(tt==tt.max()))
    if len(chosen):
        idx=int(chosen[0]);ref=Y[idx]*label_scales[J[idx]]
        fig,ax=plt.subplots(figsize=(9,5));x=np.arange(N)/N
        ax.plot(x,ref,label='oracle weak forcing / h',lw=2)
        for name in ('clark_gradient','memoryless','short_history'):
            ax.plot(x,predictions[(name,N)][idx],label=name)
        ax.set(xlabel='x',ylabel='weak forcing / h',title='Step 2A: one held-out closure forcing field')
        ax.legend();ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(out/'example_forcing_prediction.png',dpi=180);plt.close(fig)
    workflow_figure(out/'step2A_workflow.png')
    (out/'training_config.json').write_text(json.dumps({
        'device':str(device),'num_train_cases':20,'num_val_cases':5,'num_test_cases':5,
        'time_history_steps':args.history,'saved_dt':float(d['times'][1]-d['times'][0]),
        'training_resolution':64,'zero_shot_resolution':128,'target_scales':label_scales.tolist(),
        'model':'periodic Fourier neural operator; zero-sum conservative weak forcing output',
        'limitation':'One-step oracle forcing regression only. No closed-loop rollout or proven memory benefit.'
    },indent=2))
    print(f'\n[DONE] Results: {out}\n[DONE] Checkpoints: {ckpt}',flush=True)
    print('Read closure_test_metrics.csv before deciding whether history helps.')


if __name__=='__main__':main()