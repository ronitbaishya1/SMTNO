#!/usr/bin/env python3
"""SMTNO Step 2D.1 — FNO-based latent memory, multi-step FEM training, and transfer ablations."""
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


def gaussian(x, old, new):
    if new + 1e-12 < old:
        raise ValueError('This experiment only supports increasing filter widths')
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, d=1.0/n, device=x.device)
    factor = torch.exp(-0.5 * (2*np.pi*f)**2 * (new*new-old*old))
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1)*factor, n=n, dim=-1)


class Spectral1D(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(channels, channels, modes, dtype=torch.cfloat)/channels)

    def forward(self, x):
        n = x.shape[-1]
        f = torch.fft.rfft(x, dim=-1)
        keep = min(self.modes, f.shape[-1])
        out = torch.zeros_like(f)
        out[:, :, :keep] = torch.einsum('bim,iom->bom', f[:, :, :keep], self.weights[:, :, :keep])
        return torch.fft.irfft(out, n=n, dim=-1)


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.fourier = Spectral1D(width, modes)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return torch.nn.functional.gelu(self.fourier(x)+self.pointwise(x))


class ScaleMemoryFNO(nn.Module):
    def __init__(self, hidden, width, modes, xstd):
        super().__init__()
        self.hidden = hidden
        self.xstd = xstd
        self.lift = nn.Conv1d(hidden+2, width, 1)
        self.blocks = nn.Sequential(FNOBlock(width, modes), FNOBlock(width, modes))
        self.gate = nn.Conv1d(width, hidden, 1)
        self.candidate = nn.Conv1d(width, hidden, 1)
        self.head = nn.Sequential(nn.Conv1d(width+hidden, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))
        self.transfer_lift = nn.Conv1d(hidden+3, width, 1)
        self.transfer_block = FNOBlock(width, modes)
        self.transfer_head = nn.Conv1d(width, hidden, 1)

    def initial_memory(self, u):
        return u.new_zeros(u.shape[0], self.hidden, u.shape[-1])

    def step(self, u, mem, ell):
        b, n = u.shape
        ee = torch.ones((b, 1, n), dtype=u.dtype, device=u.device)*(ell/0.04)
        z = torch.cat([mem, (u/self.xstd)[:,None,:], ee], dim=1)
        f = self.blocks(self.lift(z))
        gate = torch.sigmoid(self.gate(f))
        mem = (1-gate)*mem + gate*torch.tanh(self.candidate(f))
        raw = self.head(torch.cat([f, mem], dim=1)).squeeze(1)
        return mem, raw-raw.mean(dim=-1, keepdim=True)

    def transfer(self, mem, new_u, old, new, mode):
        if mode == 'reset':
            return torch.zeros_like(mem)
        mg = gaussian(mem, old, new)
        if mode == 'gaussian':
            return mg
        if mode != 'learned':
            raise ValueError(mode)
        b, _, n = mg.shape
        o = new_u.new_full((b, 1, n), old/0.04)
        e = new_u.new_full((b, 1, n), new/0.04)
        z = torch.cat([mg, (new_u/self.xstd)[:,None,:], o, e], dim=1)
        correction = torch.tanh(self.transfer_head(self.transfer_block(self.transfer_lift(z))))
        return mg + 0.15*correction


def mass(u):
    n = u.shape[-1]
    return (4*u+torch.roll(u,1,-1)+torch.roll(u,-1,-1))/(6*n)


def convective_rhs(u):
    v = torch.roll(u,-1,-1)
    flux = (u*u+u*v+v*v)/6
    return torch.roll(flux,1,-1)-flux


def fem_step(u, s, dt, nu):
    # Differentiate through periodic P1 mass/stiffness FEM and a semi-implicit Euler step.
    n = u.shape[-1]
    h = 1/n
    theta = 2*torch.pi*torch.fft.rfftfreq(n, d=1, device=u.device, dtype=u.dtype)
    mass_diag = h*(2/3+torch.cos(theta)/3)
    stiffness = (2-2*torch.cos(theta))/h
    rhs = mass(u)/dt+convective_rhs(u)+s
    return torch.fft.irfft(torch.fft.rfft(rhs, dim=-1)/(mass_diag/dt+nu*stiffness), n=n, dim=-1)


def rel_l2(u, ref):
    delta = u-ref
    return torch.sqrt(torch.clamp((delta*mass(delta)).sum(-1), min=0) /
                      torch.clamp((ref*mass(ref)).sum(-1), min=1e-14))


def advance(u, prediction, ystd, ell_index, substeps, saved_dt, nu):
    n = u.shape[-1]
    forcing = (prediction.to(torch.float64)*(ystd[ell_index]/n))
    forcing = forcing-forcing.mean(dim=-1, keepdim=True)
    u64 = u.to(torch.float64)
    for _ in range(substeps):
        u64 = fem_step(u64, forcing, saved_dt/substeps, nu)
    return u64


def make_batch_loss(model, U, Y, ids, ells, ystd, times, args):
    # Whole trajectories, no overlapping IDs across train, validation and test.
    n_times = U.shape[2]
    warm = 4
    horizon = args.horizon
    start = random.randrange(0, n_times-warm-horizon)
    old,new = random.choice([(0,1),(0,2),(1,2)])
    switching = random.choice([True, True, False])
    switch_k = random.randint(1, horizon-2) if switching else -1
    u = U[ids,old,start+warm-1].to(torch.float64)
    mem = model.initial_memory(u.to(torch.float32))
    # Warm up recurrent memory on four observed snapshots.
    for t in range(start,start+warm):
        mem,_ = model.step(U[ids,old,t], mem, float(ells[old]))
    loss_f = u.new_zeros(())
    loss_u = u.new_zeros(())
    j = old
    for k in range(horizon):
        idx = start+warm-1+k
        if k == switch_k:
            u = gaussian(u, float(ells[old]), float(ells[new]))
            mem = model.transfer(mem, u.to(torch.float32), float(ells[old]), float(ells[new]), 'learned')
            j = new
        mem, pred = model.step(u.to(torch.float32), mem, float(ells[j]))
        loss_f = loss_f + ((pred-Y[ids,j,idx])**2).mean()
        u = advance(u, pred, ystd, j, args.substeps, float(times[1]-times[0]), args.nu)
        ref = U[ids,j,idx+1]
        loss_u = loss_u + ((u-ref)/model.xstd).square().mean()
    return loss_f/horizon + args.rollout_weight*loss_u/horizon


@torch.no_grad()
def validation_loss(model, U, Y, ids, ells, ystd, times, args):
    # Identical deterministic validation schedule for every checkpoint.
    model.eval()
    errs = []
    for old,new in [(0,2),(1,2)]:
        warm=4
        m=model.initial_memory(U[ids,old,0])
        for t in range(warm):
            m,_=model.step(U[ids,old,t],m,float(ells[old]))
        u=U[ids,old,warm-1].to(torch.float64)
        j=old
        switch_k=min(4, max(args.horizon-2, 1))
        for k in range(min(args.horizon, U.shape[2]-warm)):
            idx=warm-1+k
            if k==switch_k:
                u=gaussian(u,float(ells[old]),float(ells[new]))
                m=model.transfer(m,u.to(torch.float32),float(ells[old]),float(ells[new]),'learned')
                j=new
            m,p=model.step(u.to(torch.float32),m,float(ells[j]))
            e=((p-Y[ids,j,idx])**2).mean()
            u=advance(u,p,ystd,j,args.substeps,float(times[1]-times[0]),args.nu)
            errs.append(float(e+args.rollout_weight*((u-U[ids,j,idx+1])/model.xstd).square().mean()))
    return float(np.mean(errs))


@torch.no_grad()
def evaluate(model, U, Y, ids, ell, ystd, times, args, mode):
    # Same observed four snapshots as Step 2B/2C; autonomous afterward.
    model.eval()
    times_np = np.asarray(times)
    switch = int(np.argmin(np.abs(times_np-args.switch_time)))
    if not np.isclose(times_np[switch],args.switch_time):
        raise ValueError('Switch time must equal a saved dataset time')
    old,new = 0,2
    mem=model.initial_memory(U[ids,old,0])
    for t in range(4):
        mem,_=model.step(U[ids,old,t],mem,float(ell[old]))
    u=U[ids,old,3].to(torch.float64).clone()
    mean_initial=u.to(torch.float64).mean(-1)
    records=[]
    for t in range(3,len(times_np)):
        j=old if t<switch else new
        if t==switch:
            u=gaussian(u,float(ell[old]),float(ell[new]))
            mem=model.transfer(mem,u.to(torch.float32),float(ell[old]),float(ell[new]),mode)
        err=rel_l2(u.to(torch.float64),U[ids,j,t].to(torch.float64))
        drift=(u.to(torch.float64).mean(-1)-mean_initial).abs()
        for i,cid in enumerate(ids):
            records.append({'mode':mode,'N':u.shape[-1],'case_index':int(cid),
                            'ell':float(ell[j]),'time':float(times_np[t]),
                            'relative_error':float(err[i]),'absolute_mass_drift':float(drift[i])})
        if t==len(times_np)-1: break
        mem,p=model.step(u.to(torch.float32),mem,float(ell[j]))
        u=advance(u,p,ystd,j,args.substeps,float(times_np[1]-times_np[0]),args.nu)
    return records


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def figures(out, history, records, switch_time):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.semilogy([r['epoch'] for r in history],[r['train_loss'] for r in history],label='train')
    ax.semilogy([r['epoch'] for r in history],[r['validation_loss'] for r in history],label='validation')
    ax.set(xlabel='epoch',ylabel='combined loss',title='2D.1: FNO-memory training');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for ax,n in zip(axs,[64,128]):
        for mode in ['learned','gaussian','reset']:
            rows=[r for r in records if r['N']==n and r['mode']==mode]
            ts=sorted(set(r['time'] for r in rows))
            v=[np.mean([r['relative_error'] for r in rows if r['time']==t]) for t in ts]
            ax.plot(ts,v,label=mode)
        ax.axvline(switch_time,ls='--',color='black',lw=1)
        ax.set(xlabel='time',ylabel='mean relative FEM error',title=f'N={n}')
        ax.legend();ax.grid(alpha=.25)
    fig.suptitle('2D.1: identical warm start, scale-switch rollouts')
    fig.tight_layout();fig.savefig(out/'rollout_ablation.png',dpi=170);plt.close(fig)

    summary=[]
    final=max(r['time'] for r in records)
    for n in [64,128]:
        for mode in ['learned','gaussian','reset']:
            rs=[r for r in records if r['N']==n and r['mode']==mode]
            fs=[r['relative_error'] for r in rs if abs(r['time']-final)<1e-10]
            ds=[r['absolute_mass_drift'] for r in rs]
            summary.append({'N':n,'transfer_mode':mode,'mean_final_error':float(np.mean(fs)),
                            'max_mass_drift':float(max(ds))})
    write_csv(out/'ablation_summary.csv',summary)

    fig,ax=plt.subplots(figsize=(7,3.5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    labels=['FEM data + train/validation split','FNO spatial backbone + recurrent memory',
            'Teacher warm-up + multi-step FEM loss','Gaussian state transition + learned memory transfer',
            'Ablate learned / Gaussian / reset (N=64,128)']
    ys=np.linspace(.9,.1,len(labels))
    for i,(y,l) in enumerate(zip(ys,labels)):
        ax.text(.5,y,l,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.5',fc='white',ec='black'))
        if i<len(labels)-1: ax.annotate('',xy=(.5,ys[i+1]+.04),xytext=(.5,y-.04),
                                      arrowprops={'arrowstyle':'->'})
    fig.savefig(out/'step2D1_workflow.png',dpi=170,bbox_inches='tight');plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-dir',type=Path,default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    p.add_argument('--dataset',type=Path,default=None)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--updates-per-epoch',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--horizon',type=int,default=6)
    p.add_argument('--substeps',type=int,default=20)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--hidden',type=int,default=12)
    p.add_argument('--modes',type=int,default=12)
    p.add_argument('--lr',type=float,default=4e-4)
    p.add_argument('--rollout-weight',type=float,default=40.)
    p.add_argument('--switch-time',type=float,default=.1)
    p.add_argument('--nu',type=float,default=.02)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    path=a.dataset or a.project_dir/'datasets'/'burgers_step1F_ensemble.npz'
    if not path.exists(): raise FileNotFoundError(f'Dataset missing: {path}')
    out=a.project_dir/'results'/'step2D1';out.mkdir(parents=True,exist_ok=True)
    ck=a.project_dir/'checkpoints'/'step2D1';ck.mkdir(parents=True,exist_ok=True)
    with np.load(path,allow_pickle=False) as f:
        d={k:f[k] for k in ('coarse_N64','coarse_N128','S_tau_N64','S_tau_N128','ell','times','split','case_ids')}
    ell=d['ell'].astype(float);times=d['times'].astype(float)
    if not np.allclose(ell,[.01,.02,.04]):raise ValueError('Expected filter widths .01, .02, .04')
    if not np.allclose(np.diff(times),np.diff(times)[0]):raise ValueError('Saved timesteps must be uniform')
    if len(times)<=a.horizon+4:raise ValueError('Need more time snapshots than warm-up + horizon')
    split=d['split'].astype(str)
    train=np.where(split=='train')[0];val=np.where(split=='val')[0];test=np.where(split=='test')[0]
    if len(train)==0 or len(val)==0 or len(test)==0:raise ValueError('Missing trajectory split')
    xstd=max(float(np.std(d['coarse_N64'][train])),1e-7)
    ystd=np.array([max(float(np.sqrt(np.mean((d['S_tau_N64'][train,j]*64)**2))),1e-7)
                   for j in range(3)])
    U={n:torch.as_tensor(d[f'coarse_N{n}'],device=device,dtype=torch.float32) for n in [64,128]}
    Y={n:torch.as_tensor(d[f'S_tau_N{n}']*n/ystd[None,:,None,None],device=device,dtype=torch.float32)
       for n in [64,128]}
    model=ScaleMemoryFNO(a.hidden,a.width,a.modes,xstd).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    best=float('inf');best_state=None;history=[]
    print('Device:',device,'  train N=64, evaluate N=64 & N=128',flush=True)
    for ep in range(1,a.epochs+1):
        model.train();losses=[]
        for _ in range(a.updates_per_epoch):
            ids=np.random.choice(train,size=min(len(train),a.batch_size),replace=False)
            loss=make_batch_loss(model,U[64],Y[64],ids,ell,ystd,times,a)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.item()))
        v=validation_loss(model,U[64],Y[64],val,ell,ystd,times,a)
        history.append({'epoch':ep,'train_loss':float(np.mean(losses)),'validation_loss':v})
        print(f'epoch {ep:03d}: train={np.mean(losses):.5g} val={v:.5g}',flush=True)
        if v<best:
            best=v;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':best_state,'width':a.width,'hidden':a.hidden,'modes':a.modes,
                'xstd':xstd,'ystd':ystd.tolist(),'seed':a.seed,'best_val':best},
               ck/'scale_memory_fno.pt')
    write_csv(out/'training_losses.csv',history)
    records=[]
    for n in [64,128]:
        for mode in ['learned','gaussian','reset']:
            records+=evaluate(model,U[n],Y[n],test,ell,ystd,times,a,mode)
    write_csv(out/'rollout_by_case_time.csv',records)
    summary=figures(out,history,records,a.switch_time)
    (out/'run_config.json').write_text(json.dumps({
        'device':str(device),'train_N':64,'test_N':[64,128],
        'epochs':a.epochs,'horizon':a.horizon,'substeps':a.substeps,
        'switch_time':a.switch_time,'loss_weight':a.rollout_weight,
        'note':'Pilot; no novelty or superiority implied. Uses same 4-snapshot initialization for all transfer modes.'},indent=2))
    for r in summary:print(r,flush=True)
    print('Outputs:',out,flush=True)

if __name__=='__main__':main()