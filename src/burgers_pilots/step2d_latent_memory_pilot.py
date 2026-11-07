#!/usr/bin/env python3
"""Step 2D: pilot trainable coupled Gaussian-state / latent-memory transition.

No novel-method claim: this is a falsifiable architectural prototype.
It predicts conservative weak forcing, compares memory-transfer ablations,
and runs coarse P1 FEM rollouts at an actual physical-scale switch.
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
from matplotlib.patches import FancyBboxPatch


def gaussian(x, old, new):
    if new < old:
        raise ValueError('Coarsening only in this pilot')
    n = x.shape[-1]
    k = torch.fft.rfftfreq(n, d=1.0/n, device=x.device)
    f = torch.exp(-0.5*(2*np.pi*k)**2*(new**2-old**2))
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1)*f, n=n, dim=-1)


class CoupledScaleMemory(nn.Module):
    def __init__(self, hidden=12, scale=1.0):
        super().__init__()
        self.hidden = hidden
        self.xscale = scale
        self.gate = nn.Conv1d(hidden+2,hidden,5,padding=2,padding_mode='circular')
        self.candidate = nn.Conv1d(hidden+2,hidden,5,padding=2,padding_mode='circular')
        self.readout = nn.Sequential(
            nn.Conv1d(hidden+2,32,5,padding=2,padding_mode='circular'),
            nn.GELU(),nn.Conv1d(32,1,3,padding=1,padding_mode='circular'))
        self.transfer_net = nn.Sequential(
            nn.Conv1d(hidden+3,32,3,padding=1,padding_mode='circular'),
            nn.GELU(),nn.Conv1d(32,hidden,3,padding=1,padding_mode='circular'))

    def initial_memory(self,u):
        return u.new_zeros(u.shape[0],self.hidden,u.shape[-1])

    def features(self,u,m,ell):
        b,_,n=m.shape
        e=m.new_full((b,1,n),ell/0.04)
        return torch.cat([m,(u/self.xscale).unsqueeze(1),e],dim=1)

    def step(self,u,m,ell):
        f=self.features(u,m,ell)
        gate=torch.sigmoid(self.gate(f))
        m=(1-gate)*m+gate*torch.tanh(self.candidate(f))
        raw=self.readout(self.features(u,m,ell)).squeeze(1)
        return m,raw-raw.mean(-1,keepdim=True)

    def transfer(self,m,u_at_new_scale,ell0,ell1,mode):
        if mode=='reset': return torch.zeros_like(m)
        mg=gaussian(m,ell0,ell1)
        if mode=='gaussian': return mg
        if mode!='learned': raise ValueError(mode)
        b,_,n=mg.shape
        e0=mg.new_full((b,1,n),ell0/0.04)
        e1=mg.new_full((b,1,n),ell1/0.04)
        features=torch.cat([mg,(u_at_new_scale/self.xscale).unsqueeze(1),e0,e1],dim=1)
        return mg+0.15*torch.tanh(self.transfer_net(features))


def mass_apply(u):
    h=1/u.shape[-1]
    return h/6*(4*u+np.roll(u,1,axis=-1)+np.roll(u,-1,axis=-1))


def nonlinear_rhs(u):
    v=np.roll(u,-1,axis=-1)
    flux=(u*u+u*v+v*v)/6
    return np.roll(flux,1,axis=-1)-flux


def fem_step(u,s,dt,nu):
    n=u.shape[-1]; h=1/n
    th=2*np.pi*np.fft.rfftfreq(n)
    m=h*(2/3+np.cos(th)/3)
    k=(2-2*np.cos(th))/h
    rhs=mass_apply(u)/dt+nonlinear_rhs(u)+s
    return np.fft.irfft(np.fft.rfft(rhs,axis=-1)/(m/dt+nu*k),n=n,axis=-1)


def rel_l2(u,ref):
    d=u-ref
    num=np.sum(d*mass_apply(d),axis=-1)
    den=np.sum(ref*mass_apply(ref),axis=-1)
    return np.sqrt(np.maximum(num,0)/np.maximum(den,1e-14))


def batch_sequences(model,U,Y,ells,indices,window,device,training=True):
    b=len(indices)
    n=U.shape[-1]; tmax=U.shape[2]
    old=random.choice([0,1])
    new=random.choice(list(range(old,3)))
    start=random.randint(0,tmax-window)
    switch=random.randint(5,window-5)
    m=model.initial_memory(U[indices,old,start])
    loss=0.0
    for j in range(window):
        eidx=old if j<switch else new
        state=U[indices,eidx,start+j]
        if j==switch and new!=old:
            m=model.transfer(m,state,float(ells[old]),float(ells[new]),'learned')
        m,p=model.step(state,m,float(ells[eidx]))
        loss=loss+torch.mean((p-Y[indices,eidx,start+j])**2)
    return loss/window


@torch.no_grad()
def test_teacher(model,U,Y,ells,ids,switch,mode,dev):
    o=0; dst=2; tcount=U.shape[2]
    m=model.initial_memory(U[ids,o,0])
    pred=[];truth=[]
    for t in range(tcount):
        e=o if t<switch else dst
        u=U[ids,e,t]
        if t==switch: m=model.transfer(m,u,float(ells[o]),float(ells[dst]),mode)
        m,p=model.step(u,m,float(ells[e]))
        pred.append(p.cpu().numpy());truth.append(Y[ids,e,t].cpu().numpy())
    yp=np.stack(pred,1);yt=np.stack(truth,1)
    err=np.sqrt(np.mean((yp-yt)**2,axis=(0,2)))
    rel=float(np.linalg.norm((yp-yt)[:,switch:])/max(np.linalg.norm(yt[:,switch:]),1e-14))
    return rel,err


@torch.no_grad()
def test_rollout(model,U,ells,ids,switch,mode,ystds,dt_saved,substeps,nu,dev):
    # Recurrent updates use AUTONOMOUS predicted states; fine reference is scoring only.
    b=len(ids);n=U.shape[-1];tcount=U.shape[2]
    state=U[ids,0,0].detach().cpu().numpy().astype(np.float64)
    initial_mean=state.mean(axis=-1)
    mem=model.initial_memory(U[ids,0,0])
    eidx=0; errors=[];max_mass=0.
    for t in range(tcount):
        if t==switch:
            src=torch.as_tensor(state,dtype=torch.float32,device=dev)
            state=gaussian(src,float(ells[0]),float(ells[2])).cpu().numpy().astype(np.float64)
            newu=torch.as_tensor(state,dtype=torch.float32,device=dev)
            mem=model.transfer(mem,newu,float(ells[0]),float(ells[2]),mode)
            eidx=2
        ref=U[ids,eidx,t].cpu().numpy().astype(np.float64)
        errors.append(rel_l2(state,ref).mean())
        max_mass=max(max_mass,float(np.abs(state.mean(-1)-initial_mean).max()))
        if t==tcount-1: break
        xu=torch.as_tensor(state,dtype=torch.float32,device=dev)
        mem,weak_normalized=model.step(xu,mem,float(ells[eidx]))
        # Outputs represent S/h=N*S, standardized using Step 1F training labels.
        weak=weak_normalized.cpu().numpy().astype(np.float64)*ystds[eidx]/n
        weak-=weak.mean(-1,keepdims=True)
        for _ in range(substeps):
            state=fem_step(state,weak,dt_saved/substeps,nu)
    return np.asarray(errors),max_mass


def workflow(out):
    names=[
        'Step 2C: known Gaussian state transfer and four-snapshot history baseline',
        'Step 2D: introduce recurrent latent memory m(t) and conservative closure',
        'At scale switch: transfer resolved state with known Gaussian filter',
        'Learn a state-conditioned correction to the transferred latent memory',
        'Ablate learned / Gaussian-only / reset memory on held-out trajectories',
        'Assess one-step closure, autonomous FEM rollout, and mass conservation'
    ]
    fig,ax=plt.subplots(figsize=(11,8))
    ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,10)
    ax.set_title('SMTNO — Step 2D pilot: coupled state–memory scale transition',pad=12)
    ys=np.linspace(9.1,.9,len(names))
    for i,(y,s) in enumerate(zip(ys,names)):
        ax.add_patch(FancyBboxPatch((.6,y-.45),8.8,.9,boxstyle='round,pad=.04',fc='white',ec='black',lw=1))
        ax.text(5,y,s,ha='center',va='center',fontsize=9)
        if i+1<len(names):ax.annotate('',xy=(5,ys[i+1]+.5),xytext=(5,y-.5),arrowprops={'arrowstyle':'->'})
    fig.savefig(out/'step2D_workflow.png',dpi=160,bbox_inches='tight');plt.close(fig)


def write_csv(file,records):
    if not records:return
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-dir',type=Path,default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    p.add_argument('--dataset',type=Path,default=None)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--updates-per-epoch',type=int,default=30)
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--hidden',type=int,default=12)
    p.add_argument('--switch-time',type=float,default=.10)
    p.add_argument('--substeps',type=int,default=20)
    p.add_argument('--lr',type=float,default=6e-4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    dev=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    project=args.project_dir
    fpath=args.dataset or project/'datasets'/'burgers_step1F_ensemble.npz'
    if not fpath.is_file():raise FileNotFoundError(fpath)
    out=project/'results'/'step2D';out.mkdir(parents=True,exist_ok=True)
    ckdir=project/'checkpoints'/'step2D';ckdir.mkdir(parents=True,exist_ok=True)
    with np.load(fpath,allow_pickle=False) as f:
        d={k:f[k] for k in ('coarse_N64','coarse_N128','S_tau_N64','S_tau_N128','ell','times','split','case_ids')}
    ells=np.asarray(d['ell'],float)
    times=np.asarray(d['times'],float)
    if not np.allclose(ells,[.01,.02,.04],atol=1e-8):raise ValueError('Expected physical widths [.01,.02,.04]')
    if args.window>=len(times) or args.window<10:raise ValueError('window must be 10..T-1')
    if not np.allclose(np.diff(times),np.diff(times)[0]):raise ValueError('Uniform times required')
    switch=int(np.argmin(abs(times-args.switch_time)))
    if not np.isclose(times[switch],args.switch_time):raise ValueError('Switch time not in dataset')
    split=d['split'].astype(str)
    tr=np.flatnonzero(split=='train');val=np.flatnonzero(split=='val');te=np.flatnonzero(split=='test')
    if not (len(tr)==20 and len(val)==5 and len(te)==5):raise ValueError('Expected split 20/5/5')
    xstd=max(float(np.std(d['coarse_N64'][tr])),1e-7)
    ystd=np.array([max(float(np.sqrt(np.mean((d['S_tau_N64'][tr,j]*64)**2))),1e-7) for j in range(3)])
    U=torch.as_tensor(d['coarse_N64'],dtype=torch.float32,device=dev)
    Y=torch.as_tensor(d['S_tau_N64']*64/ystd[None,:,None,None],dtype=torch.float32,device=dev)
    model=CoupledScaleMemory(args.hidden,xstd).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    best=float('inf');best_state=None;history=[]
    print(f'[DATA] train/val/test=20/5/5, device={dev}, xstd={xstd:.5g}, ystd={ystd}')
    for epoch in range(1,args.epochs+1):
        model.train();train_loss=[]
        for _ in range(args.updates_per_epoch):
            ids=np.random.choice(tr,size=args.batch_size,replace=True)
            loss=batch_sequences(model,U,Y,ells,ids,args.window,dev)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();train_loss.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vloss=[]
            # Fixed deterministic validation windows: no random case leakage.
            for old,new in [(0,2),(1,2),(0,1)]:
                ids=val
                m=model.initial_memory(U[ids,old,0])
                tot=0.0
                for t in range(len(times)):
                    j=old if t<switch else new
                    ut=U[ids,j,t]
                    if t==switch:m=model.transfer(m,ut,float(ells[old]),float(ells[new]),'learned')
                    m,pred=model.step(ut,m,float(ells[j]))
                    tot+=float(torch.mean((pred-Y[ids,j,t])**2).item())
                vloss.append(tot/len(times))
            mean_val=float(np.mean(vloss))
        train_mean=float(np.mean(train_loss))
        history.append({'epoch':epoch,'training_mse':train_mean,'validation_mse':mean_val})
        print(f'epoch {epoch:03d} | train={train_mean:.6g} val={mean_val:.6g}',flush=True)
        if mean_val<best:
            best=mean_val;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':best_state,'hidden':args.hidden,'xstd':xstd,
                'target_std':ystd.tolist(),'ell':ells.tolist(),'training_resolution':64,
                'best_val_mse':best},ckdir/'coupled_scale_memory.pt')
    write_csv(out/'training_losses.csv',history)

    metrics=[];teacher_curves={};rollout_curves={}
    for N in [64,128]:
        ut=torch.as_tensor(d[f'coarse_N{N}'],dtype=torch.float32,device=dev)
        yt=torch.as_tensor(d[f'S_tau_N{N}']*N/ystd[None,:,None,None],dtype=torch.float32,device=dev)
        for mode in ['learned','gaussian','reset']:
            teacher_err,curve=test_teacher(model,ut,yt,ells,te,switch,mode,dev)
            rollout,drift=test_rollout(model,ut,ells,te,switch,mode,ystd,float(times[1]-times[0]),args.substeps,.02,dev)
            teacher_curves[(N,mode)]=curve
            rollout_curves[(N,mode)]=rollout
            metrics.append({'N':N,'transfer_mode':mode,'post_switch_closure_rel_l2':teacher_err,
                            'final_rollout_rel_M_l2':float(rollout[-1]),
                            'post_switch_mean_rollout_error':float(np.mean(rollout[switch:])),
                            'max_abs_mass_drift':drift})
            print(f' N={N:3d}, {mode:8s} teacher forcing err={teacher_err:.4f} '
                  f'final rollout error={rollout[-1]:.6f}',flush=True)
    write_csv(out/'transfer_ablation_metrics.csv',metrics)

    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.semilogy([r['epoch'] for r in history],[r['validation_mse'] for r in history],label='validation')
    ax.semilogy([r['epoch'] for r in history],[r['training_mse'] for r in history],label='training')
    ax.set(xlabel='epoch',ylabel='normalized MSE',title='Step 2D: recurrent closure learning')
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'recurrent_training_losses.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,N in zip(axes,[64,128]):
        for mode in ['learned','gaussian','reset']:
            ax.plot(times,rollout_curves[(N,mode)],label=mode)
        ax.axvline(args.switch_time,color='black',ls='--',alpha=.5)
        ax.set(title=f'N={N}',xlabel='time',ylabel='mean relative FEM error')
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Step 2D: physical-scale switch — learned vs fixed vs reset memory')
    fig.tight_layout();fig.savefig(out/'memory_transfer_rollout.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,N in zip(axes,[64,128]):
        for mode in ['learned','gaussian','reset']:
            ax.plot(times,teacher_curves[(N,mode)],label=mode)
        ax.axvline(args.switch_time,color='black',ls='--',alpha=.5)
        ax.set(title=f'N={N}',xlabel='time',ylabel='forcing normalized RMSE')
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Step 2D: teacher-forced closure accuracy across scale switch')
    fig.tight_layout();fig.savefig(out/'memory_transfer_forcing.png',dpi=160);plt.close(fig)
    workflow(out)
    (out/'run_config.json').write_text(json.dumps({
        'device':str(dev),'epochs':args.epochs,'hidden':args.hidden,
        'train_N':64,'test_N':[64,128], 'switch_time':args.switch_time,
        'ell_from':0.01,'ell_to':0.04,'checkpoint':str(ckdir/'coupled_scale_memory.pt'),
        'statement':'Pilot architecture; no novelty or memory improvement is assumed; ablations and rollout evaluation are required.'
    },indent=2))
    print('[DONE] Step 2D pilot output:',out)

if __name__=='__main__':main()