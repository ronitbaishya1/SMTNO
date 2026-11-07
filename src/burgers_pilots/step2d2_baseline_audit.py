#!/usr/bin/env python3
"""SMTNO Step 2D.2: matched autonomous rollout and teacher-forcing audit.

Reuses trained Step 2A (memoryless) and Step 2D.1 checkpoints.
Requires the earlier two scripts alongside this file so the original models and
FEM integrator are reused exactly, rather than reimplemented differently.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from step2b_closed_loop_rollout import load_checkpoint
from step2d1_fno_memory import ScaleMemoryFNO, gaussian, advance, rel_l2


def args_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-dir',type=Path,default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    ap.add_argument('--dataset',type=Path,default=None)
    ap.add_argument('--substeps',type=int,default=20)
    ap.add_argument('--switch-time',type=float,default=0.1)
    ap.add_argument('--nu',type=float,default=0.02)
    ap.add_argument('--max-cases',type=int,default=5)
    ap.add_argument('--max-steps',type=int,default=0,help='Optional short smoke test; 0 evaluates all saved times')
    ap.add_argument('--cpu',action='store_true')
    return ap.parse_args()


def write_rows(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def memoryless_predict(model, u, ell, scale, xstd):
    b,n=u.shape
    x=u.to(torch.float32)
    ee=x.new_full((b,n,1),float(ell)/0.04)
    channels=torch.cat([(x/xstd).unsqueeze(-1),ee],dim=-1)
    return model(channels)


def load_models(project,device,ells):
    fno,xstd0,ystd0,hist=load_checkpoint('memoryless',project/'checkpoints'/'step2A',device,1,ells)
    if hist!=1:raise ValueError('Expected a one-snapshot memoryless FNO')
    ck=torch.load(project/'checkpoints'/'step2D1'/'scale_memory_fno.pt',map_location='cpu',weights_only=True)
    rec=ScaleMemoryFNO(int(ck['hidden']),int(ck['width']),int(ck['modes']),float(ck['xstd'])).to(device)
    rec.load_state_dict(ck['state_dict']);rec.eval();fno.eval()
    return fno, rec, float(xstd0), np.asarray(ystd0,float), np.asarray(ck['ystd'],float)


@torch.no_grad()
def run_rollout(model0,model1,U,IDs,ells,ystd0,ystd1,xstd0,times,swidx,a):
    modes=('memoryless','learned','gaussian','reset')
    rows=[]; B=len(IDs)
    tstart=3
    end=len(times) if a.max_steps==0 else min(len(times),tstart+a.max_steps+1)
    times=np.asarray(times)
    for n in (64,128):
        fine=U[n]
        initial=fine[IDs,0,tstart].to(torch.float64)
        m0=model1.initial_memory(initial.to(torch.float32))
        for t in range(4):
            m0,_=model1.step(fine[IDs,0,t],m0,float(ells[0]))
        states={k:initial.clone() for k in modes}
        memories={k:m0.clone() for k in modes if k!='memoryless'}
        first_mass=initial.mean(-1)
        for t in range(tstart,end):
            j=0 if t<swidx else 2
            if t==swidx:
                for key in modes:
                    states[key]=gaussian(states[key],float(ells[0]),float(ells[2]))
                    if key!='memoryless':
                        memories[key]=model1.transfer(memories[key],states[key].float(),float(ells[0]),float(ells[2]),key)
            ref=fine[IDs,j,t].double()
            for name in modes:
                uu=states[name]
                e=rel_l2(uu,ref).cpu().numpy()
                drift=(uu.mean(-1)-first_mass).abs().cpu().numpy()
                for i,cid in enumerate(IDs):
                    rows.append({'N':n,'model':name,'case_index':int(cid),'time':float(times[t]),'ell':float(ells[j]),'relative_error':float(e[i]),'mass_drift':float(drift[i]),'period':'post_switch' if t>=swidx else 'pre_switch'})
                if t==end-1:continue
                if name=='memoryless':
                    p=memoryless_predict(model0,uu,float(ells[j]),j,xstd0)
                    states[name]=advance(uu,p,ystd0,j,a.substeps,float(times[t+1]-times[t]),a.nu)
                else:
                    memories[name],p=model1.step(uu.float(),memories[name],float(ells[j]))
                    states[name]=advance(uu,p,ystd1,j,a.substeps,float(times[t+1]-times[t]),a.nu)
    return rows


@torch.no_grad()
def forcing_audit(model0,model1,U,S,IDs,ells,ystd0,ystd1,xstd0,times,swidx,a):
    records=[]
    end=len(times) if a.max_steps==0 else min(len(times),3+a.max_steps+1)
    for n in (64,128):
        u_all=U[n];target=S[n]
        for mode in ('learned','gaussian','reset'):
            mem=model1.initial_memory(u_all[IDs,0,0])
            for t in range(3):
                mem,_=model1.step(u_all[IDs,0,t],mem,float(ells[0]))
            for t in range(3,end):
                j=0 if t<swidx else 2
                u=u_all[IDs,j,t]
                if t==swidx:
                    mem=model1.transfer(mem,u,float(ells[0]),float(ells[2]),mode)
                mem,pred=model1.step(u,mem,float(ells[j]))
                true=(target[IDs,j,t]*n/ystd1[j]).to(torch.float32)
                err=(pred-true).square().mean(-1).cpu().numpy()
                for i,cid in enumerate(IDs):
                    records.append({'N':n,'model':mode,'case_index':int(cid),'time':float(times[t]),'period':'post_switch' if t>=swidx else 'pre_switch','normalized_forcing_mse':float(err[i])})
        for t in range(3,end):
            j=0 if t<swidx else 2
            u=u_all[IDs,j,t]
            pred=memoryless_predict(model0,u,float(ells[j]),j,xstd0)
            true=(target[IDs,j,t]*n/ystd0[j]).to(torch.float32)
            err=(pred-true).square().mean(-1).cpu().numpy()
            for i,cid in enumerate(IDs):
                records.append({'N':n,'model':'memoryless','case_index':int(cid),'time':float(times[t]),'period':'post_switch' if t>=swidx else 'pre_switch','normalized_forcing_mse':float(err[i])})
    return records


def summarize(rollout,forcing):
    result=[]
    for n in (64,128):
        for mode in ('memoryless','learned','gaussian','reset'):
            rs=[r for r in rollout if r['N']==n and r['model']==mode]
            if not rs:continue
            tmax=max(r['time'] for r in rs)
            fs=[r['relative_error'] for r in rs if abs(r['time']-tmax)<1e-9]
            ss=[r['relative_error'] for r in rs if r['period']=='post_switch']
            ff=[r['normalized_forcing_mse'] for r in forcing if r['N']==n and r['model']==mode and r['period']=='post_switch']
            result.append({'N':n,'model':mode,'mean_final_relative_error':float(np.mean(fs)),
                           'mean_post_switch_relative_error':float(np.mean(ss)) if ss else float('nan'),
                           'post_switch_teacher_forced_MSE':float(np.mean(ff)) if ff else float('nan'),
                           'max_mass_drift':max(r['mass_drift'] for r in rs)})
    return result


def plot_results(folder,rollout,forcing,summary,train_losses):
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,n in zip(axs,(64,128)):
        for mode in ('memoryless','learned','gaussian','reset'):
            rows=[r for r in rollout if r['N']==n and r['model']==mode]
            if not rows:continue
            tvals=sorted(set(r['time'] for r in rows))
            y=[np.mean([r['relative_error'] for r in rows if abs(r['time']-t)<1e-9]) for t in tvals]
            ax.plot(tvals,y,label=mode)
        ax.axvline(0.1,color='black',ls='--',lw=1,label='switch')
        ax.set(xlabel='time',ylabel='mean FEM-relative L2 error',title=f'N={n}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('2D.2: matched baseline vs recurrent transfer')
    fig.tight_layout();fig.savefig(folder/'matched_rollouts.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,n in zip(axs,(64,128)):
        modes=('memoryless','learned','gaussian','reset')
        early=[];late=[]
        for mode in modes:
            r=[x for x in forcing if x['N']==n and x['model']==mode]
            early.append(np.mean([x['normalized_forcing_mse'] for x in r if x['period']=='pre_switch']) if any(x['period']=='pre_switch' for x in r) else np.nan)
            late.append(np.mean([x['normalized_forcing_mse'] for x in r if x['period']=='post_switch']) if any(x['period']=='post_switch' for x in r) else np.nan)
        xx=np.arange(len(modes));ax.bar(xx-.2,early,width=.4,label='pre');ax.bar(xx+.2,late,width=.4,label='post')
        ax.set(xticks=xx,xticklabels=modes,ylabel='teacher-forced normalized forcing MSE',title=f'N={n}')
        ax.tick_params(axis='x',labelrotation=20);ax.legend();ax.grid(alpha=.2,axis='y')
    fig.tight_layout();fig.savefig(folder/'teacher_forcing_audit.png',dpi=180);plt.close(fig)

    if train_losses.is_file():
        import pandas as pd
        d=pd.read_csv(train_losses)
        fig,ax=plt.subplots(figsize=(7,4))
        for col in ('train_loss','validation_loss'):
            ax.plot(d['epoch'],d[col],label=col)
        ax.set_yscale('log');ax.legend();ax.set(xlabel='epoch',ylabel='reported combined loss',title='2D.1: training vs validation (different window schedules)')
        ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(folder/'training_validation_gap.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    stages=('Load Step 2A + Step 2D.1 checkpoints','Same four-snapshot observed warm start','Switch scale at t = 0.10 (known Gaussian state transfer)','Compare autonomous FEM rollout on N=64 and N=128','Measure teacher-forced forcing prediction separately','Decide: closure regression vs rollout instability')
    yy=np.linspace(.91,.09,len(stages))
    for i,(y,name) in enumerate(zip(yy,stages)):
        ax.text(.5,y,name,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.45',fc='white',ec='black'))
        if i<len(stages)-1:ax.annotate('',xy=(.5,yy[i+1]+.045),xytext=(.5,y-.045),arrowprops=dict(arrowstyle='->',lw=1.4))
    fig.tight_layout();fig.savefig(folder/'step2D2_workflow.png',dpi=180);plt.close(fig)


def main():
    a=args_parser()
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    p=a.project_dir
    data_path=a.dataset or (p/'datasets'/'burgers_step1F_ensemble.npz')
    if not data_path.exists():raise FileNotFoundError(f'Missing NPZ: {data_path}')
    with np.load(data_path,allow_pickle=False) as f:
        d={k:f[k] for k in ('ell','times','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128')}
    times=np.asarray(d['times'],float);ells=np.asarray(d['ell'],float)
    swidx=int(np.argmin(np.abs(times-a.switch_time)))
    if not np.isclose(times[swidx],a.switch_time):raise ValueError('Scale switch must land on an exact saved snapshot')
    test=np.flatnonzero(d['split'].astype(str)=='test')[:a.max_cases]
    if not len(test):raise ValueError('No test trajectories')
    U={n:torch.as_tensor(d[f'coarse_N{n}'],dtype=torch.float32,device=device) for n in (64,128)}
    S={n:torch.as_tensor(d[f'S_tau_N{n}'],dtype=torch.float32,device=device) for n in (64,128)}
    model0,model1,xstd0,ystd0,ystd1=load_models(p,device,ells)
    print('Device:',device,'Test cases:',[int(i) for i in test],flush=True)
    rollout=run_rollout(model0,model1,U,test,ells,ystd0,ystd1,xstd0,times,swidx,a)
    forcing=forcing_audit(model0,model1,U,S,test,ells,ystd0,ystd1,xstd0,times,swidx,a)
    summary=summarize(rollout,forcing)
    out=p/'results'/'step2D2';out.mkdir(parents=True,exist_ok=True)
    write_rows(out/'matched_rollout_by_case_time.csv',rollout)
    write_rows(out/'teacher_forced_by_case_time.csv',forcing)
    write_rows(out/'matched_summary.csv',summary)
    plot_results(out,rollout,forcing,summary,p/'results'/'step2D1'/'training_losses.csv')
    config=dict(source_models=['Step2A memoryless','Step2D1 recurrent'],same_integrator='step2d1_fno_memory.advance',same_warm_start=4,switch=float(a.switch_time),max_cases=len(test),substeps=a.substeps,device=str(device),note='Diagnostic: no retraining or novelty claim')
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    print('Results saved:',out)
    for row in summary:print(row)


if __name__=='__main__':main()