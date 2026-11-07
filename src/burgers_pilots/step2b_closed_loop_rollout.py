#!/usr/bin/env python3
"""SMTNO Step 2B — conservative CLOSED-LOOP Burgers FEM rollouts.

Copy this file beside step2a_train_closure_fno.py in
  My Drive/SMTNO/Colab Notebooks/

Requires:
  datasets/burgers_step1F_ensemble.npz
  checkpoints/step2A/memoryless.pt
  checkpoints/step2A/short_history.pt

Call after mounting Google Drive:
  !python "/content/drive/MyDrive/SMTNO/Colab Notebooks/step2b_closed_loop_rollout.py"

Compares no-closure, classical Clark-gradient, memoryless FNO, short-history
FNO, and a fine-data ORACLE held fixed across each saved snapshot interval.

Fair warm start: ALL models start from the same last observed reference at
snapshot index history-1. The history model sees earlier reference states only
for initialization. After that, ALL models evolve without reference state input.
The oracle is a diagnostic and cannot be deployed in practice.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from step2a_train_closure_fno import ClosureFNO

METHODS = ['no_closure', 'clark_gradient', 'memoryless', 'short_history', 'oracle_held']


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir',type=Path,default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    p.add_argument('--dataset',type=Path,default=None)
    p.add_argument('--max-cases',type=int,default=5,help='Number of held-out trajectories (max 5)')
    p.add_argument('--substeps',type=int,default=20,help='Implicit FEM steps per saved interval')
    p.add_argument('--nu',type=float,default=0.02)
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


def mass_apply(u,h):
    return h/6*(4*u+np.roll(u,1,axis=-1)+np.roll(u,-1,axis=-1))


def nonlinear_rhs(u):
    v=np.roll(u,-1,axis=-1)
    flux=(u*u+u*v+v*v)/6.0
    return np.roll(flux,1,axis=-1)-flux


def periodic_step(u, s, dt, nu):
    """Exact periodic P1 FEM matrix solve; explicit convection and forcing.
       u,s are arrays (...,N), with s_i the *weak* forcing (NOT S/h).
    """
    N=u.shape[-1];h=1.0/N
    th=2*np.pi*np.fft.rfftfreq(N)
    mass_eig=h*(2/3+np.cos(th)/3)
    stiff_eig=(2-2*np.cos(th))/h
    rhs=mass_apply(u,h)/dt+nonlinear_rhs(u)+s
    return np.fft.irfft(np.fft.rfft(rhs,axis=-1)/(mass_eig/dt+nu*stiff_eig),n=N,axis=-1)


def clark_weak(u,ell):
    """Consistent with Step 2A's classical gradient/Clark baseline.
    This approximates the subfilter stress with ell**2/2 * u_x**2;
    returns conservative weak forcing S. For flux sign convention see
    nonlinear_rhs. Not the exact FEM stress forcing.
    """
    N=u.shape[-1]
    ux=(np.roll(u,-1,axis=-1)-np.roll(u,1,axis=-1))*(N/2)
    tau=0.5*ell**2 * ux**2
    edge=0.5*(tau+np.roll(tau,-1,axis=-1))
    return np.roll(edge,1,axis=-1)-edge


def relative_mass_l2(u,truth):
    N=u.shape[-1];h=1.0/N
    d=u-truth
    den=np.sum(truth*mass_apply(truth,h),axis=-1)
    num=np.sum(d*mass_apply(d,h),axis=-1)
    return np.sqrt(np.maximum(num,0)/np.maximum(den,1e-15))


def load_checkpoint(name,ckdir,device,history,ells):
    p=ckdir/f'{name}.pt'
    if not p.is_file(): raise FileNotFoundError(f'Missing checkpoint: {p}')
    ck=torch.load(p,map_location='cpu',weights_only=True)
    if ck['model'] != name: raise ValueError(f'Wrong model checkpoint at {p}')
    if not np.allclose(ck['ell'],ells): raise ValueError(f'{name} checkpoint physical widths mismatch')
    num_hist=int(ck['history'])
    if name=='short_history' and num_hist != history:
        raise ValueError('History counts in two checkpoints differ')
    sd=ck['state_dict']
    width=int(sd['lift.weight'].shape[0]);modes=int(sd['spec.0.weights'].shape[-1])
    layers=len([key for key in sd if key.startswith('spec.') and key.endswith('.weights')])
    model=ClosureFNO(num_hist+1,width=width,modes=modes,layers=layers).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, float(ck['x_std']), np.array(ck['target_scales'],dtype=float), num_hist


@torch.inference_mode()
def net_weak(model,hist,ell,x_std,y_std,device,num_hist):
    """hist shape (batch,H,N), current state first; return weak S_i."""
    B,H,N=hist.shape
    if H < num_hist: raise ValueError('Not enough historical states')
    chan=np.moveaxis(hist[:,:num_hist,:],1,-1)/x_std
    scale=np.full((B,N,1),ell/0.04,dtype=np.float32)
    xx=np.concatenate([chan.astype(np.float32),scale],axis=-1)
    pred=model(torch.from_numpy(xx).to(device)).cpu().numpy().astype(float)
    # Step2A net predicts S_i/h, and h=1/N on [0,1].
    return (pred*y_std)/N


def workflow(out):
    boxes=[
        'Step 2A: fixed-data forcing regression',
        'Load saved FNO checkpoints and held-out FEM trajectories',
        'Warm up ALL models from the same four observed snapshots',
        'Autonomous coarse P1 FEM rollout: no closure / Clark / two FNOs',
        'Oracle forcing from fine FEM used ONLY as a diagnostic',
        'Compare rollout error, final profiles, and mass conservation',
        'Next: assess instability, history benefit, and scale transitions'
    ]
    fig,ax=plt.subplots(figsize=(11,10));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    ax.set_title('SMTNO Step 2B — autonomous coarse FEM rollout',pad=12)
    ys=np.linspace(9.1,0.9,len(boxes))
    for i,(y,label) in enumerate(zip(ys,boxes)):
        ax.add_patch(FancyBboxPatch((1,y-.37),8,.74,boxstyle='round,pad=.07',
            facecolor='white',edgecolor='black',lw=1.2))
        ax.text(5,y,label,va='center',ha='center',fontsize=9.3)
        if i <len(ys)-1:
            ax.annotate('',xy=(5,ys[i+1]+.42),xytext=(5,y-.42),arrowprops={'arrowstyle':'->','lw':1.2})
    fig.savefig(out/'step2B_workflow.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    args=parse_args()
    if args.substeps<1: raise ValueError('substeps must be positive')
    if args.max_cases<1: raise ValueError('max-cases must be positive')
    if args.nu <=0: raise ValueError('viscosity must be positive')
    project=args.project_dir
    data_path=args.dataset or project/'datasets'/'burgers_step1F_ensemble.npz'
    if not data_path.is_file():raise FileNotFoundError(data_path)
    out=project/'results'/'step2B';out.mkdir(parents=True,exist_ok=True)
    with np.load(data_path,allow_pickle=False) as f:
        d={k:f[k] for k in ['ell','times','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128']}
    ell=np.asarray(d['ell'],float)
    times=np.asarray(d['times'],float)
    if len(times)<6 or not np.allclose(np.diff(times),times[1]-times[0]):
        raise ValueError('Expected >=6 uniformly spaced saved snapshots')
    held=np.flatnonzero(d['split'].astype(str)=='test')[:args.max_cases]
    if not len(held):raise ValueError('No test trajectories in NPZ')
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('[CONFIG] device:',device, 'test cases:',[int(d['case_ids'][q]) for q in held])
    ckdir=project/'checkpoints'/'step2A'
    m0,xstd0,ystd0,h0=load_checkpoint('memoryless',ckdir,device,1,ell)
    m1,xstd1,ystd1,h1=load_checkpoint('short_history',ckdir,device,4,ell)
    if not np.isclose(xstd0,xstd1) or not np.allclose(ystd0,ystd1):
        raise ValueError('Checkpoints have inconsistent normalizations')
    if h0!=1 or h1!=4:raise ValueError('Expected Step 2A checkpoints with history=1 and 4')
    if ell.max()>0.0400001 or abs(ell.max()-0.04)>1e-8:
        raise ValueError('Step2A scale channel assumes max physical ell = 0.04')
    start=h1-1
    rows=[];by_time=[];profiles={};curves={}
    dt_saved=float(times[1]-times[0]);dt=dt_saved/args.substeps
    print(f'[INFO] reference warm-up ends at t={times[start]:g}; substep dt={dt:g}',flush=True)
    for N in (64,128):
        u_all=np.asarray(d[f'coarse_N{N}'],float)
        s_all=np.asarray(d[f'S_tau_N{N}'],float)
        for jj,ee in enumerate(ell):
            truth=u_all[held,jj,start:,:]
            B,TT,num_nodes=truth.shape
            if num_nodes!=N:raise ValueError('Wrong FEM N')
            # Each model holds its own autonomous history; no truth after start.
            state={name:u_all[held,jj,start,:].copy() for name in METHODS}
            history={name:np.stack([u_all[held,jj,start-k,:] for k in range(h1)],axis=1)
                     for name in METHODS}
            paths={name:[state[name].copy()] for name in METHODS}
            for ti in range(start,len(times)-1):
                for name in METHODS:
                    x=state[name]
                    if name=='no_closure': force=np.zeros_like(x)
                    elif name=='clark_gradient':force=clark_weak(x,ee)
                    elif name=='oracle_held':force=s_all[held,jj,ti,:].copy()
                    elif name=='memoryless':force=net_weak(m0,history[name],ee,xstd0,ystd0[jj],device,1)
                    else:force=net_weak(m1,history[name],ee,xstd1,ystd1[jj],device,4)
                    if np.max(np.abs(np.sum(force,axis=-1)))>1e-5:
                        raise ValueError(f'Nonconservative weak forcing: {name}')
                    for sub in range(args.substeps):
                        x=periodic_step(x,force,dt,args.nu)
                    if not np.all(np.isfinite(x)):
                        raise FloatingPointError(f'Nonfinite rollout values in {name}, ell={ee}, N={N}')
                    state[name]=x
                    history[name]=np.concatenate([x[:,None,:],history[name][:,:-1,:]],axis=1)
                    paths[name].append(x.copy())
            reference=truth
            base_mass=np.mean(truth[:,0,:],axis=-1)
            for name in METHODS:
                pred=np.stack(paths[name],axis=1)
                e=relative_mass_l2(pred,reference) # (cases,times)
                mass_err=np.max(np.abs(np.mean(pred,axis=-1)-base_mass[:,None]))
                curves[(name,N,float(ee))]=e.mean(axis=0)
                for k,case in enumerate(held):
                    for it,t in enumerate(times[start:]):
                        by_time.append({'case_id':int(d['case_ids'][case]),'model':name,'N':N,
                            'ell':float(ee),'time':float(t),'rel_M_l2':float(e[k,it])})
                    rows.append({'case_id':int(d['case_ids'][case]),'model':name,'N':N,
                        'ell':float(ee),'final_rel_M_l2':float(e[k,-1]),
                        'mean_rel_M_l2':float(e[k].mean()),
                        'max_mass_drift':float(np.max(np.abs(np.mean(pred[k],axis=-1)-base_mass[k]))),
                        'warm_start_time':float(times[start])})
                print(f'[DONE] N={N}, ell={ee:g}, {name:14s}, final mean error={e[:,-1].mean():.5g}, mass max={mass_err:.3g}',flush=True)
                if N==64 and abs(ee-max(ell))<1e-8:
                    profiles[name]=pred[0,-1,:]
                    profiles['reference']=reference[0,-1,:]
    # Write diagnostics.
    for filename, data in [('rollout_summary_by_case.csv',rows),('rollout_errors_by_time.csv',by_time)]:
        with (out/filename).open('w',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(data[0].keys()));w.writeheader();w.writerows(data)
    # Aggregate over trajectories for each test setting.
    grouped={}
    for r in rows:
        key=(r['model'],r['N'],r['ell'])
        grouped.setdefault(key,[]).append(r)
    summary=[]
    for (name,N,ee),rr in grouped.items():
        summary.append({'model':name,'mesh_N':N,'ell':ee,'n_cases':len(rr),
            'mean_final_rel_M_l2':float(np.mean([q['final_rel_M_l2'] for q in rr])),
            'median_final_rel_M_l2':float(np.median([q['final_rel_M_l2'] for q in rr])),
            'mean_time_rel_M_l2':float(np.mean([q['mean_rel_M_l2'] for q in rr])),
            'max_abs_mass_drift':float(np.max([q['max_mass_drift'] for q in rr]))})
    with (out/'rollout_summary.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
    # 1: evolution errors for ell=0.04 across two resolutions.
    for N in (64,128):
        fig,ax=plt.subplots(figsize=(8.5,5))
        for name in METHODS:
            ax.plot(times[start:],curves[(name,N,float(max(ell)))],label=name)
        ax.set(xlabel='Physical time',ylabel='Mean relative FEM mass-norm error',
            title=f'Step 2B: autonomous rollout errors, ell={max(ell):g}, N={N}')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/f'rollout_error_vs_time_N{N}.png',dpi=180);plt.close(fig)
    # 2: final errors at all physical scales and mesh sizes.
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),sharey=True)
    for ax,N in zip(axes,[64,128]):
        xx=np.arange(len(ell));bar_w=.16
        for mi,name in enumerate(METHODS):
            yy=[next(q['mean_final_rel_M_l2'] for q in summary
                 if q['model']==name and q['mesh_N']==N and np.isclose(q['ell'],ee)) for ee in ell]
            ax.bar(xx+(mi-2)*bar_w,yy,width=bar_w,label=name)
        ax.set_xticks(xx,[f'{ee:g}' for ee in ell])
        ax.set_title(f'N={N}')
        ax.set_xlabel('Physical filtering width ell')
        ax.set_ylabel('Mean final relative FEM error');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Step 2B: final autonomous rollout error (held-out trajectories)')
    fig.tight_layout();fig.savefig(out/'final_rollout_comparison.png',dpi=180);plt.close(fig)
    # 3: predicted final solution of first test case, large ell / N64.
    fig,ax=plt.subplots(figsize=(9,5))
    xx=np.arange(64)/64
    for name in ['reference']+METHODS:
        ax.plot(xx,profiles[name],label=name,lw=2 if name=='reference' else 1.25)
    ax.set(xlabel='x',ylabel='FEM state',title='Step 2B: final field, first test case, ell=0.04, N=64')
    ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'example_closed_loop_profile.png',dpi=180);plt.close(fig)
    # 4: mass drift diagnostic.
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name in METHODS:
        xx=[z['max_abs_mass_drift'] for z in summary if z['model']==name]
        ax.bar(name,max(xx))
    ax.set(yscale='log',ylabel='Maximum absolute mean(u) drift',title='Step 2B: discrete mass conservation across rollouts')
    ax.tick_params(axis='x',labelrotation=18);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'mass_drift_by_method.png',dpi=180);plt.close(fig)
    workflow(out)
    (out/'run_config.json').write_text(json.dumps({
        'device':str(device),'test_case_ids':[int(d['case_ids'][q]) for q in held],
        'saved_dt':dt_saved,'substeps_per_snapshot':args.substeps,'substep_dt':dt,
        'nu':args.nu,'initial_observed_snapshots':h1,
        'warm_start_time':float(times[start]),
        'reference_usage':'reference states only for shared warm start and scoring; oracle method also uses future true forcing (diagnostic)',
        'closure_update':'hold predicted or oracle weak forcing constant during each snapshot interval',
        'disclaimer':'This is a coarse FEM rollout pilot, not a validation of Mori-Zwanzig theory or new SMTNO architecture.'
    },indent=2))
    print('\n[COMPLETE] Step 2B results at',out)
    print('Note: oracle_held uses ground-truth forcing, so it is NOT deployable.')
    print('Compare memoryless and short_history using rollout_summary.csv.')

if __name__=='__main__':main()