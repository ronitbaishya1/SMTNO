#!/usr/bin/env python3
"""Step 2D.4A: pilot numerical screen of spatial vs temporal multiscale transport.
The periodic finite-volume/implicit-Euler solver here is a pilot, NOT a verified
FreeFEM++/SUPG reproduction of the uploaded papers. It preserves total mobile+
immobile mass; mobile mass alone is not conserved when exchange is present.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIMES = ('control', 'heterogeneous', 'multirate')
ELL = np.array([.01, .02, .04], dtype=np.float64)
K = np.array([1.5, 20.0], dtype=np.float64)
ALPHA = np.array([.35, .20], dtype=np.float64)


def transport_matrix(n, d, velocity):
    dx = 1.0 / n
    dface = 2*d*np.roll(d,-1) / np.maximum(d+np.roll(d,-1),1e-30)
    rows,cols,vals=[],[],[]
    for i in range(n):
        im=(i-1)%n; ip=(i+1)%n
        rows.extend((i,i,i));cols.extend((im,i,ip))
        vals.extend((velocity/dx+dface[im]/dx**2,
                     -velocity/dx-(dface[im]+dface[i])/dx**2,
                     dface[i]/dx**2))
    return sp.coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsc()


def gaussian_coarse(fields, ncoarse, ell):
    # fields shape: trajectories x times x spatial_fine
    nf=fields.shape[-1]
    if nf%ncoarse: raise ValueError('n_fine must be divisible by ncoarse')
    freq=np.fft.rfftfreq(nf, d=1/nf)
    kernel=np.exp(-.5*(2*np.pi*freq*ell)**2)
    filtered=np.fft.irfft(np.fft.rfft(fields,axis=-1)*kernel,n=nf,axis=-1)
    return filtered.reshape(*fields.shape[:-1],ncoarse,nf//ncoarse).mean(-1)


def initial_data(base, n, rng):
    x=(np.arange(n)+.5)/n
    u=[];zs=[]
    for j in range(base):
        center=rng.uniform(.17,.70); width=rng.uniform(.045,.11)
        center2=rng.uniform(.2,.85)
        u0=.08+.7*np.exp(-.5*((x-center)/width)**2)+.22*np.exp(-.5*((x-center2)/.065)**2)
        u0+=rng.uniform(.02,.09)*(1+np.sin(2*np.pi*(x+rng.uniform())) )
        hidden_pattern=np.exp(-.5*((x-rng.uniform(.24,.77))/.10)**2)
        for sign in (-1,1):
            u.append(u0.copy())
            z1=np.maximum(.001,u0*(1+.5*sign)+.20*hidden_pattern)
            z2=np.maximum(.001,u0*(1-.35*sign)+.12*np.roll(hidden_pattern,n//8))
            zs.append(np.stack((z1,z2)))
    return np.array(u), np.array(zs)


def simulate(regime, n, base, nt, tend, seed, micro_delta=1/16):
    if nt<4: raise ValueError('Use at least 4 saved snapshots')
    dt_snap=tend/(nt-1)
    substeps=4;dt=dt_snap/substeps
    x=(np.arange(n)+.5)/n
    d0=.003
    d = np.full(n,d0*np.sqrt(3)) if regime=='control' else d0*(2+np.sin(2*np.pi*x/micro_delta))
    velocity=.35
    L=transport_matrix(n,d,velocity)
    beta=ALPHA*K/(1+dt*K) if regime=='multirate' else np.zeros(2)
    A=(sp.eye(n,format='csc')-dt*L+dt*float(beta.sum())*sp.eye(n,format='csc'))
    lu=splu(A)
    rng=np.random.default_rng(seed)
    U,Z=initial_data(base,n,rng)
    u=U.T.copy();z=np.transpose(Z,(1,2,0)).copy()  # (2,N,cases)
    trajectories=u.shape[-1]
    out_u=np.empty((trajectories,nt,n),dtype=np.float32)
    out_r=np.empty_like(out_u)
    total0=np.sum(u,axis=0) + (np.einsum('k,knc->c',ALPHA,z) if regime=='multirate' else 0)
    mass_drift=0.
    for ti in range(nt):
        if regime=='multirate':
            exchange=np.sum((ALPHA*K)[:,None,None]*(z-u[None]),axis=0)
        else: exchange=np.zeros_like(u)
        out_u[:,ti,:]=u.T
        out_r[:,ti,:]=(L@u+exchange).T
        total=np.sum(u,axis=0)+(np.einsum('k,knc->c',ALPHA,z) if regime=='multirate' else 0)
        mass_drift=max(mass_drift,float(np.max(np.abs(total-total0))/max(float(np.max(abs(total0))),1e-12)))
        if ti==nt-1: break
        for _ in range(substeps):
            rhs=u+dt*np.einsum('k,knc->nc',beta,z)
            u=lu.solve(rhs)
            if regime=='multirate':z=(z+dt*K[:,None,None]*u[None])/(1+dt*K[:,None,None])
    # physical homogenized coefficient of d0*(2+sin) in one dimension
    dhom=d0*np.sqrt(3)
    return out_u,out_r,mass_drift,dhom


def screen_figures(root):
    out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0,1,3001)
    fig,ax=plt.subplots(figsize=(9,4.7))
    for eps in (.05,.01,.001):
        # Cengizci et al. (2026), Example 1, Eq. (53), exact manufactured profile
        c1=np.exp(-1/eps);c2=1-c1
        u=np.exp(-1)*(c1+c2*x-np.exp((x-1)/eps))
        ax.plot(x,u,label=f'epsilon={eps:g}')
    ax.set(xlabel='x',ylabel='u(x, t=1)',title='CPC 2026, Example 1: exact boundary-layer profiles')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'paper_example1_boundary_layer.png',dpi=180);plt.close(fig)

    # Cengizci et al. (2026), Example 2, Eq. (55): analytical steep 2D hump
    xx=np.linspace(0,1,251); X,Y=np.meshgrid(xx,xx,indexing='ij');eps=1e-6;t=.5
    h=(.5+np.arctan(2*eps**(-.5)*(.25**2-(X-.5)**2-(Y-.5)**2))/np.pi)
    u=16*np.sin(np.pi*t)*X*(1-X)*Y*(1-Y)*h
    fig,ax=plt.subplots(figsize=(6.2,5))
    m=ax.pcolormesh(X,Y,u,shading='auto',cmap='viridis');fig.colorbar(m,ax=ax,label='u')
    ax.set(xlabel='x',ylabel='y',title='CPC 2026, Example 2: exact moving-height hump')
    fig.tight_layout();fig.savefig(out/'paper_example2_hump.png',dpi=170);plt.close(fig)

    # Conca & Natesan 2003 Example 6.10 periodic coefficient (smooth case)
    fig,ax=plt.subplots(figsize=(9,4))
    for delta in (1/8,1/16,1/32):
        ax.plot(x,2+np.cos(2*np.pi*x/delta),label=f'delta={delta:.5f}',lw=1.3)
    ax.set(xlabel='x',ylabel='a(x/delta)',title='Conca–Natesan 2003: smooth rapidly oscillating coefficient')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'paper_2003_oscillatory_coefficient.png',dpi=170);plt.close(fig)

    tt=np.linspace(0,1.0,250)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for k,alpha in zip(K,ALPHA): ax.plot(tt,alpha*k*k*np.exp(-k*tt),label=f'kernel term: rate={k:g}, time={1/k:.3f}')
    ax.plot(tt,sum(a*k*k*np.exp(-k*tt) for a,k in zip(ALPHA,K)),color='black',ls='--',label='sum: exact exchange memory kernel')
    ax.set(xlabel='lag time',ylabel='kernel weight',title='Proposed extension: exact two-timescale memory kernel')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'two_timescale_memory_kernel.png',dpi=170);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',type=Path,default=Path('results/step2D4_revised'))
    ap.add_argument('--n-fine',type=int,default=256)
    ap.add_argument('--base-cases',type=int,default=30)
    ap.add_argument('--snapshots',type=int,default=61)
    ap.add_argument('--end-time',type=float,default=.6)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--k-slow',type=float,default=1.5)
    ap.add_argument('--k-fast',type=float,default=20.0)
    ap.add_argument('--smoke-test',action='store_true')
    args=ap.parse_args()
    if args.smoke_test:
        args.n_fine=128;args.base_cases=6;args.snapshots=15;args.end_time=.14
    global K
    K=np.array([args.k_slow,args.k_fast],dtype='float64')
    if np.any(K<=0):raise ValueError('Relaxation rates must be positive')
    root=args.outdir;root.mkdir(parents=True,exist_ok=True)
    screen_figures(root)
    nlist=[32,64] if args.smoke_test else [64,128]
    d={'ell':ELL.astype('float32'),'times':np.linspace(0,args.end_time,args.snapshots,dtype='float32'),
       'base_id':np.repeat(np.arange(args.base_cases),2).astype('int32'),
       'variant':np.tile(np.array([-1,1]),args.base_cases).astype('int8'),
       'regimes':np.array(REGIMES)}
    sanity={}
    pair_bounds=[]
    for regime in REGIMES:
        print('Generating',regime,flush=True)
        u,r,drift,dhom=simulate(regime,args.n_fine,args.base_cases,args.snapshots,args.end_time,args.seed)
        sanity[regime]={'total_mass_max_relative_drift':drift}
        for n in nlist:
            Lcoarse=transport_matrix(n,np.full(n,dhom),.35)
            ud=[];rr=[]
            for ell in ELL:
                uc=gaussian_coarse(u,n,float(ell))
                rc=gaussian_coarse(r,n,float(ell))
                # conservative spatial closure + temporal exchange from eliminated z
                baserhs=(Lcoarse@uc.reshape(-1,n).T).T.reshape(uc.shape)
                ud.append(uc.astype('float32'))
                rr.append((rc-baserhs).astype('float32'))
            d[f'{regime}_u_N{n}']=np.stack(ud,axis=1)
            d[f'{regime}_r_N{n}']=np.stack(rr,axis=1)
            # At t=0 each paired trial has exactly the same mobile state.
            # A deterministic map from current mobile u to residual R cannot
            # distinguish the two labels; its optimal paired MSE is ΔR²/4.
            dR=d[f'{regime}_r_N{n}'][0::2,:,0,:]-d[f'{regime}_r_N{n}'][1::2,:,0,:]
            dU=d[f'{regime}_u_N{n}'][0::2,:,0,:]-d[f'{regime}_u_N{n}'][1::2,:,0,:]
            for j,ell in enumerate(ELL):
                pair_bounds.append(dict(regime=regime,N=n,ell=float(ell),
                   current_u_pair_max_abs=float(np.max(np.abs(dU[:,j]))),
                   markov_paired_mse_floor=float(np.mean(dR[:,j]**2)/4)))
        if regime=='multirate':
            # same initial mobile state for both hidden-state variants
            shared=np.max(np.abs(u[0::2,0,:]-u[1::2,0,:]))
            forcing=np.linalg.norm(r[0::2,0,:]-r[1::2,0,:],axis=-1)
            sanity['paired_initial_state_max_difference']=float(shared)
            sanity['paired_initial_rhs_mean_difference']=float(np.mean(forcing))
            if not shared<1e-12 or not np.mean(forcing)>1e-5:
                raise AssertionError('Paired identifiability control failed')
            xx=np.arange(args.n_fine)/args.n_fine
            fig,ax=plt.subplots(1,2,figsize=(12,4.5))
            ax[0].plot(xx,u[0,0],label='same mobile u at t=0');ax[0].plot(xx,u[1,0],ls='--',label='other hidden configuration')
            ax[0].set(xlabel='x',ylabel='u',title='Same resolved initial state');ax[0].legend(fontsize=8)
            ax[1].plot(xx,r[0,0],label='hidden state A');ax[1].plot(xx,r[1,0],ls='--',label='hidden state B')
            ax[1].set(xlabel='x',ylabel='instantaneous du/dt',title='Different future tendency');ax[1].legend(fontsize=8)
            for a in ax:a.grid(alpha=.2)
            fig.tight_layout();fig.savefig(root/'figures'/'matched_mobile_different_tendency.png',dpi=180);plt.close(fig)
    d['n_values']=np.array(nlist)
    with (root/'paired_markov_ambiguity.csv').open('w',newline='') as f:
        import csv
        w=csv.DictWriter(f,fieldnames=list(pair_bounds[0]));w.writeheader();w.writerows(pair_bounds)
    fig,ax=plt.subplots(figsize=(9,4.7))
    regs=list(REGIMES);positions=np.arange(len(regs))
    target_n=nlist[0];target_ell=float(ELL[-1])
    vals=[next(r['markov_paired_mse_floor'] for r in pair_bounds
             if r['regime']==rg and r['N']==target_n and r['ell']==target_ell) for rg in regs]
    ax.bar(positions,vals);ax.set_xticks(positions,regs)
    ax.set(ylabel='unavoidable paired MSE at identical u(t=0)',
           title=f'Non-identifiability of memoryless closure: N={target_n}, ell={target_ell:g}')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(root/'figures'/'paired_markov_mse_floor.png',dpi=180);plt.close(fig)
    target=root/('pilot_smoke.npz' if args.smoke_test else 'multirate_transport_dataset.npz')
    np.savez_compressed(target,**d)
    (root/'data_diagnostics.json').write_text(json.dumps({'source':'Pilot 1D conservative FV implicit Euler, NOT uploaded papers numerical reproduction','settings':vars(args)|{'outdir':str(args.outdir)},'checks':sanity,'rates':K.tolist(),'capacities':ALPHA.tolist(),'coarse_operators':'mean-coefficient transport; reference is filtered fine operator'},indent=2))
    print('Saved',target,flush=True)
    print('Checks:',json.dumps(sanity,indent=2),flush=True)

if __name__=='__main__':main()