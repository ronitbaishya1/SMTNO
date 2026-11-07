from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3b1_filter_project import dirichlet_gaussian, prepare_projection
from step3b2_weak_closure import fem_operators, mass_norm
from step3b4_build import load_case, calculate_coarse


def numbers(text, conv):
    a = [conv(v.strip()) for v in text.split(',') if v.strip()]
    if not a or len(a) != len(set(a)):
        raise ValueError('Expected unique comma-separated values')
    return a


def load_three(dirs, cid):
    items=[]
    for folder in dirs:
        src=folder/f'case_{cid:03d}'/'field_snapshots.npz'
        u,t,x,eps=load_case(src,cid)
        items.append(dict(u=u,t=t,x=x,eps=eps,step=float(t[1]-t[0]),src=str(src)))
    steps=np.array([v['step'] for v in items])
    if not np.all(np.diff(steps)<0):
        raise ValueError('Input folders must be in decreasing snapshot interval order: 0.01, 0.005, 0.0025')
    for f in items[1:]:
        if f['u'].shape[1:] != items[0]['u'].shape[1:] or not np.isclose(f['eps'],items[0]['eps']):
            raise ValueError('Fine mesh or PDE epsilon differs between runs')
        if not np.isclose(f['t'][0],items[0]['t'][0]) or not np.isclose(f['t'][-1],items[0]['t'][-1]):
            raise ValueError('Different simulation horizon')
        init_err=np.linalg.norm((f['u'][0]-items[0]['u'][0]).ravel())/(np.linalg.norm(items[0]['u'][0].ravel())+1e-15)
        if init_err>1e-6:
            raise ValueError('Initial conditions mismatch. Check case seed and configuration.')
        coarse_i=np.rint((items[0]['t']-f['t'][0])/f['step']).astype(int)
        if np.any(coarse_i>=len(f['t'])) or not np.allclose(f['t'][coarse_i],items[0]['t'],atol=1e-8):
            raise ValueError('Saved times cannot be aligned exactly')
        rel=np.linalg.norm((items[0]['u']-f['u'][coarse_i]).ravel())/(np.linalg.norm(items[0]['u'].ravel())+1e-15)
        # These runs must use the SAME underlying FEM dt, not changed dt. Only snapshot cadence differs.
        if rel>1e-4:
            raise ValueError(f'Fine FEM states mismatch at common times ({rel:.2e}). Use the same FEM dt and case settings.')
    return items


def closure_for_run(run, n, ell, projection, operators):
    f=np.stack([dirichlet_gaussian(u,ell) for u in run['u']])
    proj,closure,_,_,_,I=calculate_coarse(f,run['t'],n,run['eps'],projection,operators)
    ui=proj.reshape(len(proj),-1)[:,I]
    dt=run['step']
    du=np.full_like(ui,np.nan)
    du[2:] = (3*ui[2:] - 4*ui[1:-1] + ui[:-2])/(2*dt)
    return proj.reshape(len(proj),-1)[:,I],closure,du


def make_plots(d:pd.DataFrame, out:Path):
    g=d.groupby(['n_coarse','ell','snapshot_dt'],as_index=False).agg(
        rel_error=('closure_diff_over_fine_closure','mean'),
        diff_over_tendency=('closure_diff_over_fine_tendency','mean'))
    g.to_csv(out/'cadence_summary.csv',index=False)
    for var, filename, ylabel in [
        ('rel_error','closure_relative_difference_vs_cadence.png','Difference / finest-cadence closure (FEM mass norm)'),
        ('diff_over_tendency','closure_difference_over_tendency.png','Difference / finest-cadence state tendency')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for (n,ell),sub in g.groupby(['n_coarse','ell']):
            sub=sub.sort_values('snapshot_dt')
            ax.plot(sub.snapshot_dt,sub[var],'o-',label=f'N={int(n)}, ell={ell:g}')
        ax.set(xlabel='Saved snapshot interval (same FEM integrator dt)', ylabel=ylabel,
               title='Closure-label sensitivity to saved temporal resolution')
        ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
        fig.tight_layout();fig.savefig(out/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for step,sub in d.groupby('snapshot_dt'):
        curves=sub[(sub.case_id==0)&(sub.n_coarse==64)&np.isclose(sub.ell,0.04)]
        if not curves.empty:
            curves=curves.sort_values('time')
            ax.plot(curves.time,curves.closure_mass_norm,label=f'saved dt={step:g}')
    ax.set(title='Case 000, N=64, ell=0.04: closure norm versus time',
           xlabel='Time',ylabel='FEM mass norm of BDF2 closure rate')
    ax.grid(alpha=.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend()
    fig.tight_layout();fig.savefig(out/'case000_closure_norm_vs_time.png',dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Step 3B.6B: numerical closure convergence under snapshot-cadence refinement')
    ap.add_argument('--baseline',type=Path,default=Path('results/step3B3'))
    ap.add_argument('--fine005',type=Path,default=Path('results/step3B6/FEM_save4'))
    ap.add_argument('--fine0025',type=Path,default=Path('results/step3B6/FEM_save2'))
    ap.add_argument('--outdir',type=Path,default=Path('results/step3B6/cadence'))
    ap.add_argument('--cases',type=int,default=2)
    ap.add_argument('--coarse',default='32,64')
    ap.add_argument('--ells',default='0.01,0.02,0.04')
    args=ap.parse_args()
    if args.cases<1:
        ap.error('cases must be positive')
    ns=numbers(args.coarse,int); ells=numbers(args.ells,float)
    if min(ns)<4 or min(ells)<=0:
        ap.error('Invalid mesh or filtering scales')
    dirs=[p.expanduser().resolve() for p in [args.baseline,args.fine005,args.fine0025]]
    out=args.outdir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    records=[]; summary=[]
    for cid in range(args.cases):
        runs=load_three(dirs,cid)
        n_fine=runs[0]['u'].shape[1]-1
        if any(n>=n_fine or n_fine%n for n in ns):
            raise ValueError('Each coarse resolution must be a divisor of fine mesh size')
        print(f'Case {cid:03d}: saved intervals {[round(r["step"],6) for r in runs]}',flush=True)
        for n in ns:
            proj=prepare_projection(n)
            ops=fem_operators(n,runs[0]['eps'],proj)
            M,A,I,lu,*_=ops
            Mii=M[I][:,I].tocsr()
            for ell in ells:
                triples=[closure_for_run(r,n,ell,proj,ops) for r in runs]
                common_t=runs[0]['t']
                # Compare at EXACTLY the same physical times across all three runs.
                for ti in common_t[2:]:
                    indices=[int(round((ti-r['t'][0])/r['step'])) for r in runs]
                    if not all(j>=2 and j<len(r['t']) and abs(float(r['t'][j])-float(ti))<1e-8
                               for j,r in zip(indices,runs)):
                        raise ValueError('Invalid BDF2 common-time index')
                    finest=triples[-1][1][indices[-1]]
                    du_fin=triples[-1][2][indices[-1]]
                    cden=mass_norm(finest,Mii)+1e-12
                    tden=mass_norm(du_fin,Mii)+1e-12
                    for run,(u,c,du),j in zip(runs,triples,indices):
                        ci=c[j]
                        records.append(dict(case_id=cid,n_coarse=n,ell=ell,time=float(ti),
                            snapshot_dt=run['step'],
                            closure_mass_norm=mass_norm(ci,Mii),
                            closure_diff_over_fine_closure=mass_norm(ci-finest,Mii)/cden,
                            closure_diff_over_fine_tendency=mass_norm(ci-finest,Mii)/tden,
                            state_difference_over_finest=mass_norm(u[j]-triples[-1][0][indices[-1]],Mii)/
                                (mass_norm(triples[-1][0][indices[-1]],Mii)+1e-12)))
                summary.append(dict(case_id=cid,n_coarse=n,ell=ell,reference_snapshot_dt=runs[-1]['step']))
                print(f'  N={n}, ell={ell:g}',flush=True)
    df=pd.DataFrame(records)
    df.to_csv(out/'cadence_by_time.csv',index=False)
    by_case=df.groupby(['case_id','n_coarse','ell','snapshot_dt'],as_index=False).agg(
        mean_relative_closure_difference=('closure_diff_over_fine_closure','mean'),
        mean_difference_over_tendency=('closure_diff_over_fine_tendency','mean'),
        mean_closure_mass_norm=('closure_mass_norm','mean'),
        max_state_mismatch=('state_difference_over_finest','max'))
    by_case.to_csv(out/'cadence_by_case.csv',index=False)
    make_plots(df,out)
    (out/'config.json').write_text(json.dumps({
        'cases':args.cases,'coarse':ns,'ells':ells,'input_paths':[str(p) for p in dirs],
        'meaning':'Same FreeFEM++ integration step in all runs; only SAVED output spacing changes',
        'reference':'finest saved BDF2 label is a numerical REFERENCE, NOT a proven exact closure',
        'first_comparison_time':float(runs[0]['t'][2]),
        'not_a_test_of':'FEM time-integrator convergence or learned physical memory'},indent=2))
    print('Saved:',out,flush=True)

if __name__=='__main__':
    main()