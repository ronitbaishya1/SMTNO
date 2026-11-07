from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruct_immobile(u,z0,t,k):
    # Exact integration of z'=k(u-z) if u varies linearly between saved snapshots.
    out=np.empty_like(u,dtype=np.float64)
    out[0]=z0
    for i in range(1,len(t)):
        dt=float(t[i]-t[i-1]);kd=k*dt
        decay=np.exp(-kd);a=-np.expm1(-kd)
        w1=1.-a/kd  # coefficient of the end-of-interval mobile concentration
        out[i]=decay*out[i-1]+(a-w1)*u[i-1]+w1*u[i]
    return out


def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))


def main():
    p=argparse.ArgumentParser(description='Audit MRMT mass conservation, true memory kernel, and paired hidden-state ambiguity.')
    p.add_argument('--indir',default='results/step3C2B')
    args=p.parse_args();root=Path(args.indir)
    import pandas as pd
    manifest=pd.read_csv(root/'manifest.csv')
    if len(manifest)<4:raise ValueError('At least 2 pairs recommended')
    records=[];recon=[]
    for item in manifest.itertuples(index=False):
        f=root/f'case_{int(item.case_id):03d}'/'oracle_fields.npz'
        with np.load(f) as d:
            u=d['u'].astype('float64');zf=d['z_fast'].astype('float64');zs=d['z_slow'].astype('float64')
            t=d['t'];mass=d['mass'];kf=float(d['k_fast']);ks=float(d['k_slow']);bf=float(d['beta_fast']);bs=float(d['beta_slow'])
        zfp=reconstruct_immobile(u,zf[0],t,kf)
        zsp=reconstruct_immobile(u,zs[0],t,ks)
        ferror=rms(zfp-zf)/max(rms(zf),1e-12)
        serror=rms(zsp-zs)/max(rms(zs),1e-12)
        recon.append(dict(case_id=int(item.case_id),pair_id=int(item.pair_id),split=item.split,
                         reconstructed_fast_relative_error=ferror,reconstructed_slow_relative_error=serror,
                         max_relative_mass_drift=float(np.max(np.abs(mass-mass[0]))/max(abs(mass[0]),1e-12))))
    for pid,g in manifest.groupby('pair_id'):
        if len(g)!=2:raise ValueError(f'Pair {pid} has {len(g)} cases, expected 2')
        a,b=[root/f'case_{int(cid):03d}'/'oracle_fields.npz' for cid in sorted(g.case_id)]
        with np.load(a) as za,np.load(b) as zb:
            ua,ub=za['u'],zb['u'];ea,eb=za['exchange'],zb['exchange'];t=za['t'];
            if not np.allclose(t,zb['t']):raise ValueError('Twin times differ')
            err0=rms(ua[0]-ub[0]);fut=rms(ua[1]-ub[1]);later=rms(ua[min(5,len(t)-1)]-ub[min(5,len(t)-1)])
            ex0=rms(ea[0]-eb[0]);amp=rms(ua[0])
            if err0>1e-7:raise AssertionError('Paired twins must have identical initial mobile state')
            records.append(dict(pair_id=int(pid),split=str(g.iloc[0].split),initial_mobile_difference=err0,
                initial_exchange_difference=ex0,first_future_mobile_difference=fut,
                later_mobile_difference=later,first_future_rel_to_initial_rms=fut/max(amp,1e-12)))
    pd.DataFrame(records).to_csv(root/'paired_hidden_state_audit.csv',index=False)
    pd.DataFrame(recon).to_csv(root/'memory_kernel_audit.csv',index=False)
    summary=dict(pairs=int(len(records)),cases=int(len(recon)),
         max_initial_mobile_difference=max(x['initial_mobile_difference'] for x in records),
         mean_future_mobile_difference=float(np.mean([x['first_future_mobile_difference'] for x in records])),
         mean_fast_memory_reconstruction_error=float(np.mean([x['reconstructed_fast_relative_error'] for x in recon])),
         mean_slow_memory_reconstruction_error=float(np.mean([x['reconstructed_slow_relative_error'] for x in recon])),
         max_mass_relative_drift=max(x['max_relative_mass_drift'] for x in recon),
         note='At t=0, paired cases have identical observed mobile concentration but different hidden states and time derivatives. This establishes partial observability; a history model cannot resolve hidden states at t=0 without earlier measurements.')
    (root/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    figs=root/'figures';figs.mkdir(exist_ok=True)
    a=root/'case_000'/'oracle_fields.npz';b=root/'case_001'/'oracle_fields.npz'
    with np.load(a) as x,np.load(b) as y:
        t=x['t'];Ua=x['u'];Ub=y['u'];Ea=x['exchange'];Eb=y['exchange'];
        kf,ks,bf,bs=(float(x[k]) for k in ['k_fast','k_slow','beta_fast','beta_slow'])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,U,E,style in [('Twin 0',Ua,Ea,'-'),('Twin 1',Ub,Eb,'--')]:
        axes[0].plot(t,U[:,U.shape[1]//2,U.shape[2]//2],style,label=name)
        axes[1].plot(t,E[:,E.shape[1]//2,E.shape[2]//2],style,label=name)
    axes[0].set(title='Same initial mobile field, different future',xlabel='Time',ylabel='Mobile concentration at center')
    axes[1].set(title='Different hidden-exchange forcing',xlabel='Time',ylabel='Exchange forcing at center')
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(figs/'paired_hidden_states.png',dpi=170);plt.close(fig)
    tt=np.linspace(0,float(t[-1]),250)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(tt,bf*kf*kf*np.exp(-kf*tt),label=f'Fast: k={kf:.2f}')
    ax.plot(tt,bs*ks*ks*np.exp(-ks*tt),label=f'Slow: k={ks:.2f}')
    ax.set(xlabel='Lag time',ylabel='Memory kernel amplitude',title='Known physical exponential memory kernels (illustrative rates)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figs/'known_memory_kernels.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
