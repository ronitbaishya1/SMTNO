from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.signal import resample


def parse_floats(s):return [float(a) for a in s.split(',') if a.strip()]


def wrapped_gaussian(xx,yy,cx,cy,width):
    # Smooth periodic pulse: distance computed using periodic sinusoids.
    r2=np.sin(np.pi*(xx-cx))**2+np.sin(np.pi*(yy-cy))**2
    return np.exp(-r2/(2*(np.pi*width)**2))


def initial_mobile(n,pair_id):
    grid=np.arange(n)/n;x,y=np.meshgrid(grid,grid,indexing='xy')
    rng=np.random.default_rng(20000+pair_id)
    cx,cy=rng.uniform(.25,.7,size=2)
    sx=rng.uniform(.07,.11)
    u=wrapped_gaussian(x,y,cx,cy,sx)
    u += 0.25*wrapped_gaussian(x,y,(cx+.23)%1,(cy-.19)%1,.085)
    return u


def physical_params(pair_id):
    rng=np.random.default_rng(5000+pair_id)
    return dict(k_fast=float(rng.uniform(4.5,7.5)),k_slow=float(rng.uniform(.35,.75)),
                beta_fast=float(rng.uniform(.25,.45)),beta_slow=float(rng.uniform(.25,.50)))


def transport_multiplier(n,dt,diffusion,vx,vy):
    # integer frequencies for a unit-length periodic domain
    kx=2*np.pi*np.fft.fftfreq(n,d=1/n)
    ky=2*np.pi*np.fft.fftfreq(n,d=1/n)
    X,Y=np.meshgrid(kx,ky,indexing='xy')
    return np.exp(( -diffusion*(X*X+Y*Y)-1j*(vx*X+vy*Y))*dt)


def advance(u,zf,zs,trans_half,reaction):
    # Strang: exact Fourier advection/dispersion for mobile state;
    # exact matrix exponential for local exchange; then Fourier again.
    u=np.fft.ifft2(np.fft.fft2(u)*trans_half).real
    state=np.stack((u,zf,zs),axis=0)
    updated=np.einsum('ab,bxy->axy',reaction,state,optimize=True)
    u,zf,zs=updated[0],updated[1],updated[2]
    u=np.fft.ifft2(np.fft.fft2(u)*trans_half).real
    return u,zf,zs


def gaussian_and_resample(fields,ell,ncoarse):
    n=fields.shape[-1]
    k=2*np.pi*np.fft.fftfreq(n,d=1/n)
    KX,KY=np.meshgrid(k,k,indexing='xy')
    filt=np.exp(-.5*ell**2*(KX**2+KY**2))
    # Apply physical Gaussian before changing discrete grid resolution.
    filtered=np.fft.ifft2(np.fft.fft2(fields,axes=(-2,-1))*filt,axes=(-2,-1)).real
    coarse=resample(resample(filtered,ncoarse,axis=-1),ncoarse,axis=-2).real
    return coarse.astype('float32')


def split_for_pair(pid,npairs):
    # Splits by PAIR, so identical mobile initial fields cannot cross splits.
    ntrain=int(round(npairs*2/3))
    nval=max(1,int(round(npairs*2/15))) if npairs>=5 else max(0,npairs-ntrain-1)
    if pid<ntrain:return 'train'
    if pid<ntrain+nval:return 'val'
    return 'test'


def main():
    p=argparse.ArgumentParser(description='Published multirate mobile–immobile PDE: illustration of physical memory, not a calibrated aquifer case.')
    p.add_argument('--outdir',default='results/step3C2B')
    p.add_argument('--pairs',type=int,default=15)
    p.add_argument('--fine-n',type=int,default=64)
    p.add_argument('--coarse-n',type=int,default=32)
    p.add_argument('--dt',type=float,default=.005)
    p.add_argument('--save-dt',type=float,default=.05)
    p.add_argument('--T',type=float,default=2.5)
    p.add_argument('--diffusion',type=float,default=.003)
    p.add_argument('--vx',type=float,default=.20)
    p.add_argument('--vy',type=float,default=.12)
    p.add_argument('--ells',default='0.01,0.02,0.04')
    args=p.parse_args()
    if args.pairs<2 or args.fine_n<8 or args.coarse_n<4 or args.coarse_n>=args.fine_n:
        p.error('Need pairs>=2, fine-n>=8 and 4<=coarse-n<fine-n')
    if min(args.dt,args.save_dt,args.T,args.diffusion)<=0:p.error('Positive time/diffusion parameters required')
    nsteps=round(args.T/args.dt);stride=round(args.save_dt/args.dt)
    if not np.isclose(nsteps*args.dt,args.T,atol=1e-10) or not np.isclose(stride*args.dt,args.save_dt,atol=1e-10) or stride<1:
        p.error('T and save-dt must be integer multiples of dt')
    root=Path(args.outdir);root.mkdir(parents=True,exist_ok=True)
    ells=parse_floats(args.ells)
    if not ells or any(x<=0 for x in ells):p.error('Positive ells required')
    trans_half=transport_multiplier(args.fine_n,args.dt/2,args.diffusion,args.vx,args.vy)
    tvals=np.arange(0,nsteps+1,stride)*args.dt
    manifest=[]
    for pid in range(args.pairs):
        prm=physical_params(pid)
        ufixed=initial_mobile(args.fine_n,pid)
        # Same mobile concentration, known parameters, and total domain;
        # hidden immobile initial concentrations differ between twins.
        for twin in [0,1]:
            cid=2*pid+twin;case=root/f'case_{cid:03d}'
            expected=case/'oracle_fields.npz'
            if expected.exists():
                raise RuntimeError(f'{expected} exists. Use a new --outdir to avoid mixing configurations.')
            case.mkdir(parents=True)
            split=split_for_pair(pid,args.pairs)
            mobile=ufixed.copy()
            zfast=ufixed*(.1 if twin==0 else .75)
            zslow=ufixed*(.05 if twin==0 else .6)
            kf,ks,bf,bs=(prm[k] for k in ['k_fast','k_slow','beta_fast','beta_slow'])
            matrix=np.array([[-bf*kf-bs*ks,bf*kf,bs*ks],[kf,-kf,0],[ks,0,-ks]],dtype=float)
            R=expm(args.dt*matrix)
            U=[];ZF=[];ZS=[];masses=[]
            for k in range(nsteps+1):
                if k%stride==0:
                    U.append(mobile.astype('float32'))
                    ZF.append(zfast.astype('float32'))
                    ZS.append(zslow.astype('float32'))
                    masses.append(float(np.mean(mobile+bf*zfast+bs*zslow)))
                if k<nsteps:mobile,zfast,zslow=advance(mobile,zfast,zslow,trans_half,R)
            U=np.stack(U);ZF=np.stack(ZF);ZS=np.stack(ZS)
            # Oracle fields are for *evaluation and physical diagnostics*, not neural inputs.
            np.savez_compressed(expected,t=tvals,u=U,z_fast=ZF,z_slow=ZS,
                                exchange=(bf*kf*(ZF-U)+bs*ks*(ZS-U)).astype('float32'),
                                mass=np.array(masses),pair_id=pid,case_id=cid,split=split,
                                **prm)
            for ell in ells:
                for n in [args.coarse_n,args.fine_n]:
                    array=gaussian_and_resample(U,ell,n)
                    np.savez_compressed(case/f'mrmt_n{n}_ell{ell:g}.npz',u=array,t=tvals,
                         ell=ell,n=n,case_id=cid,pair_id=pid,split=split,
                         diffusion=args.diffusion,vx=args.vx,vy=args.vy,**prm)
            manifest.append(dict(case_id=cid,pair_id=pid,twin=twin,split=split,
                                 mass_relative_drift=max(abs(np.array(masses)-masses[0]))/max(abs(masses[0]),1e-12),
                                 **prm))
            print(f'Case {cid:03d}, pair={pid:02d}, {split}: mass drift {manifest[-1]["mass_relative_drift"]:.2e}',flush=True)
    with open(root/'manifest.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    (root/'config.json').write_text(json.dumps(dict(model='multirate mobile-immobile transport (Haggerty-Gorelick formulation)',
             disclaimer='Established equations, but illustrated here with nondimensional synthetic rates, periodic domain, and initial fields: NOT reproduced field measurements or a published calibrated parameter set.',
             fine_n=args.fine_n,coarse_n=args.coarse_n,dt=args.dt,save_dt=args.save_dt,T=args.T,
             diffusion=args.diffusion,vx=args.vx,vy=args.vy,ells=ells,pairs=args.pairs,
             split='pair level; twins kept together',
             method='Strang splitting with exact Fourier mobile transport and exact matrix-exponential exchange',
             masses='conserved integrated mobile + sum beta_j immobile',
             observed='only mobile u in mrmt files; hidden phases only oracle_fields.npz'),indent=2))
    print('Dataset saved to',root)

if __name__=='__main__':main()
