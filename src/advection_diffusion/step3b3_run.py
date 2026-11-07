from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

def locate_freefem(preferred):
    choices=[preferred, shutil.which('FreeFem++-nw'), shutil.which('FreeFem++'),
             '/Applications/FreeFem++.app/Contents/MacOS/FreeFem++-nw',
             '/Applications/FreeFem++.app/Contents/MacOS/FreeFem++',
             '/opt/homebrew/bin/FreeFem++', '/usr/local/bin/FreeFem++']
    for v in choices:
        if v and (Path(v).is_file() or shutil.which(str(v))):
            return str(v)
    raise RuntimeError('FreeFEM++ not found. Use --freefem /full/path/to/FreeFem++')


HERE=Path(__file__).resolve().parent


def build_samples(seed: int, total: int):
    rng=np.random.default_rng(seed)
    return [dict(
        amp1=float(rng.uniform(.7,1.25)),
        amp2=float(rng.uniform(.2,.65)),
        xc1=float(rng.uniform(.25,.52)),
        yc1=float(rng.uniform(.28,.56)),
        xc2=float(rng.uniform(.50,.75)),
        yc2=float(rng.uniform(.47,.75)),
        w1=float(rng.uniform(.085,.16)),
        w2=float(rng.uniform(.075,.14)),
    ) for _ in range(total)]


def parse_fields(path: Path,n: int):
    df=pd.read_csv(path)
    required={'step','time','x','y','u'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path} missing columns {required-set(df.columns)}')
    steps=np.sort(df.step.unique())
    U=np.full((len(steps),n+1,n+1),np.nan,dtype=np.float64)
    time=np.zeros(len(steps))
    for k,step in enumerate(steps):
        subset=df[df.step==step]
        time[k]=float(subset.time.iloc[0])
        ix=np.rint(subset.x.to_numpy(dtype=float)*n).astype(int)
        iy=np.rint(subset.y.to_numpy(dtype=float)*n).astype(int)
        if len(subset)!=(n+1)**2:
            raise ValueError(f'Incomplete node count for step {step}: {len(subset)}')
        if np.any(ix<0)|np.any(ix>n)|np.any(iy<0)|np.any(iy>n):
            raise ValueError('Coordinates are outside expected uniform grid')
        if len(np.unique(iy*(n+1)+ix)) != (n+1)**2:
            raise ValueError('Repeated grid locations')
        U[k,iy,ix]=subset.u.to_numpy(dtype=float)
    if np.isnan(U).any():
        raise ValueError('Incomplete nodal field')
    if not np.allclose(np.diff(time),np.diff(time)[0],rtol=1e-5,atol=1e-8):
        raise ValueError('Saved times are not equally spaced')
    return time,U


def main():
    ap=argparse.ArgumentParser(description='Step 3B.3: independently varied smooth initial conditions, FreeFEM++ P1/SUPG')
    ap.add_argument('--total-cases',type=int,default=30)
    ap.add_argument('--cases',type=int,default=2,help='Run the first K deterministic cases; use 30 for the full set')
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--n-fine',type=int,default=128)
    ap.add_argument('--dt',type=float,default=.00125)
    ap.add_argument('--T',type=float,default=.2)
    ap.add_argument('--eps',type=float,default=.01)
    ap.add_argument('--save-every',type=int,default=8)
    ap.add_argument('--method',choices=['supg','galerkin'],default='supg')
    ap.add_argument('--outdir',type=Path,default=HERE/'results'/'step3B3')
    ap.add_argument('--freefem',type=str,default=None)
    ap.add_argument('--force',action='store_true')
    a=ap.parse_args()
    if not 1<=a.cases<=a.total_cases or a.n_fine<8 or a.eps<=0 or a.dt<=0 or a.T<=0:
        ap.error('Invalid cases, mesh, time step, duration or diffusivity')
    if abs(a.T/a.dt-round(a.T/a.dt))>1e-7:
        ap.error('T/dt must be an integer for reproducible saved snapshot times')
    out=a.outdir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    ff=locate_freefem(a.freefem)
    src=HERE/'step3b3_ensemble.edp'
    if not src.exists(): raise FileNotFoundError(src)
    cases=build_samples(a.seed,a.total_cases)
    manifest=[]
    for i in range(a.cases):
        p=cases[i]
        folder=out/f'case_{i:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        npz=folder/'field_snapshots.npz'
        cfg=folder/'run_config.json'
        info=dict(case_id=i,seed=a.seed,n=a.n_fine,dt=a.dt,T=a.T,epsilon=a.eps,
                  save_every=a.save_every,method=a.method,initial=p,
                  equation='2D linear transient advection-diffusion-reaction, f=0, zero Dirichlet BC; new ensemble, not the published manufactured Example 3',
                  split='train' if i<int(a.total_cases*2/3) else ('val' if i<int(a.total_cases*5/6) else 'test'))
        if cfg.exists() and not a.force:
            old=json.loads(cfg.read_text())
            if any(old.get(k)!=info.get(k) for k in info):
                raise RuntimeError(f'Existing case config differs: {cfg}. Choose a separate outdir or --force.')
        if a.force or not npz.exists():
            cmd=[ff,'-nw',str(src),'-n',str(a.n_fine),'-eps',str(a.eps),'-dt',str(a.dt),
                 '-T',str(a.T),'-save',str(a.save_every),'-supg','1' if a.method=='supg' else '0',
                 '-out',str(folder)]
            for name,value in p.items(): cmd+=['-'+name,str(value)]
            # FreeFEM argument names for the pulse centers use xc1/yc1/xc2/yc2 and w1/w2.
            info['command']=cmd
            cfg.write_text(json.dumps(info,indent=2))
            with (folder/'run.log').open('w') as log:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
            if proc.returncode:
                lines=(folder/'run.log').read_text(errors='replace').splitlines()
                print('\n'.join(lines[-25:]))
                raise RuntimeError(f'FreeFEM++ failed in case {i}. See {folder / "run.log"}')
            time,u=parse_fields(folder/'fields.csv',a.n_fine)
            x=np.linspace(0,1,a.n_fine+1)
            np.savez_compressed(npz,u=u.astype('float32'),t=time,x=x,y=x,eps=a.eps,
                case=3,benchmark='homogeneous_CDR_ensemble_NOT_paper_example3_manufactured',
                source_zero=True,case_id=i,parameters=json.dumps(p))
        else:
            info['command']=json.loads(cfg.read_text()).get('command',[])
            print('Skipping existing',folder.name,flush=True)
        manifest.append({**{'case_id':i,'split':info['split']},**p,'path':str(npz)})
        print(f'Saved {folder.name}: {npz}',flush=True)
    pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False)
    summary=dict(total_cases=a.total_cases,completed_cases=a.cases,seed=a.seed,
        n_fine=a.n_fine,epsilon=a.eps,method=a.method,dt=a.dt,T=a.T,
        caution='This is a NEW initial-condition ensemble for the published CDR operator, not the same Example 3 manufactured solution. Current linear constant-coefficient PDE is intended for verification, not proof of non-Markovian memory.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Done:',out,flush=True)

if __name__=='__main__':main()