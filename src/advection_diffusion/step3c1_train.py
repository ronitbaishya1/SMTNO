from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse.linalg import splu
from scipy.sparse import coo_matrix


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_nums(s, typ=float):
    return [typ(x.strip()) for x in s.split(',') if x.strip()]


def read_datasets(root, n, ells, expected_cases=30):
    root = Path(root)
    all_cases = sorted(root.glob('case_*'))
    if len(all_cases) < expected_cases:
        raise RuntimeError(f'Expected {expected_cases} case folders in {root}; found {len(all_cases)}')
    cases = {}
    for folder in all_cases:
        try:
            cid = int(folder.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        for ell in ells:
            f = folder / f'coarse_n{n}_ell{ell:g}.npz'
            if not f.exists():
                raise FileNotFoundError(f)
            with np.load(f, allow_pickle=False) as z:
                u = np.asarray(z['u_coarse'], dtype=np.float32)
                t = np.asarray(z['t'], dtype=float)
                sp = str(z['split'].item())
                epsilon = float(z['epsilon'])
                e = float(z['ell'])
                c = int(z['case_id'])
            if c != cid or not np.isclose(e, ell):
                raise ValueError(f'Metadata mismatch: {f}')
            if sp not in {'train', 'val', 'test'}:
                raise ValueError(f'Unknown split {sp} in {f}')
            if (u.ndim != 3 or u.shape != (len(t), n + 1, n + 1)
                    or len(t) < 9 or not np.isfinite(u).all()
                    or not np.all(np.diff(t) > 0)
                    or not np.allclose(np.diff(t), t[1] - t[0], atol=1e-8)):
                raise ValueError(f'Invalid coarse state/times in {f}')
            if np.max(np.abs(u[:, 0, :])) > 1e-5 or np.max(np.abs(u[:, -1, :])) > 1e-5:
                raise ValueError(f'Nonzero Dirichlet boundary in {f}')
            cases[(cid, ell)] = {'u': u, 't': t, 'split': sp, 'eps': epsilon,
                                 'dt': float(t[1] - t[0]), 'n': n}
    for cid in {key[0] for key in cases}:
        splits = {cases[(cid, ell)]['split'] for ell in ells}
        if len(splits) != 1:
            raise ValueError('Split leakage within case')
    counts = Counter(cases[(cid, ells[0])]['split'] for cid in {key[0] for key in cases})
    if any(counts[s] < 1 for s in ['train', 'val', 'test']):
        raise ValueError(f'Need train/val/test cases, found {counts}')
    return cases, dict(counts)


def snapshots(cases, split, history, max_cases=None):
    rows = []
    ids = sorted({cid for (cid, ell), d in cases.items() if d['split'] == split})
    if max_cases is not None and split == 'train':
        ids = ids[:max_cases]
    for (cid, ell), d in sorted(cases.items()):
        if d['split'] != split or cid not in ids:
            continue
        for k in range(history - 1, len(d['t']) - 1):
            rows.append((cid, ell, k))
    return rows


def norm_factors(cases, train_rows):
    u2, du2, count = 0.0, 0.0, 0
    for cid, ell, k in train_rows:
        d = cases[(cid, ell)]
        a = d['u'][k, 1:-1, 1:-1].astype(float)
        b = ((d['u'][k + 1] - d['u'][k]) / d['dt'])[1:-1, 1:-1].astype(float)
        u2 += np.square(a).sum()
        du2 += np.square(b).sum()
        count += a.size
    return max((u2 / count) ** .5, 1e-6), max((du2 / count) ** .5, 1e-6)


def make_batch(cases, indices, history, mode, unorm, dnorm, device):
    xx = []
    targets = []
    for cid, ell, k in indices:
        d = cases[(cid, ell)]
        window = d['u'][k-history+1:k+1]
        if mode == 'current':
            window = np.repeat(window[-1:], history, axis=0)
        n = d['n']
        grid = np.linspace(0., 1., n+1, dtype=np.float32)
        x, y = np.meshgrid(grid, grid, indexing='xy')
        features = np.concatenate((window/unorm,
                                   np.full((1,n+1,n+1),ell/.04,dtype=np.float32),
                                   x[None], y[None]), axis=0)
        target = (d['u'][k+1]-d['u'][k])/(d['dt']*dnorm)
        xx.append(features.astype(np.float32, copy=False))
        targets.append(target[None].astype(np.float32, copy=False))
    return (torch.from_numpy(np.stack(xx)).to(device),
            torch.from_numpy(np.stack(targets)).to(device))


class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / width
        self.wtop = nn.Parameter(scale*torch.randn(width,width,modes,modes,2))
        self.wbottom = nn.Parameter(scale*torch.randn(width,width,modes,modes,2))

    def forward(self, x):
        ft = torch.fft.rfft2(x, norm='ortho')
        b, c, ny, nxh = ft.shape
        my = min(self.modes, ny//2)
        mx = min(self.modes, nxh)
        out = torch.zeros((b, c, ny, nxh),dtype=ft.dtype,device=ft.device)
        wt = torch.view_as_complex(self.wtop[:, :, :my, :mx].contiguous())
        wb = torch.view_as_complex(self.wbottom[:, :, :my, :mx].contiguous())
        out[:, :, :my, :mx] = torch.einsum('bixy,ioxy->boxy',ft[:, :, :my, :mx],wt)
        out[:, :, -my:, :mx] = torch.einsum('bixy,ioxy->boxy',ft[:, :, -my:, :mx],wb)
        return torch.fft.irfft2(out,s=(ny,2*(nxh-1) if (x.shape[-1]%2==0) else 2*(nxh-1)+1),norm='ortho')


class SmallFNO(nn.Module):
    def __init__(self, history=4, width=16, modes=8, layers=3):
        super().__init__()
        self.input = nn.Conv2d(history+3,width,1)
        self.spectral = nn.ModuleList([SpectralConv2d(width,modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(layers)])
        self.output = nn.Sequential(nn.Conv2d(width,width,1), nn.GELU(),nn.Conv2d(width,1,1))

    def forward(self, z):
        u=self.input(z)
        for s,w in zip(self.spectral,self.local):
            u=F.gelu(s(u)+w(u))
        out=self.output(u)
        return out


def masked_mse(pred, target):
    return torch.square(pred[...,1:-1,1:-1]-target[...,1:-1,1:-1]).mean()


def evaluate_val(model, cases, rows, mode, history, unorm, dnorm, device, batchsize):
    model.eval()
    total,count=0.,0
    with torch.no_grad():
        for a in range(0,len(rows),batchsize):
            x,y=make_batch(cases,rows[a:a+batchsize],history,mode,unorm,dnorm,device)
            err=masked_mse(model(x),y)
            total+=err.item()*len(x)
            count+=len(x)
    return total/max(count,1)


def train_model(model,cases,train_rows,val_rows,mode,history,unorm,dnorm,args,device,outdir):
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf')
    best_epoch=0
    rows_out=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        perm=np.random.permutation(len(train_rows))
        loss_sum=0.;n=0
        for a in range(0,len(perm),args.batch_size):
            batch=[train_rows[int(k)] for k in perm[a:a+args.batch_size]]
            x,y=make_batch(cases,batch,history,mode,unorm,dnorm,device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(x)
            loss=masked_mse(pred,y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            loss_sum+=loss.item()*len(batch);n+=len(batch)
        val=evaluate_val(model,cases,val_rows,mode,history,unorm,dnorm,device,args.batch_size)
        rows_out.append({'model':mode,'epoch':epoch,'train_mse':loss_sum/n,'val_mse':val})
        print(f'{mode}: epoch {epoch:03d}/{args.epochs}  train={loss_sum/n:.5f}  val={val:.5f}',flush=True)
        if val<best:
            best,best_epoch=val,epoch
            torch.save({'state_dict':model.state_dict(),'best_val':best,'best_epoch':best_epoch},outdir/f'{mode}_best.pt')
    model.load_state_dict(torch.load(outdir/f'{mode}_best.pt',map_location=device,weights_only=True)['state_dict'])
    return rows_out,{'best_val':best,'best_epoch':best_epoch,'parameters':sum(p.numel() for p in model.parameters())}


def p1_matrices(n, eps):
    # Exact triangle orientation and weak-form coefficients used in Step 3B.2.
    # This makes the classical baseline self-contained, including on Colab.
    idx = np.arange((n + 1)**2).reshape(n + 1, n + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    tri = np.concatenate((np.column_stack((a, b, c)),
                          np.column_stack((a, c, d))), axis=0)
    xg, yg = np.meshgrid(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    xy = np.column_stack((xg.ravel(), yg.ravel()))
    coord = xy[tri]
    x0, y0 = coord[:, 0, 0], coord[:, 0, 1]
    x1, y1 = coord[:, 1, 0], coord[:, 1, 1]
    x2, y2 = coord[:, 2, 0], coord[:, 2, 1]
    det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
    gx = np.stack((y1-y2, y2-y0, y0-y1), axis=1) / det[:, None]
    gy = np.stack((x2-x1, x0-x2, x1-x0), axis=1) / det[:, None]
    area = np.abs(det)/2
    mass = area[:, None, None]/12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])[None]
    bx, by = np.cos(np.pi/3), np.sin(np.pi/3)
    adv = area[:, None, None]*(bx*gx + by*gy)[:, None, :]/3
    stiff = (eps*(gx[:,:,None]*gx[:,None,:]+gy[:,:,None]*gy[:,None,:])
             *area[:,None,None])
    Atri = stiff + adv + mass
    rows = np.broadcast_to(tri[:,:,None], mass.shape).ravel()
    cols = np.broadcast_to(tri[:,None,:], mass.shape).ravel()
    nnode=(n+1)**2
    M = coo_matrix((mass.ravel(), (rows, cols)), shape=(nnode,nnode)).tocsr()
    A = coo_matrix((Atri.ravel(), (rows, cols)), shape=(nnode,nnode)).tocsr()
    I=idx[1:-1,1:-1].ravel()
    return M[I][:,I].tocsr(), A[I][:,I].tocsr(), I


def fem_factory(n, eps, dt, cache):
    key=(n,round(eps,9),round(dt,9))
    if key not in cache:
        Mi,Ai,I=p1_matrices(n,eps)
        lu=splu((Mi+dt*Ai).tocsc())
        cache[key]=(I,Mi,lu)
    return cache[key]


def fem_one(u,d,cache):
    n=d['n'];I,Mi,lu=fem_factory(n,d['eps'],d['dt'],cache)
    r=np.zeros_like(u,dtype=np.float32)
    flat=r.reshape(-1)
    flat[I]=lu.solve(Mi @ u.reshape(-1)[I]).astype(np.float32)
    return r


def net_one(model,history_vals,d,mode,k,unorm,dnorm,device):
    n=d['n'];ell=float(d['ell']);dt=d['dt']
    inp=history_vals if mode=='history' else np.repeat(history_vals[-1:], k, axis=0)
    grid=np.linspace(0,1,n+1,dtype=np.float32)
    x,y=np.meshgrid(grid,grid,indexing='xy')
    features=np.concatenate([inp/unorm,np.full((1,n+1,n+1),ell/.04,dtype=np.float32),x[None],y[None]],axis=0)
    with torch.no_grad():
        p=model(torch.from_numpy(features[None].astype(np.float32, copy=False)).to(device))[0,0].cpu().numpy()
    new=history_vals[-1]+dt*dnorm*p
    new[0,:]=new[-1,:]=0
    new[:,0]=new[:,-1]=0
    return new.astype(np.float32)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a,dtype=np.float64))))


def benchmark(models,cases,history,unorm,dnorm,args,device):
    results=[]
    cache={}
    modes=['current','history','persistence','linear_extrapolation','coarse_FEM_BE','history_erased']
    for (cid,ell),d0 in sorted(cases.items()):
        if d0['split'] not in {'val','test'}:
            continue
        d=dict(d0);d['ell']=ell
        t=d['t'];u=d['u']
        for start in range(history-1,len(t)-1,args.eval_stride):
            for h in args.horizons:
                if start+h>=len(t):
                    continue
                true=u[start+h]
                increment=true-u[start]
                inorm=rms(increment[1:-1,1:-1])+1e-12
                for mode in modes:
                    observed=u[start-history+1:start+1].copy()
                    hist=observed.copy()
                    for step in range(h):
                        if mode in ['current','history','history_erased']:
                            m=models['history' if mode=='history_erased' else mode]
                            m.eval()
                            vals=hist.copy()
                            if mode=='history_erased':
                                vals=np.repeat(vals[-1:],history,axis=0)
                            next_u=net_one(m,vals,d,'history' if mode=='history_erased' else mode,history,unorm,dnorm,device)
                        elif mode=='persistence':
                            next_u=hist[-1].copy()
                        elif mode=='linear_extrapolation':
                            next_u=2*hist[-1]-hist[-2]
                            next_u[0,:]=next_u[-1,:]=0;next_u[:,0]=next_u[:,-1]=0
                        else:
                            next_u=fem_one(hist[-1],d,cache)
                        hist=np.concatenate((hist[1:],next_u[None]),axis=0)
                    err=rms((hist[-1]-true)[1:-1,1:-1])
                    truth_rms=rms(true[1:-1,1:-1])+1e-12
                    results.append(dict(split=d['split'],case_id=cid,ell=ell,n=d['n'],
                        horizon=h,start_time=float(t[start]),model=mode,
                        rel_state_error=err/truth_rms,
                        rel_increment_error=err/inorm,
                        abs_state_rms=err))
        print(f'Evaluated {d["split"]} case {cid:03d} N={d["n"]} ell={ell:g}',flush=True)
    return results


def save_csv(rows,path):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description='Step 3C.1: memory necessity, negative-control state forecasts')
    p.add_argument('--indir',default='results/step3B4')
    p.add_argument('--outdir',default='results/step3C1')
    p.add_argument('--train-res',type=int,default=32)
    p.add_argument('--test-res',type=int,default=64)
    p.add_argument('--ells',default='0.01,0.02,0.04')
    p.add_argument('--history',type=int,default=4)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--modes',type=int,default=8)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--lr',type=float,default=0.0007)
    p.add_argument('--eval-stride',type=int,default=4)
    p.add_argument('--horizons',default='1,3,5')
    p.add_argument('--max-train-cases',type=int,default=None)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    args=p.parse_args()
    if args.history<2 or args.epochs<1 or args.batch_size<1 or args.eval_stride<1:
        p.error('Invalid history/epochs/batch/stride')
    args.horizons=parse_nums(args.horizons,int)
    if any(h<1 for h in args.horizons):p.error('Positive horizons required')
    if args.device=='auto':
        # Torch complex FFT operations are not supported consistently on Apple MPS.
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:device=torch.device(args.device)
    if device.type=='mps':
        p.error('This Fourier spectral implementation uses complex FFT: use --device cpu on Mac, or CUDA in Colab.')
    print('Running on:',device,flush=True)
    seed_all(args.seed)
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    ells=parse_nums(args.ells)
    training,counts=read_datasets(args.indir,args.train_res,ells)
    testing,_=read_datasets(args.indir,args.test_res,ells)
    train_rows=snapshots(training,'train',args.history,args.max_train_cases)
    val_rows=snapshots(training,'val',args.history)
    if not train_rows or not val_rows:raise ValueError('No training or validation windows')
    unorm,dnorm=norm_factors(training,train_rows)
    models={};all_losses=[];details={}
    for mode in ['current','history']:
        seed_all(args.seed)
        model=SmallFNO(args.history,args.width,args.modes,args.layers)
        losses,detail=train_model(model,training,train_rows,val_rows,mode,args.history,unorm,dnorm,args,device,outdir)
        models[mode]=model;all_losses+=losses;details[mode]=detail
    save_csv(all_losses,outdir/'training_history.csv')
    measurements=benchmark(models,training,args.history,unorm,dnorm,args,device)
    measurements+=benchmark(models,testing,args.history,unorm,dnorm,args,device)
    save_csv(measurements,outdir/'forecast_metrics.csv')
    config={'step':'3C1 negative control','source':'Step3B4 u_coarse ONLY: never closure labels',
      'train_resolution':args.train_res,'zero_shot_resolution':args.test_res,
      'ells':ells,'history':args.history,'train_windows':len(train_rows),'validation_windows':len(val_rows),
      'split_counts':counts,'training_only_u_rms':unorm,'training_only_tendency_rms':dnorm,
      'models':details,'forecast_horizons':args.horizons,'eval_stride':args.eval_stride,
      'seed':args.seed,'torch':torch.__version__,'device':str(device),
      'note':'Same-sized FNOs. History model has 4 distinct past states; current model repeats present state. Autoregressive forecasts; no future reference data inside rollout. Memory benefit here alone cannot establish a physical memory kernel. history_erased is an out-of-distribution sensitivity check.'}
    (outdir/'run_config.json').write_text(json.dumps(config,indent=2))
    print('Saved:',outdir,flush=True)

if __name__=='__main__':main()