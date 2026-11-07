from __future__ import annotations
import argparse, json, random, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the verified FNO building block from Step 3C.1; same architecture for both models.
from step3c1_train import SmallFNO


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load(root,n,ells):
    cases={};meta=pd.read_csv(root/'manifest.csv')
    for r in meta.itertuples(index=False):
        for ell in ells:
            f=root/f'case_{int(r.case_id):03d}'/f'mrmt_n{n}_ell{ell:g}.npz'
            if not f.exists():raise FileNotFoundError(f)
            with np.load(f) as z:
                u=z['u'].astype(np.float32)
                t=z['t'].astype(np.float64)
                split=str(z['split'].item())
                pair=int(z['pair_id'])
                phys=(float(z['diffusion']),float(z['vx']),float(z['vy']))
                params=np.array([float(z['k_fast'])/8,float(z['k_slow']),
                                 float(z['beta_fast'])/.5,float(z['beta_slow'])/.5],np.float32)
            if not np.isfinite(u).all() or np.any(np.diff(t)<=0):raise ValueError(f'Invalid dataset {f}')
            cases[(int(r.case_id),ell)]=dict(u=u,t=t,dt=float(t[1]-t[0]),split=split,
                                            pair=pair,params=params,ell=ell,n=n,phys=phys)
    splits={}
    for (cid,ell),d in cases.items():
        splits.setdefault(d['pair'],set()).add(d['split'])
    if any(len(s)>1 for s in splits.values()):raise RuntimeError('Paired hidden states split across train/val/test')
    return cases


def windows(cases,split,history,max_train_pairs=None):
    pairs=sorted({v['pair'] for v in cases.values() if v['split']==split})
    if split=='train' and max_train_pairs is not None:pairs=pairs[:max_train_pairs]
    result=[]
    for (cid,ell),d in sorted(cases.items()):
        if d['split']==split and d['pair'] in pairs:
            for k in range(history-1,len(d['t'])-1):result.append((cid,ell,k))
    return result


def normalizations(cases,rows):
    a=[];b=[]
    for cid,ell,k in rows:
        d=cases[(cid,ell)]
        a.append(np.square(d['u'][k].astype(np.float64)).mean())
        b.append(np.square((d['u'][k+1].astype(np.float64)-d['u'][k].astype(np.float64))/d['dt']).mean())
    return max(np.mean(a)**.5,1e-7),max(np.mean(b)**.5,1e-7)


def tensor_features(hist,d,mode,unorm,history):
    n=d['n'];u=hist if mode=='history' else np.repeat(hist[-1:],history,axis=0)
    grid=np.arange(n,dtype=np.float32)/n
    x,y=np.meshgrid(grid,grid,indexing='xy')
    feat=np.concatenate([u/unorm,
        np.broadcast_to(d['params'][:,None,None],(4,n,n)),
        np.full((1,n,n),d['ell']/.04,dtype=np.float32),x[None],y[None]],axis=0)
    return feat.astype(np.float32,copy=False)


def batch(cases,rows,history,mode,unorm,dnorm,device):
    feats=[];ys=[]
    for cid,ell,k in rows:
        d=cases[(cid,ell)]
        feat=tensor_features(d['u'][k-history+1:k+1],d,mode,unorm,history)
        target=(d['u'][k+1]-d['u'][k])/(d['dt']*dnorm)
        feats.append(feat);ys.append(target[None])
    return torch.from_numpy(np.stack(feats)).to(device),torch.from_numpy(np.stack(ys)).to(device)


def val_loss(net,cases,rows,mode,history,unorm,dnorm,device,batch_size):
    net.eval();s=0.;n=0
    with torch.no_grad():
        for st in range(0,len(rows),batch_size):
            x,y=batch(cases,rows[st:st+batch_size],history,mode,unorm,dnorm,device)
            e=torch.mean((net(x)-y)**2)
            s+=float(e)*len(x);n+=len(x)
    return s/max(n,1)


def train(net,cases,rows,val_rows,mode,args,unorm,dnorm,device,out):
    net.to(device);opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    best=np.inf;hist=[]
    for epoch in range(1,args.epochs+1):
        net.train();permutation=np.random.permutation(len(rows));loss=0.;ns=0
        for i in range(0,len(rows),args.batch_size):
            ids=permutation[i:i+args.batch_size]
            x,y=batch(cases,[rows[j] for j in ids],args.history,mode,unorm,dnorm,device)
            opt.zero_grad(set_to_none=True)
            err=torch.mean((net(x)-y)**2)
            err.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            loss+=float(err.detach())*len(ids);ns+=len(ids)
        vl=val_loss(net,cases,val_rows,mode,args.history,unorm,dnorm,device,args.batch_size)
        hist.append(dict(model=mode,epoch=epoch,train_loss=loss/ns,val_loss=vl))
        if vl<best:
            best=vl
            torch.save(dict(state_dict=net.state_dict(),epoch=epoch,val_loss=vl),out/f'{mode}_best.pt')
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            print(f'{mode} epoch {epoch}/{args.epochs}: train={loss/ns:.5g} val={vl:.5g}',flush=True)
    net.load_state_dict(torch.load(out/f'{mode}_best.pt',weights_only=True,map_location=device)['state_dict'])
    return hist


def predict(net,hist,d,mode,unorm,dnorm,device,history):
    x=tensor_features(hist,d,mode,unorm,history)
    with torch.no_grad():
        dx=net(torch.from_numpy(x[None]).to(device))[0,0].cpu().numpy()
    return (hist[-1]+d['dt']*dnorm*dx).astype(np.float32)


def no_exchange_one(u,d,cache):
    n=d['n'];dt=d['dt'];diff,vx,vy=d['phys']
    key=(n,dt,diff,vx,vy)
    if key not in cache:
        k=2*np.pi*np.fft.fftfreq(n,d=1/n);x,y=np.meshgrid(k,k,indexing='xy')
        cache[key]=np.exp((-diff*(x*x+y*y)-1j*(vx*x+vy*y))*dt)
    return np.fft.ifft2(np.fft.fft2(u)*cache[key]).real.astype(np.float32)


def compare(models,cases,args,unorm,dnorm,device):
    result=[];cache={}
    for (cid,ell),d in sorted(cases.items()):
        if d['split'] not in ['val','test']:continue
        U=d['u'];H=args.history
        for k in range(H-1,len(U)-1,args.eval_stride):
            for h in args.horizons:
                if k+h>=len(U):continue
                target=U[k+h];den=np.sqrt(np.mean(target.astype(float)**2))+1e-10
                for mode in ['current','history','linear_extrapolation','no_exchange_pde']:
                    past=U[k-H+1:k+1].copy()
                    for _ in range(h):
                        if mode in models:
                            step=predict(models[mode],past,d,mode,unorm,dnorm,device,H)
                        elif mode=='linear_extrapolation':step=2*past[-1]-past[-2]
                        else:step=no_exchange_one(past[-1],d,cache)
                        past=np.concatenate((past[1:],step[None]),axis=0)
                    err=np.sqrt(np.mean((past[-1].astype(float)-target.astype(float))**2))/den
                    result.append(dict(split=d['split'],case_id=cid,pair_id=d['pair'],ell=ell,n=d['n'],
                          start_time=float(d['t'][k]),horizon=h,model=mode,rel_state_error=err))
        print(f'Compared case={cid:03d} N={d["n"]} ell={ell:g}',flush=True)
    return result


def plot_results(root):
    hist=pd.read_csv(root/'training_history.csv');df=pd.read_csv(root/'forecast_metrics.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for m,g in hist.groupby('model'):
        ax.semilogy(g.epoch,g.val_loss,label=m)
    ax.set(xlabel='Epoch',ylabel='Validation MSE',title='MRMT validation (pair-disjoint)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(root/'validation_curves.png',dpi=170);plt.close(fig)
    test=df[df.split=='test']
    means=test.groupby(['n','horizon','model'],as_index=False).rel_state_error.mean()
    fig,ax=plt.subplots(figsize=(7,4.5))
    for (n,m),g in means.groupby(['n','model']):
        ax.plot(g.horizon,g.rel_state_error,marker='o',label=f'{m}, N={n}')
    ax.set(xlabel='Future horizon (saved steps)',ylabel='Mean relative state error',title='MRMT held-out mobile concentration forecasts')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(root/'heldout_forecasts.png',dpi=170);plt.close(fig)
    a=means.pivot_table(index=['n','horizon'],columns='model',values='rel_state_error').reset_index()
    a['gain_pct']=100*(a.current-a.history)/a.current.clip(lower=1e-10)
    a.to_csv(root/'history_gain.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for n,g in a.groupby('n'):ax.plot(g.horizon,g.gain_pct,marker='o',label=f'N={n}')
    ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Future horizon',ylabel='History gain (%)',title='Does observed history help on a known physical memory mechanism?')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(root/'history_gain.png',dpi=170);plt.close(fig)
    print('\nHistory gain on test trajectories:\n',a.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description='Current-state vs history FNO on published multirate mobile-immobile equations.')
    p.add_argument('--indir',default='results/step3C2B')
    p.add_argument('--outdir',default='results/step3C2B_train')
    p.add_argument('--train-res',type=int,default=32)
    p.add_argument('--test-res',type=int,default=64)
    p.add_argument('--ells',default='0.01,0.02,0.04')
    p.add_argument('--history',type=int,default=4)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--modes',type=int,default=8)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=.0007)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--horizons',default='1,5,10')
    p.add_argument('--eval-stride',type=int,default=7)
    p.add_argument('--max-train-pairs',type=int,default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if args.history<2 or args.eval_stride<1 or args.epochs<1:p.error('Invalid history/epoch/stride')
    args.horizons=[int(x) for x in args.horizons.split(',')]
    ells=[float(x) for x in args.ells.split(',')]
    root=Path(args.indir);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    seed_all(args.seed)
    traincases=load(root,args.train_res,ells);testcases=load(root,args.test_res,ells)
    trainrows=windows(traincases,'train',args.history,args.max_train_pairs)
    valrows=windows(traincases,'val',args.history)
    if not trainrows or not valrows or not windows(testcases,'test',args.history):
        raise ValueError('Need train/val/test pairs. Use --pairs 15 for full dataset generation.')
    unorm,dnorm=normalizations(traincases,trainrows)
    models={};log=[]
    for m in ['current','history']:
        seed_all(args.seed)
        # SmallFNO takes 3 standard channels (ell,x,y). Four physical parameter
        # channels are added by increasing the declared history/input width by 4.
        net=SmallFNO(history=args.history+4,width=args.width,modes=args.modes,layers=args.layers)
        log.extend(train(net,traincases,trainrows,valrows,m,args,unorm,dnorm,device,out))
        models[m]=net
    pd.DataFrame(log).to_csv(out/'training_history.csv',index=False)
    results=compare(models,traincases,args,unorm,dnorm,device)+compare(models,testcases,args,unorm,dnorm,device)
    df=pd.DataFrame(results);df.to_csv(out/'forecast_metrics.csv',index=False)
    df.groupby(['split','n','ell','horizon','model'],as_index=False).rel_state_error.mean().to_csv(out/'forecast_summary.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(dict(step='3C2B physical memory test',
         benchmark='MRMT published equations, nondimensional illustrative parameters',
         train_res=args.train_res,test_res=args.test_res,ells=ells,history=args.history,
         horizons=args.horizons,seed=args.seed,epochs=args.epochs,device=str(device),
         train_windows=len(trainrows),val_windows=len(valrows),
         total_trainable_parameters=sum(p.numel() for p in models['current'].parameters()),
         note='Both FNOs have identical capacity; only mobile concentration is observed, known rates and capacities are given to both; oracle immobile fields are never used in training or forecasting. Negative control 3C1 remains separate.'),indent=2))
    plot_results(out)
    print('Saved to',out,flush=True)

if __name__=='__main__':main()
