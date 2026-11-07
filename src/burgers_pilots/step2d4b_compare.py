#!/usr/bin/env python3
"""Step 2D.4B: identical-sample residual learnability comparison.
Compare zero, current-state FNO, history-FNO, per-gridpoint LSTM and FNO-LSTM.
Train only N64, check held-out base trajectories and zero-shot N128.
All evaluations are teacher-forced targets, not autonomous rollouts.
A conditional GAN is intentionally deferred until deterministic ambiguity is measured.
"""
from __future__ import annotations
import argparse,copy,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Spectral(nn.Module):
    def __init__(self,width,modes):
        super().__init__();self.modes=modes
        self.w=nn.Parameter(torch.randn(width,width,modes,dtype=torch.cfloat)/width**2)
    def forward(self,z):
        b,c,n=z.shape
        ft=torch.fft.rfft(z,dim=-1);nkeep=min(self.modes,ft.size(-1))
        y=torch.zeros_like(ft)
        y[:,:,:nkeep]=torch.einsum('bim,iom->bom',ft[:,:,:nkeep],self.w[:,:,:nkeep])
        return torch.fft.irfft(y,n=n,dim=-1)


class FNO(nn.Module):
    def __init__(self,channels,width=24,modes=10,layers=2,out_channels=1):
        super().__init__();self.lift=nn.Conv1d(channels,width,1)
        self.spec=nn.ModuleList([Spectral(width,modes) for _ in range(layers)])
        self.local=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(layers)])
        self.head=nn.Sequential(nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,out_channels,1))
    def forward(self,x):
        z=self.lift(x.transpose(1,2))
        for s,l in zip(self.spec,self.local):z=torch.nn.functional.gelu(s(z)+l(z))
        return self.head(z).transpose(1,2)


class ResidualPredictor(nn.Module):
    def __init__(self,name,hist,width=24,modes=10):
        super().__init__();self.name=name;self.hist=hist
        if name=='markov_fno': self.fno=FNO(2,width,modes)
        elif name=='history_fno':self.fno=FNO(hist+1,width,modes)
        elif name=='lstm':
            self.lstm=nn.LSTM(input_size=4,hidden_size=width,num_layers=1,batch_first=True)
            self.head=nn.Linear(width,1)
        elif name=='fno_lstm':
            self.encoder=FNO(2,width,modes,out_channels=width)
            self.lstm=nn.LSTM(input_size=width,hidden_size=width,num_layers=1,batch_first=True)
            self.head=nn.Linear(width,1)
        else:raise ValueError(name)

    def forward(self,x,e):
        # x: (B,history,N), oldest state first. e: (B,) normalized physical scale
        b,h,n=x.shape
        if self.name=='markov_fno':
            feat=torch.stack((x[:,-1],e[:,None].expand(b,n)),dim=-1)
            return self.fno(feat).squeeze(-1)
        if self.name=='history_fno':
            feat=torch.cat((x.transpose(1,2),e[:,None,None].expand(b,n,1)),dim=-1)
            return self.fno(feat).squeeze(-1)
        if self.name=='lstm':
            us=x.transpose(1,2)  # B,N,H
            left=torch.roll(us,1,dims=1);right=torch.roll(us,-1,dims=1)
            feat=torch.stack((left,us,right,e[:,None,None].expand(b,n,h)),dim=-1)
            feat=feat.reshape(b*n,h,4)
            output,_=self.lstm(feat)
            return self.head(output[:,-1]).reshape(b,n)
        if self.name=='fno_lstm':
            xflat=x.reshape(b*h,n,1)
            eflat=e[:,None].expand(b,h).reshape(b*h,1,1).expand(-1,n,1)
            enc=self.encoder(torch.cat((xflat,eflat),dim=-1))
            enc=enc.reshape(b,h,n,-1).permute(0,2,1,3).reshape(b*n,h,-1)
            output,_=self.lstm(enc)
            return self.head(output[:,-1]).reshape(b,n)
        raise ValueError(self.name)


def split_base_ids(ids,seed=2026,smoke=False):
    base=np.unique(ids);rng=np.random.default_rng(seed);rng.shuffle(base)
    if smoke:
        tr=base[:-2];va=base[-2:-1];te=base[-1:]
    else:
        if len(base)!=30:raise ValueError('Full test requires exactly 30 base trajectories')
        tr=base[:20];va=base[20:25];te=base[25:]
    return dict(train=tr,val=va,test=te)


def examples(d,regime,n,indices,hist,xstd,ystd,min_history):
    u=d[f'{regime}_u_N{n}'];r=d[f'{regime}_r_N{n}']
    ids=d['base_id']; ell=d['ell'];ncase,nell,nt,nx=u.shape
    pairselect=np.where(np.isin(ids,indices))[0]
    xx=[];yy=[];ee=[];meta=[]
    for case in pairselect:
        for j in range(nell):
            for t in range(max(hist,min_history)-1,nt):
                xx.append(u[case,j,t-hist+1:t+1]/xstd)
                yy.append(r[case,j,t]/ystd[j])
                ee.append(float(ell[j]/max(ell)))
                meta.append((int(ids[case]),int(d['variant'][case]),j,t))
    return np.stack(xx).astype('float32'),np.stack(yy).astype('float32'),np.array(ee,dtype='float32'),np.array(meta,dtype='int32')


def evaluate(model,data,device,batch,ystd):
    x,y,e,meta=data
    model.eval();pred=[]
    with torch.no_grad():
        for st in range(0,len(x),batch):
            xb=torch.as_tensor(x[st:st+batch],device=device)
            eb=torch.as_tensor(e[st:st+batch],device=device)
            pred.append(model(xb,eb).cpu().numpy())
    pred=np.concatenate(pred)
    ys=y*ystd[meta[:,2],None];ps=pred*ystd[meta[:,2],None]
    return ys,ps


def write_csv(file,rows):
    if not rows:return
    with file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,default=Path('results/step2D4_revised/multirate_transport_dataset.npz'))
    ap.add_argument('--outdir',type=Path,default=Path('results/step2D4_revised/training'))
    ap.add_argument('--regime',choices=['control','heterogeneous','multirate'],default='multirate')
    ap.add_argument('--models',default='markov_fno,history_fno,lstm,fno_lstm')
    ap.add_argument('--history',type=int,default=12)
    ap.add_argument('--min-history',type=int,default=16,help='Shared earliest eligible time across history-length ablations')
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--patience',type=int,default=9)
    ap.add_argument('--width',type=int,default=24)
    ap.add_argument('--modes',type=int,default=10)
    ap.add_argument('--smoke-test',action='store_true')
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device='cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')
    d=dict(np.load(args.dataset,allow_pickle=False))
    ns=sorted(map(int,d['n_values']))
    ntrain,ntest=ns
    hist=min(args.history,4) if args.smoke_test else args.history
    ids=d['base_id']; splits=split_base_ids(ids,args.seed,args.smoke_test)
    out=args.outdir/args.regime/f'h{hist}';out.mkdir(parents=True,exist_ok=True)
    u=d[f'{args.regime}_u_N{ntrain}'];r=d[f'{args.regime}_r_N{ntrain}']
    trmask=np.isin(ids,splits['train'])
    xstd=max(float(np.std(u[trmask])),1e-6)
    # per-scale physically normalized forcing, with fallback for nearly zero targets
    ystd=np.maximum(np.sqrt(np.mean(r[trmask]**2,axis=(0,2,3))),1e-5).astype('float32')
    min_history=min(4,args.min_history) if args.smoke_test else args.min_history
    data={(n,phase):examples(d,args.regime,n,splits[phase],hist,xstd,ystd,min_history)
          for n in ns for phase in ('train','val','test')}
    allmetrics=[];case_rows=[];track=[];predstore={}
    for name in args.models.split(','):
        if name not in ('markov_fno','history_fno','lstm','fno_lstm'):raise ValueError(name)
        model=ResidualPredictor(name,hist,width=args.width,modes=args.modes).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
        X,Y,E,_=data[(ntrain,'train')]
        loader=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(E)),
                          batch_size=args.batch_size,shuffle=True)
        best=np.inf;best_state=copy.deepcopy(model.state_dict());stale=0
        for epoch in range(1,(1 if args.smoke_test else args.epochs)+1):
            model.train();losses=[]
            for xb,yb,eb in loader:
                xb=xb.to(device);yb=yb.to(device);eb=eb.to(device)
                loss=((model(xb,eb)-yb)**2).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                losses.append(float(loss.item()))
            xval,yval,eval_,_=data[(ntrain,'val')]
            model.eval();vals=[]
            with torch.no_grad():
                for st in range(0,len(xval),args.batch_size):
                    xbv=torch.from_numpy(xval[st:st+args.batch_size]).to(device)
                    ybv=torch.from_numpy(yval[st:st+args.batch_size]).to(device)
                    ebv=torch.from_numpy(eval_[st:st+args.batch_size]).to(device)
                    vals.append((float(((model(xbv,ebv)-ybv)**2).sum().item()),len(xbv)*ybv.shape[-1]))
            val=sum(s for s,_ in vals)/sum(k for _,k in vals)
            record=dict(model=name,epoch=epoch,train_mse_normalized=float(np.mean(losses)),val_mse_normalized=val)
            track.append(record)
            if epoch%5==0 or epoch==1:print(args.regime,name,epoch,'val',f'{val:.5f}',flush=True)
            if val<best-1e-6:best=val;best_state=copy.deepcopy(model.state_dict());stale=0
            else:
                stale+=1
                if stale>=args.patience:break
        model.load_state_dict(best_state)
        torch.save({'state_dict':model.state_dict(),'name':name,'history':hist,'regime':args.regime,'ntrain':ntrain,
                    'xstd':xstd,'ystd':ystd.tolist()},out/f'{name}.pt')
        for n in ns:
            yd,pred=evaluate(model,data[(n,'test')],device,args.batch_size,ystd)
            meta=data[(n,'test')][3];predstore[(name,n)]=(yd,pred,meta)
            for j,ell in enumerate(d['ell']):
                for label in ('all',):
                    mask=meta[:,2]==j
                    er=yd[mask]-pred[mask]
                    rmse=float(np.sqrt(np.mean(er**2)))
                    relative=float(np.linalg.norm(er)/max(np.linalg.norm(yd[mask]),1e-8))
                    zero=float(np.sqrt(np.mean(yd[mask]**2)))
                    allmetrics.append(dict(regime=args.regime,model=name,N=n,ell=float(ell),group=label,
                        rmse=rmse,relative_l2=relative,zero_baseline_rmse=zero,
                        improvement_over_zero_pct=100*(1-rmse/max(zero,1e-8)),samples=int(mask.sum())))
            for base in np.unique(meta[:,0]):
                m=meta[:,0]==base
                case_rows.append(dict(regime=args.regime,N=n,model=name,base_id=int(base),
                    rmse=float(np.sqrt(np.mean((yd[m]-pred[m])**2))),
                    reference_rms=float(np.sqrt(np.mean(yd[m]**2)))))
    write_csv(out/'test_metrics.csv',allmetrics)
    write_csv(out/'case_level_metrics.csv',case_rows)
    write_csv(out/'train_curves.csv',track)
    fig,ax=plt.subplots(figsize=(8.5,5))
    for name in args.models.split(','):
        rec=[t for t in track if t['model']==name]
        ax.semilogy([t['epoch'] for t in rec],[t['val_mse_normalized'] for t in rec],label=name)
    ax.set(xlabel='epoch',ylabel='validation normalized MSE',title=f'Step 2D.4B — {args.regime}: held-out validation')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'validation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,n in zip(axes,ns):
        for name in args.models.split(','):
            rows=[m for m in allmetrics if m['N']==n and m['model']==name]
            ax.plot([r['ell'] for r in rows],[r['relative_l2'] for r in rows],marker='o',label=name)
        ax.set(xlabel='filter width ell',ylabel='relative L2 residual error',title=f'N={n} '+('train resolution' if n==ntrain else 'zero-shot'))
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'residual_errors_by_scale.png',dpi=170);plt.close(fig)
    # Explicit caution: regression gain not synonymous with hidden-state identifiability
    (out/'config.json').write_text(json.dumps(dict(regime=args.regime,device=device,training_n=ntrain,
        eval_n=ns,history=hist,min_history=min_history,original_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        split_base_ids={k:v.tolist() for k,v in splits.items()},
        caution='Teacher-forced local closure regression only; no autonomous rollout, no proof of Mori-Zwanzig memory or scale transport.'),indent=2))
    print('Saved results:',out,flush=True)

if __name__=='__main__':main()