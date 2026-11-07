#!/usr/bin/env python3
"""SMTNO Step 2D.3: baseline-anchored recurrent residual closure.
Uses existing Step 2A memoryless FNO checkpoint, frozen; trains only a new
small persistent-memory correction. All modes share the same base model.
This is a controlled pilot, not evidence of a new architecture or MZ memory.
"""
from __future__ import annotations
import argparse, csv, json, random, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from step2b_closed_loop_rollout import load_checkpoint
from step2d1_fno_memory import gaussian, advance, rel_l2

class SpectralMemory(nn.Module):
    def __init__(self, hidden=12, modes=10, residual_scale=.1):
        super().__init__()
        self.hidden = hidden
        self.modes = modes
        self.residual_scale = residual_scale
        self.lift = nn.Conv1d(hidden+3, hidden, 1)
        self.spectral = nn.Parameter(torch.randn(hidden, hidden, modes, dtype=torch.cfloat)*.01)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.gate = nn.Conv1d(hidden, hidden, 1)
        self.candidate = nn.Conv1d(hidden, hidden, 1)
        self.out = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.transfer_in = nn.Conv1d(hidden+3, hidden, 1)
        self.transfer_out = nn.Conv1d(hidden, hidden, 1)
        nn.init.zeros_(self.transfer_out.weight)
        nn.init.zeros_(self.transfer_out.bias)

    def initial_memory(self, u):
        return u.new_zeros((u.shape[0], self.hidden, u.shape[-1]))

    def _mix(self, x):
        ft=torch.fft.rfft(x,dim=-1)
        out=torch.zeros_like(ft)
        k=min(self.modes,ft.shape[-1]);out[:,:,:k]=torch.einsum('bim,iom->bom',ft[:,:,:k],self.spectral[:,:,:k])
        return torch.nn.functional.gelu(torch.fft.irfft(out,n=x.shape[-1],dim=-1)+self.pointwise(x))

    def forward(self,u,base_pred,m,ell,xstd):
        b,n=u.shape
        e=u.new_full((b,1,n),ell/.04)
        z=torch.cat((m,(u/xstd)[:,None],base_pred[:,None],e),dim=1)
        q=self._mix(torch.nn.functional.gelu(self.lift(z)))
        gate=torch.sigmoid(self.gate(q))
        m_new=(1-gate)*m+gate*torch.tanh(self.candidate(q))
        delta=self.residual_scale*torch.tanh(self.out(m_new)).squeeze(1)
        pred=base_pred+delta
        return m_new,pred-pred.mean(-1,keepdim=True)

    def transfer(self,m,new_u,old,new,mode,xstd):
        if mode=='reset':return torch.zeros_like(m)
        mg=gaussian(m,old,new)
        if mode=='gaussian':return mg
        if mode!='learned':raise ValueError(mode)
        b,_,n=m.shape
        channels=torch.cat((mg,(new_u/xstd)[:,None],new_u.new_full((b,1,n),old/.04),new_u.new_full((b,1,n),new/.04)),dim=1)
        correction=.10*torch.tanh(self.transfer_out(torch.nn.functional.gelu(self.transfer_in(channels))))
        return mg+correction


def base_forcing(base, u, ell, xstd):
    b,n=u.shape
    uf=u.to(torch.float32)
    inp=torch.cat(((uf/xstd).unsqueeze(-1),uf.new_full((b,n,1),ell/.04)),dim=-1)
    return base(inp)


def batch_sequence(model,base,U,Y,idx,ells,ystd,xstd,times,a,start,old,new,switch_k):
    warm=4; horizon=a.horizon
    m=model.initial_memory(U[idx,old,start])
    for t in range(start,start+warm):
        v=U[idx,old,t]
        with torch.no_grad():bp=base_forcing(base,v,float(ells[old]),xstd)
        m,_=model(v,bp,m,float(ells[old]),xstd)
    u=U[idx,old,start+warm-1].double()
    fo=0.;ro=0.;j=old
    dt=float(times[1]-times[0])
    for k in range(horizon):
        tt=start+warm-1+k
        if k==switch_k:
            u=gaussian(u,float(ells[old]),float(ells[new]))
            m=model.transfer(m,u.float(),float(ells[old]),float(ells[new]),'learned',xstd)
            j=new
        with torch.no_grad():
            bp=base_forcing(base,u,float(ells[j]),xstd)
        m,p=model(u.float(),bp,m,float(ells[j]),xstd)
        fo=fo+(p-Y[idx,j,tt]).square().mean()
        u=advance(u,p,ystd,j,a.substeps,dt,a.nu)
        ro=ro+((u-U[idx,j,tt+1].double())/xstd).square().mean()
    return fo/horizon+a.rollout_weight*ro/horizon


def val_score(model,base,U,Y,ids,ells,ystd,xstd,times,a):
    model.eval();vals=[]
    with torch.no_grad():
        for start in (0,20,45):
            if start+4+a.horizon>len(times):continue
            for old,new in ((0,2),(1,2)):
                vals.append(float(batch_sequence(model,base,U,Y,ids,ells,ystd,xstd,times,a,start,old,new,2)))
    return float(np.mean(vals))


@torch.no_grad()
def rollout(model,base,U,ids,ells,ystd,xstd,times,a,n,mode):
    ti=int(np.flatnonzero(np.isclose(times,a.switch_time))[0]); t0=3
    mem=model.initial_memory(U[ids,0,0]); u=U[ids,0,t0].double().clone()
    for t in range(4):
        bp=base_forcing(base,U[ids,0,t],float(ells[0]),xstd)
        mem,_=model(U[ids,0,t],bp,mem,float(ells[0]),xstd)
    initial_mass=u.mean(-1); rows=[]
    for t in range(t0,len(times)):
        j=0 if t<ti else 2
        if t==ti:
            u=gaussian(u,float(ells[0]),float(ells[2]))
            if mode!='baseline':mem=model.transfer(mem,u.float(),float(ells[0]),float(ells[2]),mode,xstd)
        err=rel_l2(u,U[ids,j,t].double()).cpu().numpy()
        drift=(u.mean(-1)-initial_mass).abs().cpu().numpy()
        for k,cid in enumerate(ids):
            rows.append(dict(N=n,mode=mode,case_id=int(cid),time=float(times[t]),relative_error=float(err[k]),mass_drift=float(drift[k])))
        if t==len(times)-1:break
        bp=base_forcing(base,u,float(ells[j]),xstd)
        if mode=='baseline':p=bp
        else:mem,p=model(u.float(),bp,mem,float(ells[j]),xstd)
        u=advance(u,p,ystd,j,a.substeps,float(times[t+1]-times[t]),a.nu)
    return rows


@torch.no_grad()
def closure_audit(model,base,U,Y,ids,ells,xstd,times,n,mode,switch_time):
    ti=int(np.flatnonzero(np.isclose(times,switch_time))[0]); mem=model.initial_memory(U[ids,0,0]);rows=[]
    for t in range(len(times)):
        j=0 if t<ti else 2
        uu=U[ids,j,t];bp=base_forcing(base,uu,float(ells[j]),xstd)
        if t==ti and mode!='baseline':mem=model.transfer(mem,uu,float(ells[0]),float(ells[2]),mode,xstd)
        if mode=='baseline':p=bp
        else:mem,p=model(uu,bp,mem,float(ells[j]),xstd)
        err=(p-Y[ids,j,t]).square().mean(-1).cpu().numpy()
        for k,cid in enumerate(ids):rows.append(dict(N=n,mode=mode,case_id=int(cid),time=float(times[t]),forcing_mse=float(err[k])))
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plots(out,losses,rows,summ):
    fig,ax=plt.subplots(figsize=(7,4));ax.semilogy([x['epoch'] for x in losses],[x['train'] for x in losses],label='train');ax.semilogy([x['epoch'] for x in losses],[x['validation'] for x in losses],label='val');ax.legend();ax.set(xlabel='epoch',ylabel='combined loss',title='Step 2D.3 baseline-preserving residual training');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,n in zip(axs,(64,128)):
        for mode in ('baseline','learned','gaussian','reset'):
            v=[r for r in rows if r['N']==n and r['mode']==mode]
            tt=sorted(set(z['time'] for z in v));ee=[np.mean([z['relative_error'] for z in v if z['time']==t]) for t in tt]
            ax.plot(tt,ee,label=mode)
        ax.axvline(.1,linestyle='--',color='k',lw=1);ax.set(xlabel='time',ylabel='mean relative FEM L2 error',title=f'N={n}');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Scale-switch rollouts: frozen FNO vs residual memory');fig.tight_layout();fig.savefig(out/'scale_switch_rollouts.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    labels=['Load frozen, accurate Step 2A memoryless FNO','Train ONLY persistent spectral memory residual','Initialize residual at zero (exact FNO at start)','Use short autonomous FEM loss and matched validation','Transfer hidden memory: learned / Gaussian / reset','Audit closure forcing and N=64 / N=128 rollouts']
    ys=np.linspace(.89,.11,len(labels))
    for i,(y,label) in enumerate(zip(ys,labels)):
        ax.text(.5,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.5',fc='white',ec='black'))
        if i<len(labels)-1:ax.annotate('',xy=(.5,ys[i+1]+.043),xytext=(.5,y-.043),arrowprops=dict(arrowstyle='->',lw=1.3))
    fig.tight_layout();fig.savefig(out/'step2D3_workflow.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-dir',type=Path,default=Path('/content/drive/MyDrive/SMTNO/Colab Notebooks'))
    ap.add_argument('--dataset',type=Path,default=None)
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--updates-per-epoch',type=int,default=16)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--horizon',type=int,default=6)
    ap.add_argument('--substeps',type=int,default=20)
    ap.add_argument('--rollout-weight',type=float,default=8.)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--hidden',type=int,default=12)
    ap.add_argument('--modes',type=int,default=10)
    ap.add_argument('--residual-scale',type=float,default=.1)
    ap.add_argument('--switch-time',type=float,default=.1)
    ap.add_argument('--nu',type=float,default=.02)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    p=a.project_dir; path=a.dataset or p/'datasets'/'burgers_step1F_ensemble.npz'
    if not path.exists():raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k] for k in ('ell','times','split','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128')}
    ells=data['ell'].astype(float);times=data['times'].astype(float);split=data['split'].astype(str)
    if not np.allclose(ells,[.01,.02,.04]):raise ValueError('Filter widths mismatch')
    if not np.any(np.isclose(times,a.switch_time)):raise ValueError('Switch time not in saved snapshots')
    train=np.flatnonzero(split=='train');val=np.flatnonzero(split=='val');test=np.flatnonzero(split=='test')
    if (len(train),len(val),len(test))!=(20,5,5):raise ValueError('Expected train/val/test = 20/5/5')
    base,xstd,ystd,hist=load_checkpoint('memoryless',p/'checkpoints'/'step2A',device,1,ells)
    if hist!=1:raise RuntimeError('Expected one-frame FNO')
    for param in base.parameters():param.requires_grad_(False)
    base.eval();ystd=np.asarray(ystd,float)
    U={n:torch.as_tensor(data[f'coarse_N{n}'],device=device,dtype=torch.float32) for n in (64,128)}
    Y={n:torch.as_tensor(data[f'S_tau_N{n}']*n/ystd[None,:,None,None],device=device,dtype=torch.float32) for n in (64,128)}
    model=SpectralMemory(hidden=a.hidden,modes=a.modes,residual_scale=a.residual_scale).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
    out=p/'results'/'step2D3'; ck=p/'checkpoints'/'step2D3';out.mkdir(parents=True,exist_ok=True);ck.mkdir(parents=True,exist_ok=True)
    best=val_score(model,base,U[64],Y[64],val,ells,ystd,xstd,times,a)
    state=copy.deepcopy(model.state_dict());best_epoch=0;losses=[]
    print(f'Device={device}; frozen Step 2A FNO; initial val={best:.6g}',flush=True)
    for ep in range(1,a.epochs+1):
        model.train();loss_this=[]
        for _ in range(a.updates_per_epoch):
            ids=np.random.choice(train,min(a.batch_size,len(train)),replace=False)
            start=random.randint(0,len(times)-4-a.horizon)
            old,new=random.choice(((0,1),(0,2),(1,2)))
            switch_k=random.choice((-1,2,3))
            loss=batch_sequence(model,base,U[64],Y[64],ids,ells,ystd,xstd,times,a,start,old,new,switch_k)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_this.append(float(loss.item()))
        vs=val_score(model,base,U[64],Y[64],val,ells,ystd,xstd,times,a)
        losses.append(dict(epoch=ep,train=float(np.mean(loss_this)),validation=vs))
        if vs<best:best=vs;state=copy.deepcopy(model.state_dict());best_epoch=ep
        print(f'epoch={ep:03d} train={np.mean(loss_this):.5g} val={vs:.5g} best={best:.5g}',flush=True)
    model.load_state_dict(state);model.eval()
    torch.save(dict(state_dict=state,epoch=best_epoch,baseline='step2A/memoryless.pt',best_val=best,config=vars(a)|{'project_dir':str(p),'dataset':str(path)}),ck/'residual_memory.pt')
    rows=[];forcing=[]
    for n in (64,128):
        for mode in ('baseline','learned','gaussian','reset'):
            rows+=rollout(model,base,U[n],test,ells,ystd,xstd,times,a,n,mode)
            forcing+=closure_audit(model,base,U[n],Y[n],test,ells,xstd,times,n,mode,a.switch_time)
    write_csv(out/'rollout_by_case_time.csv',rows);write_csv(out/'forcing_by_case_time.csv',forcing);write_csv(out/'training_losses.csv',losses)
    summary=[]
    for n in (64,128):
        for mode in ('baseline','learned','gaussian','reset'):
            rr=[r for r in rows if r['N']==n and r['mode']==mode];ff=[r for r in forcing if r['N']==n and r['mode']==mode]
            tf=max(r['time'] for r in rr)
            final=np.mean([r['relative_error'] for r in rr if r['time']==tf]);post=np.mean([r['forcing_mse'] for r in ff if r['time']>=a.switch_time])
            summary.append(dict(N=n,mode=mode,mean_final_relative_error=float(final),post_switch_teacher_forced_mse=float(post),max_mass_drift=float(max(r['mass_drift'] for r in rr))))
    write_csv(out/'ablation_summary.csv',summary)
    plots(out,losses,rows,summary)
    (out/'run_config.json').write_text(json.dumps(dict(device=str(device),best_epoch=best_epoch,best_validation=best,freeze_base=True,training_resolution=64,test_resolutions=[64,128],n_test_cases=len(test),note='Baseline-anchored pilot; no proven memory improvement or architectural novelty.'),indent=2))
    for s in summary:print(s,flush=True)
    print('Output folder:',out,flush=True)
if __name__=='__main__':main()