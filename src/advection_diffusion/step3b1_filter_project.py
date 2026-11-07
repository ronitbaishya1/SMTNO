from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dstn, idstn
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def dirichlet_gaussian(field: np.ndarray, ell: float) -> np.ndarray:
    """Dirichlet heat-semigroup Gaussian filter, exp((ell^2/2) Delta_D)."""
    if ell < 0:
        raise ValueError('Filtering width must be nonnegative')
    if ell == 0:
        return field.copy()
    n = field.shape[0] - 1
    if field.shape != (n+1, n+1):
        raise ValueError('Expected a square uniform nodal grid')
    freq = np.arange(1, n, dtype=np.float64)
    lam = (np.pi**2) * (freq[:,None]**2 + freq[None,:]**2)
    spec=dstn(field[1:-1,1:-1], type=1,norm='ortho')
    result=np.zeros_like(field, dtype=np.float64)
    result[1:-1,1:-1] = idstn(spec*np.exp(-.5*ell*ell*lam),type=1,norm='ortho')
    return result


def mesh_info(n: int):
    def node(i,j):return j*(n+1)+i
    tri=[]
    for j in range(n):
        for i in range(n):
            a,b,c,d=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            tri.append((a,b,c))
            tri.append((a,c,d))
    tris=np.array(tri,dtype=np.int64)
    xx,yy=np.meshgrid(np.linspace(0,1,n+1),np.linspace(0,1,n+1))
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    return xy,tris


def prepare_projection(n:int):
    xy,tri=mesh_info(n)
    nnode=(n+1)**2
    area=0.5/n**2
    local=(area/12.0)*np.array([[2,1,1],[1,2,1],[1,1,2]],float)
    rows=np.broadcast_to(tri[:,:,None],(len(tri),3,3)).ravel()
    cols=np.broadcast_to(tri[:,None,:],(len(tri),3,3)).ravel()
    vals=np.tile(local.ravel(),len(tri))
    M=coo_matrix((vals,(rows,cols)),shape=(nnode,nnode)).tocsr()
    idx=np.arange(nnode).reshape(n+1,n+1)
    interior=idx[1:-1,1:-1].ravel()
    lu=splu(M[interior][:,interior].tocsc())
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    qpts=np.einsum('qv,tvd->tqd',bary,xy[tri])
    return xy,tri,M,interior,lu,bary,qpts,area


def project_field_from_fine(filtered:np.ndarray, projection):
    xy,tri,M,interior,lu,bary,qpts,area=projection
    nf=filtered.shape[0]-1
    x=np.linspace(0,1,nf+1)
    interp=RegularGridInterpolator((x,x),filtered,bounds_error=True,method='linear')
    # RegularGridInterpolator uses (y,x) point convention, unlike mesh_info (x,y).
    fv=interp(qpts[...,::-1].reshape(-1,2)).reshape(len(tri),3)
    btri=area/3*np.einsum('tq,qv->tv',fv,bary)
    rhs=np.bincount(tri.ravel(),weights=btri.ravel(),minlength=len(xy))
    uh=np.zeros(len(xy),float)
    uh[interior]=lu.solve(rhs[interior])
    return uh.reshape(-1,int(np.sqrt(len(xy))))


def sample_fine_on_coarse(field:np.ndarray,nc:int):
    nf=field.shape[0]-1
    if nf % nc:
        raise ValueError('Fine mesh N must be divisible by coarse N')
    stride=nf//nc
    return field[::stride,::stride].copy()


def p1_reconstruction_at_fine(coarse:np.ndarray,nf:int):
    nc=coarse.shape[0]-1
    if nf % nc:
        raise ValueError('Fine mesh N must be divisible by coarse N')
    v=np.linspace(0,nc,nf+1)
    xg,yg=np.meshgrid(v,v)
    i=np.minimum(np.floor(xg).astype(int),nc-1)
    j=np.minimum(np.floor(yg).astype(int),nc-1)
    ax=xg-i;ay=yg-j
    c00=coarse[j,i];c10=coarse[j,i+1];c11=coarse[j+1,i+1];c01=coarse[j+1,i]
    lower=(1-ax)*c00+(ax-ay)*c10+ay*c11
    upper=(1-ay)*c00+ax*c11+(ay-ax)*c01
    return np.where(ay<=ax,lower,upper)


def relative_rms(a:np.ndarray,b:np.ndarray):
    return float(np.linalg.norm((a-b).ravel())/(np.linalg.norm(b.ravel())+1e-15))


def quad_integral(u:np.ndarray):
    n=u.shape[0]-1
    w=np.ones(n+1);w[0]=.5;w[-1]=.5
    return float(np.einsum('i,j,ij->',w,w,u)/n**2)


def plot_example(out:Path,base:np.ndarray,filtered:np.ndarray,coarse:np.ndarray,t:float,ell:float,nc:int):
    nf=len(base)-1
    reconstruction=p1_reconstruction_at_fine(coarse,nf)
    err=np.abs(reconstruction-filtered)
    fig,axes=plt.subplots(1,4,figsize=(16,3.8),constrained_layout=True)
    vmin=min(float(base.min()),float(filtered.min()),float(reconstruction.min()))
    vmax=max(float(base.max()),float(filtered.max()),float(reconstruction.max()))
    for ax,field,title in zip(axes[:3],[base,filtered,reconstruction],
                              ['Fine FEM','Filtered fine FEM','Coarse P1 reconstructed']):
        im=ax.imshow(field,origin='lower',extent=(0,1,0,1),aspect='equal',vmin=vmin,vmax=vmax)
        ax.set(title=title,xlabel='x',ylabel='y')
    im2=axes[-1].imshow(err,origin='lower',extent=(0,1,0,1),aspect='equal',cmap='magma')
    axes[-1].set(title='|Coarse P1 − filtered fine|',xlabel='x',ylabel='y')
    fig.colorbar(im,ax=list(axes[:3]),shrink=.75)
    fig.colorbar(im2,ax=axes[-1],shrink=.75)
    fig.suptitle(f'Example 3: t={t:.3f}, Nfine={nf}, Ncoarse={nc}, ell={ell:g}')
    fig.savefig(out/f'scale_n{nc}_ell{ell:g}_final.png',dpi=160)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='SMTNO Step 3B.1: Dirichlet Gaussian filtering + P1 projection')
    ap.add_argument('--source',type=Path,required=True,help='Existing Example 3 field_snapshots.npz')
    ap.add_argument('--coarse',type=str,default='32,64')
    ap.add_argument('--ells',type=str,default='0.01,0.02,0.04')
    ap.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'results'/'step3B1')
    ap.add_argument('--allow-sharp',action='store_true')
    a=ap.parse_args()
    a.source=a.source.expanduser().resolve();out=a.outdir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    with np.load(a.source,allow_pickle=False) as z:
        u=np.asarray(z['u'],dtype=np.float64)
        t=np.asarray(z['t'],dtype=np.float64)
        epsilon=float(z['eps']);case=int(z['case'])
        x=np.asarray(z['x']);y=np.asarray(z['y'])
    if case!=3:
        raise ValueError('This first step is restricted to Example 3 with zero Dirichlet boundaries.')
    if epsilon<1e-3 and not a.allow_sharp:
        raise ValueError('Sharp-layer ε too small for verified reference. Use ε=0.01; --allow-sharp only for exploratory tests.')
    if u.ndim!=3 or u.shape[1]!=u.shape[2]:
        raise ValueError('Expected [time, y, x] array with uniform square grids')
    nf=u.shape[1]-1
    if x.size!=nf+1 or y.size!=nf+1 or abs(x[0])>1e-9 or abs(x[-1]-1)>1e-9:
        raise ValueError('Source grid must be uniform on [0,1]^2')
    if np.max(np.abs(np.diff(t)-np.diff(t)[0]))>1e-7:
        raise ValueError('Expected uniformly spaced saved snapshot times')
    if np.max(np.abs(u[:,:,0]))>1e-5 or np.max(np.abs(u[:,:,-1]))>1e-5 or np.max(np.abs(u[:,0,:]))>1e-5 or np.max(np.abs(u[:,-1,:]))>1e-5:
        raise ValueError('Source violates zero Dirichlet boundaries, unsuitable for this filter')
    ns=[int(n) for n in a.coarse.split(',')];ells=[float(e) for e in a.ells.split(',')]
    for n in ns:
        if n<4 or nf % n != 0 or n>=nf:
            raise ValueError(f'Ncoarse={n} must divide Nfine={nf} and be smaller')
    if any(e<=0 for e in ells) or len(set(ells))!=len(ells):
        raise ValueError('ell list must contain distinct positive widths')
    print('Source:',a.source,'Nfine:',nf,'snapshots:',len(t),'epsilon:',epsilon,flush=True)
    filts={ell:np.stack([dirichlet_gaussian(s,ell) for s in u]) for ell in ells}
    # Check heat-semigroup consistency at the two extreme widths.
    emin,emax=min(ells),max(ells)
    semigroup=np.stack([dirichlet_gaussian(s,np.sqrt(max(0,emax*emax-emin*emin))) for s in filts[emin]])
    semidefect=relative_rms(semigroup,filts[emax])
    rows=[];datasets=[]
    for nc in ns:
        proj=prepare_projection(nc)
        for ell in ells:
            f=filts[ell]
            p=np.stack([project_field_from_fine(s,proj) for s in f])
            s=np.stack([sample_fine_on_coarse(a0,nc) for a0 in f])
            coarse_grid=np.linspace(0,1,nc+1)
            outpath=out/f'coarse_n{nc}_ell{ell:g}.npz'
            np.savez_compressed(outpath, u_filtered_fine=f.astype('float32'),
                u_projected_coarse=p.astype('float32'),u_sampled_coarse=s.astype('float32'),
                x_fine=x,y_fine=y,x_coarse=coarse_grid,y_coarse=coarse_grid,
                t=t,epsilon=epsilon,ell=ell,n_fine=nf,n_coarse=nc)
            datasets.append(outpath.name)
            for k,tk in enumerate(t):
                recon=p1_reconstruction_at_fine(p[k],nf)
                rows.append({'n_fine':nf,'n_coarse':nc,'ell':ell,'time':tk,
                    'filter_difference_rel':relative_rms(f[k],u[k]),
                    'projection_reconstruction_rel':relative_rms(recon,f[k]),
                    'projection_vs_sampled_rel':relative_rms(p[k],s[k]),
                    'fine_integral':quad_integral(u[k]),
                    'filtered_integral':quad_integral(f[k]),
                    'projection_integral':quad_integral(p[k]),
                    'semigroup_defect':semidefect})
            plot_example(out,u[-1],f[-1],p[-1],float(t[-1]),ell,nc)
            print(f'Created {outpath.name}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'filter_projection_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for (nc,ell),g in df.groupby(['n_coarse','ell']):
        ax.plot(g.time,g.projection_reconstruction_rel,label=f'N={nc}, ell={ell:g}')
    ax.set(xlabel='Physical time',ylabel='Relative nodal-grid reconstruction error',
           title='Coarse P1 projection versus filtered fine FEM')
    ax.grid(alpha=.3);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(out/'projection_error_vs_time.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for ell in ells:
        g=df[(df.ell==ell)&(df.n_coarse==ns[0])]
        ax.plot(g.time,g.filtered_integral,label=f'filtered ell={ell:g}')
    g=df[df.n_coarse==ns[0]]
    ax.plot(g[g.ell==ells[0]].time,g[g.ell==ells[0]].fine_integral,'--',label='fine field')
    ax.set(xlabel='Physical time',ylabel='Domain integral',
           title='Dirichlet Gaussian filtering: boundary can exchange scalar quantity')
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'filtered_integrals.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(ells),figsize=(4*len(ells),3.9),constrained_layout=True,squeeze=False)
    for ax,ell in zip(axes[0],ells):
        im=ax.imshow(filts[ell][-1],origin='lower',extent=(0,1,0,1),vmin=0,vmax=max(1e-12,float(np.max(u[-1]))))
        ax.set(title=f'ell={ell:g}',xlabel='x',ylabel='y')
    fig.colorbar(im,ax=list(axes[0]),shrink=.75)
    fig.suptitle('Same fine FEM state filtered at three physical scales')
    fig.savefig(out/'three_physical_filters.png',dpi=160);plt.close(fig)
    config={'source':str(a.source),'n_fine':nf,'coarse':ns,'ells':ells,
        'epsilon':epsilon,'snapshots':len(t),'semigroup_defect':semidefect,
        'filter':'Dirichlet heat-semigroup Gaussian; exp(0.5 * ell^2 * Delta_D)',
        'projection':'P1 finite-element L2 mass projection using 3-point triangle quadrature and bilinear reconstruction of fine-grid filtered data',
        'error_metric':'node-grid relative RMS for coarse P1 reinterpolation; not exact FEM mass norm',
        'boundary_caution':'Dirichlet filtering does not conserve the domain integral',
        'datasets':datasets}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print(f'Finished. Semigroup defect = {semidefect:.3e}; saved to {out}',flush=True)


if __name__=='__main__':main()