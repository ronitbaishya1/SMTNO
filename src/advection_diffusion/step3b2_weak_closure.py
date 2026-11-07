from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from step3b1_filter_project import (
    dirichlet_gaussian, prepare_projection, project_field_from_fine,
)

PI = np.pi
BX, BY, REACTION = np.cos(PI / 3), np.sin(PI / 3), 1.0


def analytical_fields(x: np.ndarray, y: np.ndarray, time: float, eps: float):
    xx, yy = np.meshgrid(x, y, indexing='xy')
    q = 0.5*np.sin(PI*xx)*np.sin(PI*yy)
    qx = 0.5*PI*np.cos(PI*xx)*np.sin(PI*yy)
    qy = 0.5*PI*np.sin(PI*xx)*np.cos(PI*yy)
    zz = (xx + yy - time - 0.5)/np.sqrt(eps)
    tt = np.tanh(zz)
    sech2 = 1.0-tt**2
    u = q*(1+tt)
    ut = -q*sech2/np.sqrt(eps)
    ux = qx*(1+tt)+q*sech2/np.sqrt(eps)
    uy = qy*(1+tt)+q*sech2/np.sqrt(eps)
    lap = -2*PI**2*q*(1+tt)+2*(qx+qy)*sech2/np.sqrt(eps)-4*q*tt*sech2/eps
    f = ut-eps*lap+BX*ux+BY*uy+REACTION*u
    return u, ut, f


def fem_operators(n: int, eps: float, projection):
    xy, tri, M, interior, lu, bary, qpts, area = projection
    xyz = xy[tri]
    x0, y0 = xyz[:, 0, 0], xyz[:, 0, 1]
    x1, y1 = xyz[:, 1, 0], xyz[:, 1, 1]
    x2, y2 = xyz[:, 2, 0], xyz[:, 2, 1]
    det = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
    if np.min(np.abs(det))<1e-14:
        raise RuntimeError('Degenerate triangle')
    gx = np.stack([y1-y2, y2-y0, y0-y1], axis=1)/det[:,None]
    gy = np.stack([x2-x1, x0-x2, x1-x0], axis=1)/det[:,None]
    diffusion = eps*(gx[:,:,None]*gx[:,None,:]+gy[:,:,None]*gy[:,None,:])*(np.abs(det)/2)[:,None,None]
    velocity_grad = BX*gx+BY*gy
    adv = (np.abs(det)/2)[:,None,None]*velocity_grad[:,None,:]/3
    local_mass = (np.abs(det)/2/12)[:,None,None]*np.array([[2,1,1],[1,2,1],[1,1,2]])[None,:,:]
    local_A = diffusion+adv+REACTION*local_mass
    rows = np.broadcast_to(tri[:,:,None],local_A.shape).ravel()
    cols = np.broadcast_to(tri[:,None,:],local_A.shape).ravel()
    A = coo_matrix((local_A.ravel(),(rows,cols)),shape=M.shape).tocsr()
    return M,A,interior,lu,tri,bary,qpts,np.abs(det)/2


def source_vector(f: np.ndarray, fine_x: np.ndarray, operators):
    M,A,I,lu,tri,bary,qpts,areas = operators
    itp = RegularGridInterpolator((fine_x,fine_x), f, bounds_error=True)
    sample = itp(qpts[...,::-1].reshape(-1,2)).reshape(len(tri),len(bary))
    local = areas[:,None]*np.einsum('tq,qi->ti',sample,bary)/3
    return np.bincount(tri.ravel(),weights=local.ravel(),minlength=M.shape[0])[I]


def mass_norm(v: np.ndarray, Mii):
    return float(np.sqrt(max(0,float(v @ (Mii @ v)))))


def compute_one(dataset_path: Path, outdir: Path):
    with np.load(dataset_path, allow_pickle=False) as z:
        u_fem = np.asarray(z['u_projected_coarse'],dtype=float)
        time = np.asarray(z['t'],dtype=float)
        xfine = np.asarray(z['x_fine'],dtype=float)
        nf,nc = int(z['n_fine']),int(z['n_coarse'])
        ell,eps = float(z['ell']),float(z['epsilon'])
    if eps < 1e-3:
        raise ValueError('Use the verified smoother benchmark, epsilon >= 0.001')
    if len(time)<4 or not np.allclose(np.diff(time),np.diff(time)[0],atol=1e-8):
        raise ValueError('Need at least four uniformly spaced snapshots')
    if u_fem.shape!=(len(time),nc+1,nc+1) or len(xfine)!=nf+1:
        raise ValueError(f'Unexpected dataset shape: {dataset_path}')
    dt = float(time[1]-time[0])
    proj = prepare_projection(nc)
    ops = fem_operators(nc,eps,proj)
    M,A,I,lu,_,_,_,_ = ops
    Mii=M[I][:,I].tocsr()
    Aii=A[I][:,I].tocsr()
    u_exact=[]; ut_exact=[]; forcing=[]
    for ti in time:
        ex, ext, f = analytical_fields(xfine,xfine,float(ti),eps)
        u_exact.append(project_field_from_fine(dirichlet_gaussian(ex,ell),proj))
        ut_exact.append(project_field_from_fine(dirichlet_gaussian(ext,ell),proj))
        forcing.append(source_vector(dirichlet_gaussian(f,ell),xfine,ops))
    u_exact=np.asarray(u_exact);ut_exact=np.asarray(ut_exact)
    forcing=np.asarray(forcing)
    npoints=len(I)
    uf=np.stack([v.ravel()[I] for v in u_fem])
    ue=np.stack([v.ravel()[I] for v in u_exact])
    ute=np.stack([v.ravel()[I] for v in ut_exact])
    # This difference is diagnostic, not a trainable memory label.
    duf_central=np.gradient(uf,dt,axis=0,edge_order=2)
    duf_bdf2=np.empty_like(uf)
    duf_bdf2[0]=(uf[1]-uf[0])/dt
    duf_bdf2[1]=(uf[1]-uf[0])/dt
    duf_bdf2[2:]=(3*uf[2:]-4*uf[1:-1]+uf[:-2])/(2*dt)
    closure_exact=[];closure_fem=[];closure_bdf=[];uncorrected=[]
    for k in range(len(time)):
        ce = Mii @ ute[k]+Aii @ ue[k]-forcing[k]
        cf = Mii @ duf_central[k]+Aii @ uf[k]-forcing[k]
        cb = Mii @ duf_bdf2[k]+Aii @ uf[k]-forcing[k]
        closure_exact.append(lu.solve(ce))
        closure_fem.append(lu.solve(cf))
        closure_bdf.append(lu.solve(cb))
        uncorrected.append(lu.solve(forcing[k]-Aii@ue[k]))
    closure_exact=np.asarray(closure_exact)
    closure_fem=np.asarray(closure_fem)
    closure_bdf=np.asarray(closure_bdf)
    uncorrected=np.asarray(uncorrected)
    # Store nodewise rates for visualization, but metrics use FE mass norm.
    def extend(a):
        result=np.zeros((len(time),(nc+1)**2),dtype=np.float32)
        result[:,I]=a.astype(np.float32)
        return result.reshape(len(time),nc+1,nc+1)
    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outdir/f'weak_closure_n{nc}_ell{ell:g}.npz',
        time=time,x=np.linspace(0,1,nc+1),y=np.linspace(0,1,nc+1),
        u_projected_fem=u_fem.astype(np.float32),
        u_projected_exact=u_exact.astype(np.float32),
        closure_rate_exact=extend(closure_exact),
        closure_rate_fem_central=extend(closure_fem),
        closure_rate_fem_bdf2=extend(closure_bdf),
        reference_dt_exact=extend(ute),
        unclosed_rate=extend(uncorrected),
        epsilon=eps,ell=ell,n_coarse=nc,
        note='M_I dot(u)_I + A_I u_I - F_I(G_ell f); closure_rate = inv(M_II) residual, zero boundary; analytic source. Not a Mori-Zwanzig memory label.')
    rows=[]
    for k,tk in enumerate(time):
        norme=mass_norm(closure_exact[k],Mii)
        rows.append({'n_coarse':nc,'ell':ell,'time':tk,
            'closure_exact_mass_norm':norme,
            'closure_fem_mass_norm':mass_norm(closure_fem[k],Mii),
            'fem_vs_exact_closure_rel':mass_norm(closure_fem[k]-closure_exact[k],Mii)/(norme+1e-12),
            'bdf2_vs_exact_closure_rel':mass_norm(closure_bdf[k]-closure_exact[k],Mii)/(norme+1e-12) if k>=2 else np.nan,
            'fem_state_vs_exact_rel':mass_norm(uf[k]-ue[k],Mii)/(mass_norm(ue[k],Mii)+1e-12),
            'reference_dt_mass_norm':mass_norm(ute[k],Mii),
            'unclosed_dt_error_rel':mass_norm(uncorrected[k]-ute[k],Mii)/(mass_norm(ute[k],Mii)+1e-12)})
    # Side-by-side figures for representative snapshot.
    k=len(time)//2
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    arrs=[extend(closure_exact)[k],extend(closure_fem)[k],extend(closure_fem-closure_exact)[k]]
    titles=['Exact-state filtered closure','FEM-state finite-difference closure','FEM closure minus exact-state closure']
    maxv=max(np.max(abs(a)) for a in arrs[:2])
    for ax,v,title in zip(axes,arrs,titles):
        im=ax.imshow(v,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-maxv,vmax=maxv)
        ax.set(xlabel='x',ylabel='y',title=title)
    fig.colorbar(im,ax=axes,shrink=.75)
    fig.suptitle(f'Weak residual rate at t={time[k]:g}, N={nc}, ell={ell:g}')
    fig.savefig(outdir/f'closure_maps_n{nc}_ell{ell:g}.png',dpi=160)
    plt.close(fig)
    return rows


def main():
    ap=argparse.ArgumentParser(description='SMTNO Step 3B2: weak-form coarse closure with exact-reference comparison')
    ap.add_argument('--indir',type=Path,default=Path('results/step3B1'))
    ap.add_argument('--outdir',type=Path,default=Path('results/step3B2'))
    ap.add_argument('--coarse',type=str,default='32,64')
    ap.add_argument('--ells',type=str,default='0.01,0.02,0.04')
    args=ap.parse_args()
    src=args.indir.expanduser().resolve();out=args.outdir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    inputs=[]
    for nc in [int(x) for x in args.coarse.split(',')]:
        for ell in [float(x) for x in args.ells.split(',')]:
            p=src/f'coarse_n{nc}_ell{ell:g}.npz'
            if not p.exists(): raise FileNotFoundError(p)
            inputs.append(str(p))
            rows.extend(compute_one(p,out))
            print(f'Processed N={nc}, ell={ell:g}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'weak_closure_metrics.csv',index=False)
    for metric,ylabel,fname in [
        ('closure_exact_mass_norm','Mass-norm of exact-state closure rate','closure_norm_vs_time.png'),
        ('fem_vs_exact_closure_rel','Relative closure discrepancy','closure_reference_discrepancy.png'),
        ('unclosed_dt_error_rel','Relative unclosed derivative error','unclosed_tendency_error.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for (nc,ell),g in df.groupby(['n_coarse','ell']):
            ax.plot(g.time,g[metric],label=f'N={nc}, ell={ell:g}')
        ax.set(xlabel='Physical time',ylabel=ylabel,title=metric.replace('_',' '))
        ax.grid(alpha=.3);ax.legend(fontsize=8,ncol=2)
        if metric=='fem_vs_exact_closure_rel':ax.set_yscale('symlog',linthresh=.01)
        fig.tight_layout();fig.savefig(out/fname,dpi=160);plt.close(fig)
    conf={'input_datasets':inputs,'coarse_operator':'P1 Galerkin mass + epsilon diffusion + constant-velocity advective + reaction','forcing':'analytical manufactured f, filtered using the same Dirichlet heat-semigroup Gaussian','exact_reference':'analytic exact u and u_t filtered then P1 projected','finite_element_reference':'prior Step 3A SUPG solution, centered / BDF2 time derivatives on stored snapshots','caution':'Exact-state closure includes projection, boundary/commutator, quadrature effects. It is not an identified physical memory kernel. The FEM-derived closure also includes FEM and temporal discretization errors.','not_a_full_closure_validation':True}
    (out/'config.json').write_text(json.dumps(conf,indent=2))
    print('Saved to:',out,flush=True)

if __name__=='__main__':main()