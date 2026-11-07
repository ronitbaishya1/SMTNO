from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3b1_filter_project import dirichlet_gaussian
from step3b2_weak_closure import BX, BY, REACTION


def csv_values(text, cast=float):
    vals = [cast(s.strip()) for s in text.split(',') if s.strip()]
    if not vals or len(vals) != len(set(vals)):
        raise ValueError('Expected unique comma-separated values')
    return vals


def load_case(file: Path, case_id: int):
    if not file.exists():
        raise FileNotFoundError(file)
    with np.load(file, allow_pickle=False) as z:
        u = np.asarray(z['u'], dtype=np.float64)
        t = np.asarray(z['t'], dtype=np.float64)
        eps = float(z['eps'])
        if not bool(z['source_zero']) or int(z['case_id']) != case_id:
            raise ValueError(f'{file} must be source-free case {case_id}')
    if u.ndim != 3 or u.shape[1] != u.shape[2] or u.shape[0] != len(t):
        raise ValueError(f'Unexpected [time, y, x] data: {file}')
    if not np.isfinite(u).all() or len(t) < 3 or eps <= 0:
        raise ValueError(f'Invalid fields or parameters: {file}')
    if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-5, atol=1e-8):
        raise ValueError('Saved times must be uniform')
    edge = max(np.max(np.abs(u[:, 0])), np.max(np.abs(u[:, -1])),
               np.max(np.abs(u[:, :, 0])), np.max(np.abs(u[:, :, -1])))
    if edge > 1e-4:
        raise ValueError('Dirichlet case does not have zero boundary nodes')
    return u, t, eps


def physical_parts_dirichlet(u: np.ndarray, eps: float):
    n = u.shape[0] - 1
    h = 1.0/n
    adv = np.zeros_like(u)
    diff = np.zeros_like(u)
    adv[1:-1, 1:-1] = (BX*(u[1:-1, 2:] - u[1:-1, :-2]) +
                            BY*(u[2:, 1:-1] - u[:-2, 1:-1]))/(2*h)
    diff[1:-1, 1:-1] = -eps*(u[1:-1, 2:] + u[1:-1, :-2] +
                                    u[2:, 1:-1] + u[:-2, 1:-1] - 4*u[1:-1, 1:-1])/h**2
    react = REACTION*u.copy()
    react[[0, -1], :] = 0
    react[:, [0, -1]] = 0
    return adv, diff, react


def periodic_controls(u0: np.ndarray, time: float, eps: float, ell: float):
    # Mathematical torus control: same sampled initial field, DIFFERENT boundary condition.
    # The periodic state is evolved with the exact Fourier semigroup (not a FreeFEM run).
    n = u0.shape[0]
    k = 2*np.pi*np.fft.fftfreq(n, d=1/n)
    kx, ky = np.meshgrid(k, k, indexing='xy')
    k2 = kx*kx + ky*ky
    p = BX*kx + BY*ky
    state_hat = np.fft.fft2(u0)*np.exp((-eps*k2 - REACTION - 1j*p)*time)
    filt_hat = np.exp(-.5*ell**2*k2)
    state = np.fft.ifft2(state_hat).real
    filtered = np.fft.ifft2(state_hat*filt_hat).real

    def oper(a):
        h = 1/n
        lap = (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
               np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4*a)/h**2
        gx = (np.roll(a, -1, 1) - np.roll(a, 1, 1))/(2*h)
        gy = (np.roll(a, -1, 0) - np.roll(a, 1, 0))/(2*h)
        return -eps*lap + BX*gx + BY*gy + REACTION*a

    op_g = oper(filtered)
    g_op = np.fft.ifft2(np.fft.fft2(oper(state))*filt_hat).real
    defect = op_g - g_op
    # Also verify the continuum spectral operator and exact time derivative.
    spectral_A = eps*k2 + REACTION + 1j*p
    spectral_comm = np.fft.ifft2(spectral_A*(filt_hat*state_hat) -
                                 filt_hat*(spectral_A*state_hat)).real
    # u_t + A u = 0, evaluated using exact Fourier semigroup.
    exact_resid = np.fft.ifft2((-spectral_A)*state_hat*filt_hat +
                               spectral_A*state_hat*filt_hat).real
    return dict(periodic_fd_rms=rms(defect),
                periodic_fd_rel=rms(defect)/(rms(op_g)+1e-12),
                periodic_spectral_rms=rms(spectral_comm),
                periodic_exact_residual_rms=rms(exact_resid),
                periodic_state_rms=rms(state), periodic_filtered_rms=rms(filtered))


def rms(v):
    return float(np.sqrt(np.mean(np.square(np.asarray(v, dtype=float)))))


def boundary_fraction(v, width):
    n = v.shape[0]-1
    coord = np.linspace(0, 1, n+1)
    X,Y = np.meshgrid(coord, coord, indexing='xy')
    mask = (np.minimum.reduce((X, 1-X, Y, 1-Y)) <= width+1e-12)
    # Only free DOFs; no boundary values are included in energy.
    mask = mask[1:-1,1:-1]
    w = v[1:-1,1:-1]**2
    return float(np.sum(w[mask])/(np.sum(w)+1e-28))


def plot_results(df: pd.DataFrame, out: Path, example):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label,col in [('Dirichlet total','dirichlet_commutator_rel'),
                      ('Dirichlet advection','dirichlet_advection_rel'),
                      ('Periodic finite difference','periodic_fd_rel')]:
        g = df.groupby('ell')[col]
        mean = g.mean().sort_index()
        ax.plot(mean.index, np.maximum(mean.to_numpy(),1e-17), 'o-', label=label)
    ax.set(xlabel='Filter width ell',ylabel='Relative commutator magnitude',
           title='Effect of boundary conditions on filtering/operator commutation')
    ax.set_yscale('log'); ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'commutator_by_filter.png',dpi=180);plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,4.6))
    sub = df.groupby('ell').agg({'dirichlet_commutator_boundary_fraction':'mean'}).reset_index()
    ax.plot(sub.ell,sub.dirichlet_commutator_boundary_fraction,'o-')
    ax.set(xlabel='Filter width ell',ylabel='Fraction of squared commutator in boundary strip',
           title='Dirichlet commutator localization',ylim=(0,1.05))
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'commutator_boundary_fraction.png',dpi=180);plt.close(fig)
    if example is not None:
        u, gu, a, aa, diff = example
        fig,axes=plt.subplots(1,4,figsize=(15,3.7),constrained_layout=True)
        for j,(field,title) in enumerate([(u,'Dirichlet state'),(gu,'Filtered state'),
                                          (a,'Total commutator'),(aa,'Advection commutator')]):
            lim=max(float(np.max(np.abs(field))),1e-12)
            if j<2:
                im=axes[j].imshow(field,origin='lower',extent=(0,1,0,1),cmap='viridis')
            else:
                im=axes[j].imshow(field,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-lim,vmax=lim)
            axes[j].set(title=title,xlabel='x',ylabel='y')
            fig.colorbar(im,ax=axes[j],shrink=.8)
        fig.suptitle('Case 000, final time, largest ell: discrete FD commutator diagnostic')
        fig.savefig(out/'case000_commutator_maps.png',dpi=160);plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Step 3B.6A/C: boundary and commutator controls')
    ap.add_argument('--indir',type=Path,default=Path('results/step3B3'))
    ap.add_argument('--outdir',type=Path,default=Path('results/step3B6/commutator'))
    ap.add_argument('--cases',type=int,default=2)
    ap.add_argument('--ells',default='0.01,0.02,0.04')
    ap.add_argument('--times',default='0.02,0.10,0.20')
    ap.add_argument('--boundary-width',type=float,default=.10)
    args=ap.parse_args()
    if not 0<args.boundary_width<.5 or args.cases<1:
        ap.error('Need positive cases and 0 < boundary-width < 0.5')
    ells=csv_values(args.ells)
    targets=csv_values(args.times)
    if min(ells)<=0 or min(targets)<0:
        ap.error('Filter widths must be positive and requested times nonnegative')
    src=args.indir.expanduser().resolve()
    out=args.outdir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    rows=[];example=None
    for cid in range(args.cases):
        u,t,eps=load_case(src/f'case_{cid:03d}'/'field_snapshots.npz',cid)
        n=u.shape[1]-1
        t0=float(t[0]);u0=np.asarray(u[0,:-1,:-1],float)
        for target in targets:
            j=int(np.argmin(np.abs(t-target)))
            if abs(float(t[j])-target)>max(1e-8,.51*(t[1]-t[0])):
                raise ValueError(f'Target time {target} absent in case {cid}')
            frame=u[j]
            base_parts=physical_parts_dirichlet(frame,eps)
            for ell in ells:
                gf=dirichlet_gaussian(frame,ell)
                Lgf=physical_parts_dirichlet(gf,eps)
                gL=[dirichlet_gaussian(p,ell) for p in base_parts]
                comps=[Lgf[k]-gL[k] for k in range(3)]
                comm=sum(comps)
                den=rms(sum(Lgf)[1:-1,1:-1])+1e-12
                per=periodic_controls(u0,float(t[j]-t0),eps,ell)
                rows.append(dict(case_id=cid,time=float(t[j]),ell=ell,n_fine=n,
                    dirichlet_commutator_rms=rms(comm[1:-1,1:-1]),
                    dirichlet_commutator_rel=rms(comm[1:-1,1:-1])/den,
                    dirichlet_advection_rel=rms(comps[0][1:-1,1:-1])/den,
                    dirichlet_diffusion_rel=rms(comps[1][1:-1,1:-1])/den,
                    dirichlet_reaction_rel=rms(comps[2][1:-1,1:-1])/den,
                    dirichlet_commutator_boundary_fraction=boundary_fraction(comm,args.boundary_width),
                    **per))
                if cid==0 and j==len(t)-1 and ell==max(ells):
                    example=(frame,gf,comm,comps[0],comps[1])
        print(f'Case {cid:03d}: commutator audit complete',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'commutator_metrics.csv',index=False)
    summary=df.groupby('ell').agg({
        'dirichlet_commutator_rel':'mean',
        'dirichlet_advection_rel':'mean',
        'dirichlet_diffusion_rel':'mean',
        'periodic_fd_rel':'mean',
        'dirichlet_commutator_boundary_fraction':'mean'}).reset_index()
    summary.to_csv(out/'commutator_summary.csv',index=False)
    plot_results(df,out,example)
    (out/'config.json').write_text(json.dumps(dict(
        cases=args.cases,ells=ells,times=targets,boundary_width=args.boundary_width,
        PDE='u_t + b.grad(u) - epsilon.Laplacian(u) + u = 0',
        Dirichlet='FreeFEM nodal trajectory with discrete FD operator, Dirichlet sine heat filter',
        periodic='Same initial samples, evolved by exact periodic Fourier semigroup; distinct boundary condition, not a second FreeFEM trajectory',
        interpretation='Commutator audit isolates filter/operator compatibility; does not identify temporal memory or P1 FEM residual'),indent=2))
    print('Saved:',out,flush=True)


if __name__=='__main__':
    main()