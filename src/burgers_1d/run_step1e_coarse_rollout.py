#!/usr/bin/env python3
"""SMTNO Step 1E: coarse Burgers rollout and exact-closure diagnostic.

Uses the existing Step 1C dataset (NO new FreeFEM++ run):
  SMTNO/datasets/processed/burgers_1d/fine_filtered_and_stress.npz
  SMTNO/datasets/processed/burgers_1d/coarse_FEM_ell{ell:g}_N{N}.npz

Compares:
 A) Coarse periodic P1 Galerkin Burgers time evolution WITHOUT closure
 B) Same coarse solver FORCED BY exact subfilter stress from Step 1C
    (diagnostic oracle, not deployable because it uses the fine solution)
against the periodic P1 projection of physically filtered fine FEM states.

Saves CSV diagnostics and automatically generated plots / a non-overlapping
flowchart to SMTNO/results/burgers_1d/step1E/.

Run:
  python FEM/burgers_1d/run_step1e_coarse_rollout.py

Requires numpy, pandas, matplotlib; no SciPy necessary.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / 'datasets' / 'processed' / 'burgers_1d'
OUT = ROOT / 'results' / 'burgers_1d' / 'step1E'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = PROCESSED / 'fine_filtered_and_stress.npz'
NU = 0.02
# Use the SAME timestep as the fine FEM in Step 1C.
DT = 0.000125


def mass_apply(a: np.ndarray, h: float) -> np.ndarray:
    """M a, periodic P1 consistent mass matrix."""
    return (h / 6.0) * (4.0*a + np.roll(a, 1) + np.roll(a, -1))


def mass_norm(v: np.ndarray, h: float) -> float:
    return float(np.sqrt(max(0.0, np.dot(v, mass_apply(v, h)))))


def relative_mass_l2(pred: np.ndarray, truth: np.ndarray, h: float) -> float:
    return mass_norm(pred-truth, h) / max(mass_norm(truth, h), 1e-14)


def nonlinear_rhs(a: np.ndarray) -> np.ndarray:
    """C_i(a) = integral (u_h^2/2) * phi_i' dx, integrated exactly for P1.

    Edge i is from node i to node i+1. Its mean flux is
    (a_i^2 + a_i*a_{i+1} + a_{i+1}^2)/6.
    """
    b = np.roll(a, -1)
    flux_mean = (a*a + a*b + b*b)/6.0
    return np.roll(flux_mean, 1) - flux_mean


def stress_rhs_on_coarse(tau_fine: np.ndarray, ncoarse: int) -> np.ndarray:
    """S_i = integral tau(x) phi_i'(x) dx, with fine-grid composite trapezoid.

    This is a diagnostic approximation to the FEM forcing from an oracle stress.
    It is not the exact closure for the discrete projected dynamics.
    """
    n = len(tau_fine)
    if n % ncoarse:
        raise ValueError('Fine spatial resolution must be divisible by coarse N.')
    ratio = n // ncoarse
    # One fine-grid sample at each endpoint plus interior samples per element.
    starts = np.arange(ncoarse)*ratio
    elements = tau_fine[(starts[:, None] + np.arange(ratio+1)[None, :]) % n]
    # Element-wise average = integral(tau) / element_length.
    edge_mean = (0.5*elements[:, 0] + elements[:, 1:-1].sum(axis=1)
                 + 0.5*elements[:, -1]) / ratio
    return np.roll(edge_mean, 1) - edge_mean


def periodic_step(a: np.ndarray, dt: float, nu: float,
                  tau_rhs: np.ndarray | None = None) -> np.ndarray:
    """(M/dt + nu K) a_new = M a_old/dt + C(a_old) + S_tau.

    M,K circulant on periodic uniform 1D mesh; FFT gives exact discrete solve
    to floating-point precision (not a Fourier pseudospectral PDE solver).
    """
    n = len(a)
    h = 1.0/n
    ang = 2*np.pi*np.arange(n)/n
    m_eigs = h*(2.0/3.0 + np.cos(ang)/3.0)
    k_eigs = (2.0 - 2.0*np.cos(ang))/h
    rhs = mass_apply(a,h)/dt + nonlinear_rhs(a)
    if tau_rhs is not None:
        rhs = rhs + tau_rhs
    new = np.fft.ifft(np.fft.fft(rhs)/(m_eigs/dt + nu*k_eigs)).real
    return new


def time_interp_field(times: np.ndarray, field: np.ndarray, t: float) -> np.ndarray:
    """Piecewise-linear interpolation of stored fine-scale stress in physical time."""
    if t <= times[0]:
        return field[0]
    if t >= times[-1]:
        return field[-1]
    hi = int(np.searchsorted(times, t, side='right'))
    lo = hi - 1
    a = (t-times[lo])/(times[hi]-times[lo])
    return (1-a)*field[lo] + a*field[hi]


def run_rollout(times: np.ndarray, initial: np.ndarray,
                tau_time_series: np.ndarray | None,
                fine_times: np.ndarray, dt: float, nu: float) -> np.ndarray:
    """Return coarse FEM coefficients at every requested Step 1C snapshot."""
    out = np.empty((len(times),len(initial)))
    a = initial.copy()
    out[0] = a
    ncoarse = len(a)
    t0 = float(times[0])
    t1 = float(times[-1])
    steps = int(round((t1-t0)/dt))
    if abs(steps*dt - (t1-t0)) > 1e-9:
        raise ValueError('T/dt is not an integer; use the Step 1C time increment.')
    next_snapshot = 1
    for step in range(steps):
        current_time = t0 + step*dt
        src = None
        if tau_time_series is not None:
            tau = time_interp_field(fine_times,tau_time_series,current_time)
            src = stress_rhs_on_coarse(tau,ncoarse)
        a = periodic_step(a,dt,nu,src)
        t_new = t0+(step+1)*dt
        # Step 1C snapshots sit on multiples of DT; detect rounding safely.
        while next_snapshot < len(times) and t_new >= times[next_snapshot]-1e-10:
            if abs(t_new-times[next_snapshot]) > 1e-9:
                raise ValueError('Requested snapshot is not aligned with the chosen timestep.')
            out[next_snapshot] = a
            next_snapshot += 1
    if next_snapshot != len(times):
        raise RuntimeError('Failed to produce all snapshots.')
    return out


def plot_workflow():
    stages = [
        '1A–1B\nFEM solver + convergence',
        '1C\nFine states + Gaussian filtering',
        '1D\nFEM eigenmodes + scale checks',
        '1E\nReuse projected filtered FEM states',
        'Two coarse rollouts:\nno closure / oracle stress',
        'Compare trajectories, errors,\nand mass conservation',
        'Next: multiple initial conditions,\nthen learn a closure'
    ]
    fig, ax = plt.subplots(figsize=(10.7, 10.4))
    ax.set_xlim(0, 10);ax.set_ylim(0, 10);ax.axis('off')
    ax.set_title('SMTNO: Step 1E — measuring the missing coarse-scale dynamics',pad=18,fontsize=14)
    y_positions=np.linspace(9,1,len(stages))
    x=2.0; width=6.0; height=.81
    for i,(label,y) in enumerate(zip(stages,y_positions)):
        rect=FancyBboxPatch((x,y-height/2),width,height,
             boxstyle='round,pad=0.10',facecolor='white',edgecolor='black',lw=1.4)
        ax.add_patch(rect)
        ax.text(x+width/2,y,label,ha='center',va='center',fontsize=10.3)
        if i < len(stages)-1:
            y_lower=y_positions[i+1]
            arrow=FancyArrowPatch((5,y-height/2-.08),(5,y_lower+height/2+.08),
                                  arrowstyle='-|>', mutation_scale=13,lw=1.4,color='black')
            ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(OUT/'step1A_to_step1E_workflow.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def make_plots(times, results, mesh_choice=64):
    # Error trajectories for each ell and N.
    fig,ax=plt.subplots(figsize=(9,5.3))
    for (ell,nc), r in sorted(results.items()):
        if nc != mesh_choice: continue
        ax.plot(times,r['err_no'],label=f'ell={ell:g} no closure')
        ax.plot(times,r['err_oracle'],ls='--',label=f'ell={ell:g} oracle stress')
    ax.set(xlabel='Physical time', ylabel='Relative FEM mass-norm error',
           title=f'Coarse Burgers evolution error (N={mesh_choice})')
    ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'coarse_rollout_error_vs_time.png',dpi=180);plt.close(fig)

    # Final-time profile at ell=0.04, N=64: reference vs no closure vs oracle.
    ell_choice=max(k[0] for k in results)
    r=results[(ell_choice,mesh_choice)]
    x=np.arange(mesh_choice)/mesh_choice
    fig,ax=plt.subplots(figsize=(9,5.3))
    ax.plot(x,r['ref'][-1],lw=2.5,label='Projected filtered reference',marker='.',ms=3)
    ax.plot(x,r['unclosed'][-1],label='Coarse Burgers, no closure')
    ax.plot(x,r['oracle'][-1],label='Coarse Burgers + exact stress from fine field')
    ax.set(xlabel='x',ylabel='FEM coefficients / nodal value',
           title=f'Final coarse field: ell={ell_choice:g}, N={mesh_choice}')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'final_coarse_profile_comparison.png',dpi=180);plt.close(fig)

    # Relative effect of closure for different scales / N, at final time.
    fig,ax=plt.subplots(figsize=(8,5.3))
    names=[];a=[];b=[]
    for (ell,nc),r in sorted(results.items()):
        names.append(f'ell={ell:g}\nN={nc}')
        a.append(r['err_no'][-1]);b.append(r['err_oracle'][-1])
    xx=np.arange(len(names)); w=.37
    ax.bar(xx-w/2,a,w,label='No closure')
    ax.bar(xx+w/2,b,w,label='Oracle stress')
    ax.set_xticks(xx);ax.set_xticklabels(names)
    ax.set(ylabel='Relative FEM mass-norm error at t=0.2',
           title='Final rollout error by physical scale and FEM resolution')
    ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'final_error_by_scale.png',dpi=180);plt.close(fig)

    # Independently useful closure target: S_tau(t), scaled to per-step L2 rhs.
    fig,ax=plt.subplots(figsize=(8.7,5.2))
    for (ell,nc),r in sorted(results.items()):
        if nc==mesh_choice:
            ax.plot(times,r['stress_force_norm'],label=f'ell={ell:g}')
    ax.set(xlabel='Physical time',ylabel='Euclidean norm of FEM forcing S_tau',
           title=f'True subfilter forcing on N={mesh_choice} coarse FEM')
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'subfilter_weak_forcing_vs_time.png',dpi=180);plt.close(fig)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f'Cannot find {INPUT}. Run Step 1C first on your Mac; do not use Step 1D CSV in its place.')
    fine=np.load(INPUT)
    for key in ('times','x','ell','subfilter_stress'):
        if key not in fine:
            raise KeyError(f'Expected Step 1C NPZ array: {key}')
    times=np.asarray(fine['times'],float)
    xfine=np.asarray(fine['x'],float)
    ell_values=np.asarray(fine['ell'],float)
    stress=np.asarray(fine['subfilter_stress'],float)
    if stress.shape!=(len(times),len(ell_values),len(xfine)):
        raise ValueError('Unexpected shape for subfilter_stress in Step 1C data.')
    if not (np.all(np.isfinite(stress)) and np.all(np.isfinite(times))):
        raise ValueError('Fine reference contains NaN or infinity; check Step 1C data.')
    if not (np.allclose(np.diff(times),np.diff(times)[0]) and times[0]==0):
        raise ValueError('Expected equally spaced Step 1C physical times starting at 0.')

    print(f'[INFO] Fine reference: {len(times)} snapshots, {len(xfine)} grid points')
    print('[INFO] Oracle forcing uses ground-truth fine-scale stress. Not a trained model.')
    diag=[]; results={};summary=[]

    for ell in ell_values:
        idx=int(np.argmin(abs(ell_values-ell)))
        tau_time_series=stress[:,idx,:]
        for ncoarse in (64,128):
            path=PROCESSED/f'coarse_FEM_ell{ell:g}_N{ncoarse}.npz'
            if not path.exists():
                raise FileNotFoundError(f'Expected Step 1C coarse projection data: {path}')
            coarse=np.load(path)
            a_ref=np.asarray(coarse['coeffs'],float)
            if a_ref.shape!=(len(times),ncoarse):
                raise ValueError(f'Unexpected coarse shape for {path}')
            if not np.all(np.isfinite(a_ref)):
                raise ValueError(f'Coarse reference contains NaN or infinity: {path}')
            if not np.allclose(coarse['times'],times):
                raise ValueError(f'Time stamps for {path} do not match Step 1C.')

            print(f'[RUN] ell={ell:g}, N={ncoarse}: unclosed coarse PDE...')
            a_no=run_rollout(times,a_ref[0],None,times,DT,NU)
            print(f'[RUN] ell={ell:g}, N={ncoarse}: oracle-stress coarse PDE...')
            a_oracle=run_rollout(times,a_ref[0],tau_time_series,times,DT,NU)
            h=1.0/ncoarse
            err_no=np.array([relative_mass_l2(v,r,h) for v,r in zip(a_no,a_ref)])
            err_oracle=np.array([relative_mass_l2(v,r,h) for v,r in zip(a_oracle,a_ref)])
            # Weak projected forcing from the exact stress snapshot.
            s_norm=np.array([np.linalg.norm(stress_rhs_on_coarse(tau,ncoarse))
                              for tau in tau_time_series])
            max_mass_err_no=float(np.max(np.abs(np.mean(a_no,axis=1)-np.mean(a_no[0]))))
            max_mass_err_oracle=float(np.max(np.abs(np.mean(a_oracle,axis=1)-np.mean(a_oracle[0]))))
            for it,t in enumerate(times):
                diag.append({
                    'time':t,'ell':ell,'N_coarse':ncoarse,
                    'rel_L2_no_closure':err_no[it],
                    'rel_L2_oracle_stress':err_oracle[it],
                    'weak_stress_force_norm':s_norm[it],
                })
            summary.append({
                'ell':ell,'N_coarse':ncoarse,
                'final_rel_L2_no_closure':err_no[-1],
                'final_rel_L2_oracle_stress':err_oracle[-1],
                'oracle_error_ratio':err_oracle[-1]/max(err_no[-1],1e-14),
                'max_mass_error_no_closure':max_mass_err_no,
                'max_mass_error_oracle':max_mass_err_oracle,
            })
            results[(float(ell),ncoarse)]={
                'ref':a_ref,'unclosed':a_no,'oracle':a_oracle,
                'err_no':err_no,'err_oracle':err_oracle,
                'stress_force_norm':s_norm,
            }
    pd.DataFrame(diag).to_csv(OUT/'coarse_rollout_time_diagnostics.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'coarse_rollout_summary.csv',index=False)
    make_plots(times,results)
    plot_workflow()
    print('\n=== Final relative FEM L2 errors ===')
    print(pd.DataFrame(summary)[['ell','N_coarse','final_rel_L2_no_closure',
                                 'final_rel_L2_oracle_stress','oracle_error_ratio']].to_string(index=False))
    print('\nResults:',OUT)
    print('[NOTE] If the oracle-stress error is nonzero, FEM projection and time')
    print('       discretization also contribute. This is expected.')
    print('[NEXT] Build a multi-trajectory dataset before training SMTNO.')

if __name__ == '__main__':
    main()
