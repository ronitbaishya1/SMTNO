#!/usr/bin/env python3
"""SMTNO / Stage 1F — diverse Burgers FEM trajectories for closure learning.

Run on the Mac (FreeFem++ must be installed):
    cd "$HOME/Desktop/IIT Guwahati/SMTNO"
    python FEM/burgers_1d/run_step1f_ensemble.py

First try a smaller test:
    python FEM/burgers_1d/run_step1f_ensemble.py --quick

Outputs:
    datasets/raw/burgers_1d/step1F/             case_###.csv/.edp/.log
    datasets/processed/burgers_1d/step1F/       ensemble.npz, manifest.csv,
                                                diagnostics.csv, metadata.json
    results/burgers_1d/step1F/                 PNG plots and workflow diagram

No model training: this is ground-truth data generation and quality control.
The dataset is split BY TRAJECTORY, preventing time-snapshot leakage.

On a 1-periodic domain, filtering uses the physical Gaussian kernel
P_ell u = FFT^-1 [ FFT(u) * exp(-0.5*(2*pi*k*ell)**2) ].
The nonlinear subfilter flux tau=0.5*(P_ell(u^2)-(P_ell u)^2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / 'datasets' / 'raw' / 'burgers_1d' / 'step1F'
PROC = PROJECT / 'datasets' / 'processed' / 'burgers_1d' / 'step1F'
OUT = PROJECT / 'results' / 'burgers_1d' / 'step1F'

N_FINE = 512
COARSE_N = (64, 128)
ELLS = np.array([0.01, 0.02, 0.04], dtype=float)
NU = 0.02
DT = 0.000125
T_FINAL = 0.20
SAVE_EVERY = 20
NSTEPS = round(T_FINAL / DT)
NSNAP = NSTEPS // SAVE_EVERY + 1
assert np.isclose(NSTEPS * DT, T_FINAL)
assert NSTEPS % SAVE_EVERY == 0


def find_freefem() -> str:
    p = shutil.which('FreeFem++')
    if p:
        return p
    fallback = Path('/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++')
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError('FreeFem++ not found. Verify the macOS installation and PATH.')


def sampled_ic(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate diverse mean-zero, smooth periodic initial conditions.

    Fourier amplitudes taper with k but contain appreciable k=3..6 content.
    """
    ks = np.arange(1, 7, dtype=float)
    amps = rng.uniform(0.20, 1.00, size=len(ks)) / np.sqrt(ks)
    amps[0] = rng.uniform(0.60, 1.15)
    phases = rng.uniform(-np.pi, np.pi, size=len(ks))
    # Make the magnitude comparable between trajectories while keeping shapes diverse.
    tmp_x = np.arange(512) / 512
    u = sum(float(a) * np.sin(2*np.pi*int(k)*tmp_x + float(p))
            for k, a, p in zip(ks, amps, phases))
    peak = float(np.max(np.abs(u)))
    desired_peak = rng.uniform(0.9, 1.3)
    amps = amps * (desired_peak / peak)
    return amps, phases


def edp_source(case_id: int, amps: np.ndarray, phases: np.ndarray, csv_file: Path) -> str:
    terms = []
    for k, (a, ph) in enumerate(zip(amps, phases), start=1):
        terms.append(f'({float(a):.17g})*sin(2*pi*{k}*x + ({float(ph):.17g}))')
    ic = ' + '.join(terms)
    # Formula follows the validated 1C FreeFEM++ implementation.
    return f'''// SMTNO Step 1F case {case_id:03d}. Generated automatically; do not hand-edit.
verbosity = 0;
int N = {N_FINE};
real nu = {NU:.17g};
real dt = {DT:.17g};
int steps = {NSTEPS};
int saveEvery = {SAVE_EVERY};
meshL Th = segment(N);
fespace Vh(Th, P1, periodic=[[1],[2]]);
Vh u, uold, v;
u = {ic};
problem Burgers(u, v, solver=UMFPACK)
    = int1d(Th)(u*v/dt + nu*dx(u)*dx(v))
    - int1d(Th)(uold*v/dt + 0.5*uold*uold*dx(v));
ofstream fout("{csv_file.as_posix()}");
fout.precision(17);
fout << "time,x,u" << endl;
real mass0 = int1d(Th)(u);
for (int n = 0; n <= steps; n++) {{
    if (n % saveEvery == 0) {{
        real t = n*dt;
        for (int j = 0; j <= N; j++) {{
            real xx = 1.0*j/N;
            fout << t << "," << xx << "," << u(xx) << endl;
        }}
    }}
    if (n < steps) {{
        uold = u;
        Burgers;
    }}
}}
cout << "Case={case_id:03d} mass drift=" << int1d(Th)(u)-mass0 << endl;
'''


def run_fem(case_id: int, amps: np.ndarray, phases: np.ndarray, freefem: str,
            reuse: bool = True) -> tuple[np.ndarray, np.ndarray]:
    case = f'case_{case_id:03d}'
    edp_path = RAW / f'{case}.edp'
    csv_path = RAW / f'{case}.csv'
    log_path = RAW / f'{case}.log'
    source = edp_source(case_id, amps, phases, csv_path)
    if reuse and csv_path.exists() and edp_path.exists() and edp_path.read_text() == source:
        print(f'  [CACHE] {case}: using previously saved FEM trajectory')
    else:
        edp_path.write_text(source)
        p = subprocess.run([freefem, '-nw', str(edp_path)], capture_output=True, text=True)
        log_path.write_text((p.stdout or '') + '\n' + (p.stderr or ''))
        if p.returncode != 0 or not csv_path.exists():
            raise RuntimeError(f'{case} failed: inspect {log_path}')
        print(f'  [FEM] {case}: completed')
    df = pd.read_csv(csv_path)
    expected_rows = NSNAP * (N_FINE + 1)
    if len(df) != expected_rows:
        raise ValueError(f'{case}: expected {expected_rows} CSV rows, got {len(df)}')
    vals = df['u'].to_numpy(float).reshape(NSNAP, N_FINE + 1)
    t = df['time'].to_numpy(float).reshape(NSNAP, N_FINE + 1)
    x = df['x'].to_numpy(float).reshape(NSNAP, N_FINE + 1)
    if not np.allclose(t, t[:, :1], atol=1e-12):
        raise ValueError(f'{case}: inconsistent times within a snapshot')
    if not np.allclose(x, np.linspace(0, 1, N_FINE+1)[None, :], atol=1e-12):
        raise ValueError(f'{case}: unexpected periodic mesh')
    if not np.allclose(t[:, 0], np.arange(NSNAP) * DT * SAVE_EVERY, atol=1e-12):
        raise ValueError(f'{case}: expected {NSNAP} evenly spaced snapshots')
    if np.max(np.abs(vals[:, 0]-vals[:, -1])) > 1e-9:
        raise ValueError(f'{case}: periodic boundary mismatch')
    u = vals[:, :-1]
    if not np.all(np.isfinite(u)):
        raise ValueError(f'{case}: nonfinite FEM results')
    return t[:, 0], u


def gaussian_filter(states: np.ndarray, ell: float) -> np.ndarray:
    """FFT along final spatial axis; periodic physical filtering, not mesh coarsening."""
    n = states.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1/n)
    multiplier = np.exp(-.5*(ell*k)**2)
    return np.fft.irfft(np.fft.rfft(states, axis=-1)*multiplier, n=n, axis=-1)


def projector(nfine: int, ncoarse: int):
    """Mass-projection of fine nodal field onto periodic coarse P1 space.

    RHS uses periodic composite trapezoid quadrature, as in Step 1C;
    mass matrix inversion is exact for the uniform periodic P1 mesh.
    """
    assert nfine % ncoarse == 0
    x = np.arange(nfine)/nfine
    xc = np.arange(ncoarse)/ncoarse
    h = 1/ncoarse
    dist = np.abs(x[:, None] - xc[None, :])
    hats = np.maximum(1 - np.minimum(dist, 1-dist)/h, 0.0)
    mass_eigs = h * (2/3 + np.cos(2*np.pi*np.arange(ncoarse)/ncoarse)/3)
    return hats, mass_eigs


def fem_l2_projection(fields: np.ndarray, hats: np.ndarray, mass_eigs: np.ndarray) -> np.ndarray:
    """fields shape (snapshots, nfine), return coefficients (snapshots, ncoarse)."""
    nfine = fields.shape[-1]
    b = fields @ hats / nfine
    return np.fft.ifft(np.fft.fft(b, axis=-1)/mass_eigs[None, :], axis=-1).real


def weak_stress_rhs(stresses: np.ndarray, ncoarse: int) -> np.ndarray:
    """Construct S_i = integral tau * phi_i' dx on Ncoarse nodes. 

    This reproduces the discretization used in Step 1E for training labels.
    """
    nfine = stresses.shape[-1]
    nsteps = nfine//ncoarse
    if nsteps*ncoarse != nfine:
        raise ValueError('N_FINE must be divisible by coarse mesh size')
    begins = np.arange(ncoarse)[:, None]*nsteps
    offsets = np.arange(nsteps+1)[None, :]
    sample = stresses[:, (begins + offsets)%nfine]
    edge_means = (.5*sample[:, :, 0] + sample[:, :, 1:-1].sum(axis=-1)
                  + .5*sample[:, :, -1])/nsteps
    return np.roll(edge_means, 1, axis=-1)-edge_means


def plot_workflow() -> None:
    stages = [
        '1A–1B  |  Solve and validate 1D Burgers FEM',
        '1C  |  Filter fine solution and calculate subfilter stress',
        '1D  |  Verify eigenmodes and Gaussian-scale relations',
        '1E  |  Verify coarse rollout with / without oracle closure',
        '1F  |  Generate many independent fine FEM trajectories',
        '1F  |  Filter and project each trajectory to coarse FEM spaces',
        '1F  |  Save closure labels and trajectory-separated data splits',
        'Next  |  Train and compare memoryless vs. memory-based closures'
    ]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.set_title('SMTNO workflow: numerical validation to multi-trajectory learning data',fontsize=13,pad=18)
    ys = np.linspace(9.25,0.85,len(stages))
    for i,(y,label) in enumerate(zip(ys,stages)):
        box = FancyBboxPatch((1.15,y-.37), 7.7,.74,
            boxstyle='round,pad=.09',facecolor='white',edgecolor='black',lw=1.3)
        ax.add_patch(box)
        ax.text(5,y,label,ha='center',va='center',fontsize=10)
        if i<len(stages)-1:
            ax.add_patch(FancyArrowPatch((5,y-.47),(5,ys[i+1]+.47),
                         arrowstyle='-|>',lw=1.3,mutation_scale=13))
    fig.tight_layout()
    fig.savefig(OUT/'step1A_to_step1F_workflow.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot_outputs(fine: np.ndarray, tau: np.ndarray, times: np.ndarray,
                 all_params: pd.DataFrame, split_manifest: pd.DataFrame,
                 diag: pd.DataFrame) -> None:
    # fine shape [cases, times, space]
    xx = np.arange(N_FINE)/N_FINE
    fig,ax=plt.subplots(figsize=(9,5))
    for i in range(min(8,len(fine))):
        ax.plot(xx,fine[i,0],label=f'case {i:03d}',lw=1.4)
    ax.set(xlabel='x',ylabel='u(x,0)',title='Step 1F: varied initial conditions (first 8 trajectories)')
    ax.legend(ncol=2,fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'ensemble_initial_conditions.png',dpi=180);plt.close(fig)

    # Sample trajectory: final-state filter comparison, same fine grid
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(xx,fine[0,-1],label='Fine reference (case 000)',lw=2)
    for ie,ell in enumerate(ELLS):
        ax.plot(xx, gaussian_filter(fine[0,-1],float(ell)),label=f'ell={ell:.2f}')
    ax.set(xlabel='x',ylabel='u',title='Step 1F: physical filtering for one representative case')
    ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'sample_filtered_profiles.png',dpi=180);plt.close(fig)

    # Ensemble average of unresolved stress RMS vs time
    fig,ax=plt.subplots(figsize=(9,5))
    for j,ell in enumerate(ELLS):
        rms=np.sqrt(np.mean(tau[:,j]**2,axis=-1)) # shape cases x time
        mean=rms.mean(axis=0)
        q10=np.quantile(rms,.1,axis=0);q90=np.quantile(rms,.9,axis=0)
        line=ax.plot(times,mean,label=f'ell={ell:g}')[0]
        ax.fill_between(times,q10,q90,alpha=.15,color=line.get_color())
    ax.set(xlabel='Physical time',ylabel='RMS subfilter stress',
           title='Step 1F: unresolved stress across independent trajectories')
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'ensemble_stress_vs_time.png',dpi=180);plt.close(fig)

    # Histogram for dataset splits
    sizes=split_manifest.groupby('split',observed=True)['case_id'].count().reindex(['train','val','test']).fillna(0)
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    ax.bar(sizes.index,sizes.values)
    ax.set(xlabel='Dataset split',ylabel='Number of independent FEM trajectories',
           title='Step 1F: train / validation / test are trajectory-disjoint')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'trajectory_split_counts.png',dpi=180);plt.close(fig)

    # Global mass drift across trajectories
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.hist(diag['fine_mass_drift_max'].to_numpy(),bins=min(12,len(diag)))
    ax.set(xlabel='Max absolute fine mass drift',ylabel='Trajectory count',
           title='Step 1F: FEM mass-conservation diagnostics')
    fig.tight_layout();fig.savefig(OUT/'mass_conservation_distribution.png',dpi=180);plt.close(fig)
    plot_workflow()


def parse_args():
    p=argparse.ArgumentParser(description='Generate diverse SMTNO Burgers FEM trajectories.')
    p.add_argument('--ntraj',type=int,default=30,help='Number of independent trajectories; default 30')
    p.add_argument('--seed',type=int,default=2026,help='Reproducible initial-condition seed')
    p.add_argument('--quick',action='store_true',help='Run 6 trajectories to test the pipeline')
    p.add_argument('--force',action='store_true',help='Recompute FreeFEM output even if identical cached EDP exists')
    return p.parse_args()


def main():
    args=parse_args()
    ntraj = 6 if args.quick else args.ntraj
    if ntraj < 3:
        raise ValueError('At least 3 trajectories needed for train/val/test separation.')
    for d in (RAW,PROC,OUT):d.mkdir(parents=True,exist_ok=True)
    freefem=find_freefem()
    rng=np.random.default_rng(args.seed)

    all_ic=[]
    for i in range(ntraj):
        amps,phases=sampled_ic(rng)
        all_ic.append((amps,phases))

    # Split based on entire trajectories, never individual time snapshots.
    perm = np.random.default_rng(args.seed+7).permutation(ntraj)
    ntest=max(1,int(round(ntraj/6)))
    nval=max(1,int(round(ntraj/6)))
    if ntraj-ntest-nval<1: ntest=nval=1
    split=np.full(ntraj,'train',dtype='<U5')
    split[perm[-ntest:]]='test'
    split[perm[-(ntest+nval):-ntest]]='val'

    fine=np.empty((ntraj,NSNAP,N_FINE),dtype=np.float32)
    filtered=np.empty((ntraj,len(ELLS),NSNAP,N_FINE),dtype=np.float32)
    tau=np.empty_like(filtered)
    coarse={nc:np.empty((ntraj,len(ELLS),NSNAP,nc),dtype=np.float32) for nc in COARSE_N}
    forcing={nc:np.empty((ntraj,len(ELLS),NSNAP,nc),dtype=np.float32) for nc in COARSE_N}
    projectors={nc:projector(N_FINE,nc) for nc in COARSE_N}
    times_ref=None
    diagnostics=[]
    manifests=[]
    print(f'[BEGIN] {ntraj} distinct FEM trajectories; NU={NU}, N_FINE={N_FINE}, snapshots={NSNAP}')
    for i,(amps,phases) in enumerate(all_ic):
        times,states=run_fem(i,amps,phases,freefem,reuse=not args.force)
        if times_ref is None: times_ref=times
        if not np.allclose(times,times_ref):raise ValueError('Mismatched time grids between trajectories')
        fine[i]=states
        drift=float(np.max(np.abs(states.mean(axis=1)-states[0].mean())))
        energy=0.5*np.mean(states*states,axis=1)
        diagnostics.append({'case_id':i,'split':split[i],'fine_mass_drift_max':drift,
            'energy_initial':energy[0],'energy_final':energy[-1],
            'energy_nonmonotone_count':int(np.sum(np.diff(energy)>1e-8))})
        row={'case_id':i,'split':split[i],'seed':args.seed}
        for k,(a,p) in enumerate(zip(amps,phases),1):
            row[f'amp{k}']=float(a);row[f'phase{k}']=float(p)
        manifests.append(row)
        for j,ell in enumerate(ELLS):
            bar=gaussian_filter(states,float(ell))
            bar2=gaussian_filter(states*states,float(ell))
            s=.5*(bar2-bar*bar)
            filtered[i,j]=bar
            tau[i,j]=s
            for nc in COARSE_N:
                hats,mass_eigs=projectors[nc]
                coarse[nc][i,j]=fem_l2_projection(bar,hats,mass_eigs)
                forcing[nc][i,j]=weak_stress_rhs(s,nc)

    if not np.all(np.isfinite(tau)) or not np.all(np.isfinite(fine)):
        raise RuntimeError('Non-finite fields detected')
    max_mass=float(max(x['fine_mass_drift_max'] for x in diagnostics))
    print(f'[CHECK] maximum numerical mass drift across fine trajectories: {max_mass:.3e}')
    if max_mass>1e-6:
        raise RuntimeError('Fine trajectories failed mass conservation check; do not train on them')
    # Confirm mass conservation under filtering and FEM projection for all cases/times.
    for nc in COARSE_N:
        orig=filtered.mean(axis=-1)
        proj=coarse[nc].mean(axis=-1)
        err=float(np.max(np.abs(orig-proj)))
        print(f'[CHECK] max mass difference after L2 projection to N={nc}: {err:.3e}')
        if err>2e-5: raise RuntimeError(f'Mass projection check failed for N={nc}')

    manifest=pd.DataFrame(manifests)
    diag=pd.DataFrame(diagnostics)
    manifest.to_csv(PROC/'trajectory_manifest.csv',index=False)
    diag.to_csv(PROC/'trajectory_diagnostics.csv',index=False)
    metadata={'step':'1F','description':'30 independent periodic Burgers FEM trajectories, or custom ntraj',
      'ntraj':ntraj,'seed':args.seed,'nu':NU,'t_final':T_FINAL,'dt':DT,
      'fine_nodes_periodic':N_FINE,'snapshots':NSNAP,'ell':ELLS.tolist(),
      'coarse_nodes_periodic':list(COARSE_N),'split':'trajectory-disjoint',
      'filter':'periodic Gaussian exp(-0.5*(2*pi*k*ell)^2)',
      'tau':'0.5*(P(u^2) - P(u)^2)',
      'spatial_projection':'periodic P1 FEM mass projection using fine uniform-node quadrature',
      'precision':'arrays float32, validation computed float64'}
    (PROC/'metadata.json').write_text(json.dumps(metadata,indent=2))
    x=np.arange(N_FINE)/N_FINE
    dest=PROC/('burgers_step1F_quick.npz' if args.quick else 'burgers_step1F_ensemble.npz')
    np.savez_compressed(dest,
      times=np.asarray(times_ref,dtype=np.float64),x=np.asarray(x),ell=ELLS,
      split=split,case_ids=np.arange(ntraj),fine=fine,filtered=filtered,tau=tau,
      coarse_N64=coarse[64],coarse_N128=coarse[128],
      S_tau_N64=forcing[64],S_tau_N128=forcing[128])
    plot_outputs(fine,tau,np.asarray(times_ref),manifest,manifest,diag)
    print(f'[DONE] Saved dataset: {dest}')
    print(f'[DONE] Diagnostic CSV and metadata: {PROC}')
    print(f'[DONE] Plots and workflow figure: {OUT}')
    print('       Next: examine quality control and train memoryless / memory-based baselines.')

if __name__=='__main__':
    main()
