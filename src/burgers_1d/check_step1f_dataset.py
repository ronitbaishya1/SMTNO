#!/usr/bin/env python3
"""SMTNO Step 1G: validate actual Stage 1F tensors before Colab training.

Run on Mac:
    cd "$HOME/Desktop/IIT Guwahati/SMTNO"
    python FEM/burgers_1d/check_step1f_dataset.py

Inputs:
    datasets/processed/burgers_1d/step1F/burgers_step1F_ensemble.npz
    datasets/processed/burgers_1d/step1F/trajectory_manifest.csv
    datasets/processed/burgers_1d/step1F/trajectory_diagnostics.csv

Outputs:
    results/burgers_1d/step1G/data_quality_report.json
    results/burgers_1d/step1G/closure_signal_by_scale.png
    results/burgers_1d/step1G/weak_forcing_by_scale.png
    results/burgers_1d/step1G/step1G_workflow.png

This is a data-integrity check, NOT evidence that history or eigenvalues help learning.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "datasets" / "processed" / "burgers_1d" / "step1F"
OUT = ROOT / "results" / "burgers_1d" / "step1G"
OUT.mkdir(parents=True, exist_ok=True)


def gaussian_filter(fields: np.ndarray, ell: float) -> np.ndarray:
    n = fields.shape[-1]
    k = 2*np.pi*np.fft.rfftfreq(n, d=1/n)
    multiplier = np.exp(-0.5*(ell*k)**2)
    return np.fft.irfft(np.fft.rfft(fields, axis=-1)*multiplier, n=n, axis=-1)


def weak_forcing(tau: np.ndarray, n_coarse: int) -> np.ndarray:
    """Independent periodic fine-grid trapezoid evaluation of S_i=int tau phi_i' dx."""
    n_fine=tau.shape[-1]
    if n_fine % n_coarse:
        raise ValueError("Fine grid must divide coarse grid")
    stride=n_fine//n_coarse
    left=tau[..., ::stride]
    right=np.roll(left, -1, axis=-1)
    if stride > 1:
        mid=tau.reshape(*tau.shape[:-1], n_coarse, stride)[...,1:].sum(axis=-1)
    else:
        mid=np.zeros_like(left)
    edge_integral_mean=(0.5*left + mid + 0.5*right)/stride
    return np.roll(edge_integral_mean, 1, axis=-1)-edge_integral_mean


def diagram(path: Path):
    descriptions=[
        "30 independent initial conditions\n(trajectories, not time snapshots)",
        "512-node fine FEM states\n81 snapshots, t=0 to 0.20",
        "Gaussian filtering\n3 physical widths, distinct from mesh h",
        "Compute subfilter stress tau\nand conservative weak forcing S_tau",
        "Mass-project filtered states\nonto N=64 and N=128 FEM spaces",
        "Check splits, identities, conservation\nthen transfer NPZ to Colab",
        "Next: memoryless closure vs.\nshort-history closure (same data splits)"
    ]
    fig,ax=plt.subplots(figsize=(11,10))
    ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    ax.set_title('SMTNO Step 1G — dataset readiness',fontsize=14,pad=20)
    ys=np.linspace(9.1,0.9,len(descriptions))
    for i,(y,txt) in enumerate(zip(ys,descriptions)):
        ax.add_patch(FancyBboxPatch((1.3,y-0.44),7.4,0.88,boxstyle='round,pad=0.07',facecolor='white',edgecolor='black',linewidth=1.3))
        ax.text(5,y,txt,ha='center',va='center',fontsize=10)
        if i+1<len(ys):
            ax.annotate('',xy=(5,ys[i+1]+0.51),xytext=(5,y-0.51),arrowprops={'arrowstyle':'->','lw':1.5})
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    archive=PROC/'burgers_step1F_ensemble.npz'
    manifest_file=PROC/'trajectory_manifest.csv'
    diagnostic_file=PROC/'trajectory_diagnostics.csv'
    for p in (archive,manifest_file,diagnostic_file):
        if not p.is_file():
            sys.exit(f"[MISSING] {p}\nRun the full Step 1F script first; do not rerun --quick after a full run.")

    with np.load(archive,allow_pickle=False) as d:
        available=set(d.files)
        required=['fine','filtered','tau','times','ell','split','case_ids','coarse_N64','coarse_N128','S_tau_N64','S_tau_N128']
        missing=set(required)-available
        if missing:
            sys.exit(f"[ERROR] Missing required arrays: {sorted(missing)}")
        fine=d['fine'].astype(np.float64)
        filtered=d['filtered'].astype(np.float64)
        tau=d['tau'].astype(np.float64)
        ell=d['ell'].astype(np.float64)
        times=d['times'].astype(np.float64)
        splits=d['split'].astype(str)
        ids=d['case_ids']
        coarse={n:d[f'coarse_N{n}'].astype(np.float64) for n in (64,128)}
        rhs={n:d[f'S_tau_N{n}'].astype(np.float64) for n in (64,128)}
    manifest=pd.read_csv(manifest_file)
    diagnostics=pd.read_csv(diagnostic_file)
    C,T,N=fine.shape
    L=len(ell)
    assert filtered.shape==tau.shape==(C,L,T,N), 'Incorrect filtered/tau tensor shape'
    assert len(times)==T and len(splits)==C and len(ids)==C
    assert np.all(np.isfinite(fine)) and np.all(np.isfinite(filtered)) and np.all(np.isfinite(tau))
    assert np.all(np.diff(times)>0), 'Time snapshots not monotonically increasing'
    assert len(set(ids.tolist()))==C
    assert np.allclose(ell,[0.01,0.02,0.04]),f'Unexpected scale widths: {ell}'
    assert manifest['case_id'].is_unique and diagnostics['case_id'].is_unique
    split_lookup={int(row.case_id):str(row.split) for row in manifest[['case_id','split']].itertuples(index=False)}
    assert all(split_lookup[int(i)]==s for i,s in zip(ids,splits)), 'Manifest/NPZ split inconsistency'
    counts={s:int((splits==s).sum()) for s in ('train','val','test')}
    assert counts == {'train':20,'val':5,'test':5},counts
    print(f'[OK] {C} trajectories, {T} snapshots, fine nodes={N}, ell={ell.tolist()}, splits={counts}')

    # Independent reconstruction of physical filter and exact stress from saved FEM fields.
    filter_error=0.0
    stress_error=0.0
    avg_stress_rms=[]
    for j,scale in enumerate(ell):
        b=gaussian_filter(fine,float(scale))
        true_tau=0.5*(gaussian_filter(fine*fine,float(scale))-b*b)
        filter_error=max(filter_error,float(np.max(np.abs(b-filtered[:,j]))))
        stress_error=max(stress_error,float(np.max(np.abs(true_tau-tau[:,j]))))
        avg_stress_rms.append(float(np.sqrt(np.mean(tau[:,j]**2,axis=-1)).mean()))

    mass_drift=float(np.max(np.abs(fine.mean(axis=-1)-fine.mean(axis=-1)[:,:1])))
    projection_mass_errors={}
    forcing_sum_errors={}
    forcing_label_errors={}
    for n in (64,128):
        if coarse[n].shape!=(C,L,T,n) or rhs[n].shape!=(C,L,T,n):
            raise ValueError(f'Incorrect N={n} tensor shape')
        projection_mass_errors[str(n)]=float(np.max(np.abs(filtered.mean(axis=-1)-coarse[n].mean(axis=-1))))
        forcing_sum_errors[str(n)]=float(np.max(np.abs(rhs[n].sum(axis=-1))))
        forcing_label_errors[str(n)]=float(np.max(np.abs(rhs[n]-weak_forcing(tau,n))))

    # Floating-point saved arrays are float32; discrepancies of ~1e-7 are normal.
    checks={
        'filter_identity': filter_error < 3e-6,
        'stress_identity': stress_error < 3e-6,
        'fine_mass_conservation': mass_drift < 3e-6,
        'projection_mass_conservation': max(projection_mass_errors.values()) < 3e-6,
        'weak_forcing_conservation': max(forcing_sum_errors.values()) < 3e-5,
        'forcing_labels': max(forcing_label_errors.values()) < 3e-6,
    }
    # Plot: stress over time for each scale with 10%-90% ensemble band.
    fig, ax=plt.subplots(figsize=(8,5))
    for j,scale in enumerate(ell):
        val=np.sqrt(np.mean(tau[:,j]**2,axis=-1))
        ln=ax.plot(times,val.mean(axis=0),label=f'ell={scale:g}')[0]
        ax.fill_between(times,np.quantile(val,.1,axis=0),np.quantile(val,.9,axis=0),alpha=.15,color=ln.get_color())
    ax.set(xlabel='time',ylabel='RMS tau',title='Step 1G: closure variation across scales and time')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'closure_signal_by_scale.png',dpi=180);plt.close(fig)

    # Plot: mean weak forcing magnitude across time and scales, at N=64.
    fig,ax=plt.subplots(figsize=(8,5))
    for j,scale in enumerate(ell):
        val=np.sqrt(np.mean(rhs[64][:,j]**2,axis=-1))
        ax.plot(times,val.mean(axis=0),label=f'ell={scale:g}')
    ax.set(xlabel='time',ylabel='RMS weak forcing S_tau (N=64)',title='Step 1G: learnable conservative closure forcing')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'weak_forcing_by_scale.png',dpi=180);plt.close(fig)
    diagram(OUT/'step1G_workflow.png')

    report={
        'data':{'ntraj':C,'snapshots':T,'fine_nodes':N,'ell':ell.tolist(),'time_min':float(times.min()),'time_max':float(times.max()),'splits':counts},
        'max_abs_errors':{
            'filter_identity':filter_error,'stress_identity':stress_error,'fine_mass_drift':mass_drift,
            'projection_mass_by_N':projection_mass_errors,'forcing_sum_by_N':forcing_sum_errors,
            'forcing_label_by_N':forcing_label_errors,
        },
        'mean_tau_rms_by_ell':{f'{float(scale):g}':v for scale,v in zip(ell,avg_stress_rms)},
        'checks':checks,'all_checks_passed':all(checks.values()),
        'note':'S_tau is an ORACLE weak forcing label from fine-scale FEM, not a predicted closure. No memory learning has occurred.'
    }
    path=OUT/'data_quality_report.json'
    path.write_text(json.dumps(report,indent=2))
    print('\nNumerical integrity checks:')
    for k,v in checks.items():print(f'  {k}: {"PASS" if v else "FAIL"}')
    print(f'\nSummary: {path}')
    print(f'Figures: {OUT}')
    if not all(checks.values()):
        raise SystemExit('Validation failed; do not transfer the dataset to Colab yet.')
    print('DATASET READY FOR FIRST BASELINE TRAINING (pending evaluation on unseen trajectories).')

if __name__=='__main__':
    main()
