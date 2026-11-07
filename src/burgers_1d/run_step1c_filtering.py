#!/usr/bin/env python3
"""SMTNO Stage 1C — periodic Gaussian filtering and coarse FEM projections.

Location on user's Mac:
    SMTNO/FEM/burgers_1d/run_step1c_filtering.py
Run from anywhere:
    python /path/to/SMTNO/FEM/burgers_1d/run_step1c_filtering.py

Workflow:
1. FreeFEM++ generates a fine periodic Burgers trajectory at many physical times.
2. Apply *physical* Gaussian filters of several widths ell on the same fine mesh.
3. Compute the exact filtered nonlinear (subfilter) stress from those snapshots.
4. Project each filtered state onto periodic piecewise-linear coarse FEM meshes
   using a quadrature-assembled L2 projection; filter width ell and mesh h
   remain independent variables.
5. Save data, diagnostics, plots and a conceptual workflow figure.

Requirements: FreeFem++ in PATH, numpy, pandas, matplotlib.
No neural network training is performed here.
"""

from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Locations and numerical parameters ----------
PROJECT = Path(__file__).resolve().parents[2]
BASENAME = "burgers_step1c_fine"
FEM_DIR = PROJECT / "FEM" / "burgers_1d"
RAW_DIR = PROJECT / "datasets" / "raw"
PROCESSED_DIR = PROJECT / "datasets" / "processed" / "burgers_1d"
OUTPUT_DIR = PROJECT / "results" / "burgers_1d" / "step1C"
for folder in (FEM_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

N_FINE = 512           # 512 periodic fine-grid samples (513 including x=1)
DT = 0.000125
T_FINAL = 0.20
SAVE_EVERY = 20        # save a snapshot every 0.0025 physical-time units
NU = 0.02
ELL_VALUES = (0.01, 0.02, 0.04)  # physical Gaussian widths, NOT mesh sizes
COARSE_N_VALUES = (64, 128)

FINE_CSV = RAW_DIR / (BASENAME + ".csv")
EDP = FEM_DIR / (BASENAME + ".edp")
LOG = OUTPUT_DIR / "freefem_run.log"


def freefem_executable() -> str:
    exe = shutil.which("FreeFem++")
    if exe:
        return exe
    mac_app = Path("/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++")
    if mac_app.is_file():
        return str(mac_app)
    raise FileNotFoundError("FreeFem++ not found. Run `command -v FreeFem++` in Terminal.")


def generate_fine_trajectory() -> None:
    steps = round(T_FINAL / DT)
    assert np.isclose(steps * DT, T_FINAL)
    assert steps % SAVE_EVERY == 0
    edp = textwrap.dedent(f'''
        verbosity = 0;
        int N = {N_FINE};
        real dt = {DT:.17g};
        real nu = {NU:.17g};
        int steps = {steps};
        int saveEvery = {SAVE_EVERY};

        meshL Th = segment(N);
        fespace Vh(Th, P1, periodic=[[1],[2]]);
        Vh u, uold, v;
        u = sin(2*pi*x) + 0.25*sin(4*pi*x);

        problem Burgers(u,v,solver=UMFPACK)
          = int1d(Th)(u*v/dt + nu*dx(u)*dx(v))
          - int1d(Th)(uold*v/dt + 0.5*uold*uold*dx(v));

        ofstream fout("{FINE_CSV.as_posix()}");
        fout.precision(17);
        fout << "time,x,u" << endl;
        real mass0 = int1d(Th)(u);
        for (int n=0; n<=steps; n++) {{
            if (n % saveEvery == 0) {{
                real t = n*dt;
                for (int j=0; j<=N; j++) {{
                    real xx = 1.0*j/N;
                    fout << t << "," << xx << "," << u(xx) << endl;
                }}
            }}
            if (n < steps) {{
                uold = u;
                Burgers;
            }}
        }}
        cout << "Step1C completed. mass error = "
             << int1d(Th)(u)-mass0 << endl;
    ''')
    EDP.write_text(edp)
    proc = subprocess.run([freefem_executable(), "-nw", str(EDP)],
                          capture_output=True, text=True)
    LOG.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0 or not FINE_CSV.exists():
        raise RuntimeError(f"FreeFEM++ run failed. Inspect {LOG}")
    print(f"[OK] Generated FEM trajectory: {FINE_CSV}")


def load_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(FINE_CSV)
    if not {"time", "x", "u"}.issubset(df.columns):
        raise ValueError("Unexpected FEM CSV columns")
    groups = list(df.groupby("time", sort=True))
    times = np.array([float(t) for t, _ in groups])
    states = []
    for time, grp in groups:
        grp = grp.sort_values("x")
        xp = grp.x.to_numpy(dtype=float)
        up = grp.u.to_numpy(dtype=float)
        if len(xp) != N_FINE+1 or not np.allclose(xp, np.linspace(0,1,N_FINE+1)):
            raise ValueError(f"Bad x-grid at t={time}")
        if abs(up[0]-up[-1]) > 1e-10:
            raise ValueError(f"Periodic boundary mismatch at t={time}")
        states.append(up[:-1])   # omit duplicate x=1 from FFT
    return times, np.arange(N_FINE)/N_FINE, np.asarray(states)


def filter_gaussian(u: np.ndarray, ell: float) -> np.ndarray:
    # Periodic physical Gaussian filter: Ghat(k) = exp(-ell^2*k^2/2)
    n = len(u)
    k = 2*np.pi*np.fft.rfftfreq(n, d=1/n)
    return np.fft.irfft(np.fft.rfft(u)*np.exp(-0.5*(ell*k)**2), n=n)


def l2_project_filtered_fem(u: np.ndarray, ncoarse: int) -> tuple[np.ndarray, np.ndarray]:
    """Return P1 periodic FEM coefficients and reconstructed field on fine grid.

    Uses numerical quadrature for b_i = integral u(x) phi_i(x) dx.
    Uniform periodic mass matrix has entries Mii=2h/3 and Mi,i+/-1=h/6;
    it can be inverted by FFT without an external sparse linear solver.
    """
    nfine = len(u)
    if nfine % ncoarse != 0:
        raise ValueError("For this starter code, fine N must divide coarse N")
    xf = np.arange(nfine)/nfine
    h = 1/ncoarse
    xc = np.arange(ncoarse)/ncoarse
    dist = np.abs(xf[None, :] - xc[:, None])
    dist = np.minimum(dist, 1-dist)
    hats = np.clip(1 - dist/h, 0, 1)
    rhs = (hats @ u) / nfine
    wave = 2*np.pi*np.arange(ncoarse)/ncoarse
    mass_eigs = h*(2/3 + np.cos(wave)/3)
    a = np.fft.ifft(np.fft.fft(rhs)/mass_eigs).real
    # periodic continuous P1 reconstruction on fine samples
    u_proj = np.interp(xf, np.r_[xc, 1.0], np.r_[a, a[0]])
    return a, u_proj


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-15))


def save_stage_diagram():
    stages = [
        "Step 1A: solve nonlinear Burgers PDE with FreeFEM++",
        "Step 1B: verify mesh and time convergence",
        "Step 1C.1: generate dense, fine FEM time series",
        "Step 1C.2: filter at physical lengths ell = 0.01, 0.02, 0.04",
        "Step 1C.3: measure subfilter stress tau_ell",
        "Step 1C.4: project filtered states onto coarser FEM spaces",
        "Next: compare coarse PDE evolution and learn memory closure"
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("SMTNO: numerical foundations and physical coarse-graining", fontsize=15, pad=18)
    ys = np.linspace(0.90, 0.12, len(stages))
    for i, (y, msg) in enumerate(zip(ys, stages)):
        ax.text(.5, y, msg, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"))
        if i < len(stages)-1:
            ax.annotate("", xy=(.5, ys[i+1]+.043), xytext=(.5, y-.043),
                        arrowprops=dict(arrowstyle="->", lw=1.6))
    fig.savefig(OUTPUT_DIR / "step1A_1B_1C_workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(x, u_final, filtered_final, projections):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, u_final, label="Fine FEM u(x,T)", linewidth=2)
    for ell, uf in filtered_final.items():
        ax.plot(x, uf, label=f"Filtered ell={ell:g}", linewidth=1.7)
    ax.set(xlabel="x", ylabel="u", title="Step 1C: same fine grid, different physical filter widths")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filtered_profiles.png", dpi=180); plt.close(fig)

    ellmid = 0.02
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.plot(x, filtered_final[ellmid], label=f"Filtered reference ell={ellmid}", linewidth=2.3)
    for nc, proj in projections[ellmid].items():
        ax.plot(x, proj, label=f"Periodic P1 FEM L2 projection N={nc}", linewidth=1.4)
    ax.set(xlabel="x", ylabel="u", title="Step 1C: physical filtering vs numerical mesh projection")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coarse_FEM_projection.png", dpi=180); plt.close(fig)


def plot_closure(x, tau_final, times, rms_tau):
    fig, ax = plt.subplots(figsize=(9,5.5))
    for ell, tau in tau_final.items():
        ax.plot(x, tau, label=f"ell={ell:g}")
    ax.set(xlabel="x", ylabel=r"$\tau_\ell(x,T)$",
           title="Step 1C: exact subfilter nonlinear stress")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subfilter_stress_profiles.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5.5))
    for ell, values in rms_tau.items():
        ax.plot(times, values, label=f"ell={ell:g}")
    ax.set(xlabel="time", ylabel=r"RMS $\tau_\ell$",
           title="Step 1C: unresolved stress changes over physical time")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subfilter_stress_evolution.png", dpi=180); plt.close(fig)


def main():
    print("[Step 1C] Generate high-resolution FEM trajectory")
    generate_fine_trajectory()
    times, x, states = load_trajectory()
    nt, nf = states.shape
    if nt < 3:
        raise RuntimeError("Need multiple time snapshots")
    print(f"[OK] {nt} time snapshots, {nf} independent periodic spatial nodes")

    filtered = np.empty((nt, len(ELL_VALUES), nf), dtype=float)
    stress = np.empty_like(filtered)
    diagnostic_rows = []
    projection_rows = []
    final_filters = {}
    final_stress = {}
    final_projections = {}
    rms_tau = {ell: [] for ell in ELL_VALUES}

    for it, (t, u) in enumerate(zip(times, states)):
        for ie, ell in enumerate(ELL_VALUES):
            uf = filter_gaussian(u, ell)
            stress_ell = 0.5*(filter_gaussian(u*u, ell) - uf*uf)
            filtered[it,ie] = uf
            stress[it,ie] = stress_ell
            rms_tau[ell].append(float(np.sqrt(np.mean(stress_ell**2))))
            mass_difference = float(np.mean(uf)-np.mean(u))
            diagnostic_rows.append({
                "time":t, "ell":ell, "mass_filter_difference":mass_difference,
                "rms_tau":rms_tau[ell][-1],
                "mean_tau":float(np.mean(stress_ell)),
                "energy_filtered":float(0.5*np.mean(uf**2)),
                "energy_fine":float(0.5*np.mean(u**2)),
            })
            if it==nt-1:
                final_filters[ell] = uf
                final_stress[ell] = stress_ell
                final_projections[ell] = {}

            for nc in COARSE_N_VALUES:
                coeffs, projected = l2_project_filtered_fem(uf, nc)
                projection_rows.append({
                    "time":t, "ell":ell, "N_coarse":nc,
                    "h_coarse":1/nc,
                    "rel_l2_project_error":rel_l2(projected, uf),
                    "mass_project_difference":float(np.mean(projected)-np.mean(uf)),
                })
                if it == nt-1:
                    final_projections[ell][nc] = projected
                # Save FEM coefficients for future model development; one file per ell, nc.
                # Collect time series in memory via dictionary keyed by (ell,nc).
                if it == 0:
                    coeff_series[(ell,nc)] = []
                coeff_series[(ell,nc)].append(coeffs)

    np.savez_compressed(PROCESSED_DIR / "fine_filtered_and_stress.npz",
                        times=times, x=x, fine_states=states,
                        ell=np.array(ELL_VALUES), filtered_states=filtered,
                        subfilter_stress=stress)
    for (ell,nc), coefflist in coeff_series.items():
        np.savez_compressed(PROCESSED_DIR / f"coarse_FEM_ell{ell:g}_N{nc}.npz",
                            times=times, x_coarse=np.arange(nc)/nc,
                            coeffs=np.asarray(coefflist), ell=ell, N_coarse=nc)
    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT_DIR/"filter_diagnostics.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(OUTPUT_DIR/"projection_diagnostics.csv", index=False)

    plot_profiles(x, states[-1], final_filters, final_projections)
    plot_closure(x, final_stress, times, rms_tau)
    save_stage_diagram()

    mdiff=max(abs(r["mass_filter_difference"]) for r in diagnostic_rows)
    psummary=pd.DataFrame(projection_rows)
    print(f"[OK] Filtering preserves mass to max absolute error {mdiff:.2e}")
    print("[INFO] Final-time coarse FEM L2 projection errors:")
    print(psummary[np.isclose(psummary.time, times[-1])][
        ["ell","N_coarse","rel_l2_project_error","mass_project_difference"]
    ].to_string(index=False))
    print("[OK] Data saved:", PROCESSED_DIR)
    print("[OK] Plots/diagrams/diagnostics saved:", OUTPUT_DIR)


if __name__ == "__main__":
    coeff_series = {}
    main()
