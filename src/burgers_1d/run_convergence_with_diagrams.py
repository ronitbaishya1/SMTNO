from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import shutil

# -----------------------------
# Project paths
# -----------------------------
SCRIPT_PATH = Path(__file__).resolve()
# Assumes script is saved to SMTNO/FEM/burgers_1d/run_convergence_with_diagrams.py
PROJECT_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) >= 3 else SCRIPT_PATH.parent
FEM_DIR = PROJECT_ROOT / "FEM" / "burgers_1d"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "burgers_1d"
CONV_DIR = RESULTS_DIR / "convergence"
STEP1A_DIR = RESULTS_DIR / "step1A"
RUNS_DIR = CONV_DIR / "runs"

for p in [RESULTS_DIR, CONV_DIR, STEP1A_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FREEFEM = shutil.which("FreeFem++") or "FreeFem++"

# -----------------------------
# Small plotting helpers
# -----------------------------
def save_simple_workflow(path: Path, title: str, boxes: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=20)

    n = len(boxes)
    xs = np.linspace(0.08, 0.92, n)
    y = 0.5
    for i, (x, text) in enumerate(zip(xs, boxes)):
        ax.text(
            x, y, text,
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black")
        )
        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i+1]-0.07, y), xytext=(x+0.07, y),
                arrowprops=dict(arrowstyle="->", lw=1.8)
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def relative_l2(x: np.ndarray, u: np.ndarray, uref: np.ndarray) -> float:
    num = np.trapz((u - uref) ** 2, x)
    den = np.trapz((uref) ** 2, x)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


# -----------------------------
# Step 1A validation figures
# -----------------------------
def make_step1a_figures() -> None:
    csv_path = RAW_DIR / "burgers_1d.csv"
    if not csv_path.exists():
        print(f"[INFO] Step 1A CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if not {"time", "x", "u"}.issubset(df.columns):
        print("[INFO] Step 1A CSV does not have expected columns.")
        return

    times = sorted(df["time"].unique())

    # Solution evolution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in times:
        sub = df[df["time"] == t].sort_values("x")
        ax.plot(sub["x"], sub["u"], label=f"t={t:.2f}")
    ax.set_title("Step 1A: Burgers FEM solution evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(STEP1A_DIR / "step1A_solution_evolution.png", dpi=200)
    plt.close(fig)

    # Energy plot
    energy_rows = []
    for t in times:
        sub = df[df["time"] == t].sort_values("x")
        x = sub["x"].to_numpy()
        u = sub["u"].to_numpy()
        e = 0.5 * np.trapz(u**2, x)
        m = np.trapz(u, x)
        energy_rows.append((t, e, m))
    energy_df = pd.DataFrame(energy_rows, columns=["time", "energy", "mass"])
    energy_df.to_csv(STEP1A_DIR / "step1A_energy_mass.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(energy_df["time"], energy_df["energy"], marker="o")
    ax.set_title("Step 1A: Burgers energy decay")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$E(t)=\frac{1}{2}\int_0^1 u^2\,dx$")
    fig.tight_layout()
    fig.savefig(STEP1A_DIR / "step1A_energy_decay.png", dpi=200)
    plt.close(fig)

    # Step 1A workflow
    boxes = [
        "Install FreeFEM++",
        "Solve 1D viscous Burgers equation",
        "Save FEM trajectory to CSV",
        "Check mass conservation",
        "Check energy decay"
    ]
    save_simple_workflow(STEP1A_DIR / "step1A_workflow.png", "Step 1A workflow", boxes)
    print(f"[OK] Step 1A plots saved to: {STEP1A_DIR}")


# -----------------------------
# FreeFEM run machinery
# -----------------------------
def build_edp_text(N: int, dt: float, T: float, nu: float, out_csv: str) -> str:
    return textwrap.dedent(f"""
    verbosity = 0;

    int N = {N};
    real nu = {nu};
    real dt = {dt};
    real T = {T};
    int steps = T/dt;

    meshL Th = segment(N);
    fespace Vh(Th, P1, periodic=[[1],[2]]);
    Vh u, uold, v;

    u = sin(2*pi*x) + 0.25*sin(4*pi*x);

    problem Burgers(u, v, solver=UMFPACK)
        = int1d(Th)(
              u*v/dt
            + nu*dx(u)*dx(v)
          )
        - int1d(Th)(
              uold*v/dt
            + 0.5*uold*uold*dx(v)
          );

    ofstream fout("{out_csv}");
    fout.precision(17);
    fout << "time,x,u" << endl;

    real mass0 = int1d(Th)(u);

    for (int n = 0; n < steps; n++)
    {{
        uold = u;
        Burgers;
    }}

    for (int j = 0; j <= N; j++)
    {{
        real xp = 1.0*j/N;
        fout << T << "," << xp << "," << u(xp) << endl;
    }}

    cout << "Final mass error = " << int1d(Th)(u)-mass0 << endl;
    cout << "Done: N=" << N << " dt=" << dt << endl;
    """)


def run_case(case_name: str, N: int, dt: float, T: float = 0.2, nu: float = 0.02) -> dict:
    case_dir = RUNS_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    edp_path = case_dir / f"{case_name}.edp"
    out_csv = case_dir / f"{case_name}.csv"
    log_path = case_dir / f"{case_name}.log"

    edp_text = build_edp_text(N=N, dt=dt, T=T, nu=nu, out_csv=out_csv.as_posix())
    edp_path.write_text(edp_text)

    cmd = [FREEFEM, "-nw", str(edp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""))

    if result.returncode != 0:
        raise RuntimeError(f"FreeFEM failed for {case_name}. See {log_path}")

    # Parse mass error if printed
    mass_error = np.nan
    for line in (result.stdout or "").splitlines():
        if "Final mass error" in line:
            try:
                mass_error = float(line.split("=")[-1].strip())
            except Exception:
                pass

    df = pd.read_csv(out_csv)
    df = df.sort_values("x")
    x = df["x"].to_numpy()
    u = df["u"].to_numpy()
    energy = 0.5 * np.trapz(u**2, x)

    return {
        "case": case_name,
        "N": N,
        "dt": dt,
        "T": T,
        "nu": nu,
        "mass_error": mass_error,
        "energy": energy,
        "csv": out_csv,
        "x": x,
        "u": u,
    }


# -----------------------------
# Convergence study
# -----------------------------
def make_comparison_plots(spatial_results, temporal_results):
    # Mesh profile plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for res in spatial_results:
        ax.plot(res["x"], res["u"], label=f'N={res["N"]}, dt={res["dt"]}')
    ax.set_title("Step 1B: final-time profiles for spatial study")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,T)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONV_DIR / "mesh_profiles.png", dpi=200)
    plt.close(fig)

    # Time profile plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for res in temporal_results:
        ax.plot(res["x"], res["u"], label=f'dt={res["dt"]}')
    ax.set_title("Step 1B: final-time profiles for temporal study")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,T)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CONV_DIR / "time_profiles.png", dpi=200)
    plt.close(fig)


def make_convergence_plots(summary_df: pd.DataFrame):
    spatial = summary_df[summary_df["study"] == "spatial"].sort_values("h")
    temporal = summary_df[summary_df["study"] == "temporal"].sort_values("dt")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(spatial["h"], spatial["rel_l2_error"], marker="o")
    ax.set_title("Step 1B: spatial convergence")
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("relative L2 error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONV_DIR / "spatial_convergence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(temporal["dt"], temporal["rel_l2_error"], marker="o")
    ax.set_title("Step 1B: temporal convergence")
    ax.set_xlabel("time step dt")
    ax.set_ylabel("relative L2 error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CONV_DIR / "temporal_convergence.png", dpi=200)
    plt.close(fig)


def make_stage1_workflow():
    boxes = [
        "Step 1A\nRun Burgers FEM baseline",
        "Plot Step 1A solution\nand energy decay",
        "Step 1B\nRun mesh-refinement study",
        "Run time-step\nrefinement study",
        "Estimate errors and\ncreate convergence plots",
        "Move to filtering\nand eigenvalue step"
    ]
    save_simple_workflow(CONV_DIR / "stage1A_stage1B_workflow.png", "Stage 1 workflow (Step 1A + Step 1B)", boxes)


def main():
    print("[1/4] Generating Step 1A validation plots and diagram (if baseline CSV exists)...")
    make_step1a_figures()

    print("[2/4] Running convergence cases with FreeFEM++...")
    # Spatial study: vary N, keep dt sufficiently small and fixed.
    spatial_specs = [
        ("spatial_N64", 64, 2.5e-4),
        ("spatial_N128", 128, 2.5e-4),
        ("spatial_N256", 256, 2.5e-4),
        ("spatial_ref_N512", 512, 2.5e-4),
    ]
    # Temporal study: vary dt, keep N fixed and fine.
    temporal_specs = [
        ("temporal_dt1e3", 256, 1.0e-3),
        ("temporal_dt5e4", 256, 5.0e-4),
        ("temporal_dt25e4", 256, 2.5e-4),
        ("temporal_ref_dt125e4", 256, 1.25e-4),
    ]

    spatial_results = [run_case(*spec) for spec in spatial_specs]
    temporal_results = [run_case(*spec) for spec in temporal_specs]

    print("[3/4] Computing errors and saving summary table...")
    spatial_ref = next(r for r in spatial_results if r["case"] == "spatial_ref_N512")
    temporal_ref = next(r for r in temporal_results if r["case"] == "temporal_ref_dt125e4")

    rows = []

    for r in spatial_results:
        if r["case"] == spatial_ref["case"]:
            continue
        uref = np.interp(r["x"], spatial_ref["x"], spatial_ref["u"])
        err = relative_l2(r["x"], r["u"], uref)
        rows.append({
            "study": "spatial",
            "case": r["case"],
            "N": r["N"],
            "h": 1.0 / r["N"],
            "dt": r["dt"],
            "rel_l2_error": err,
            "mass_error": r["mass_error"],
            "energy": r["energy"],
            "reference_case": spatial_ref["case"],
        })

    for r in temporal_results:
        if r["case"] == temporal_ref["case"]:
            continue
        uref = np.interp(r["x"], temporal_ref["x"], temporal_ref["u"])
        err = relative_l2(r["x"], r["u"], uref)
        rows.append({
            "study": "temporal",
            "case": r["case"],
            "N": r["N"],
            "h": 1.0 / r["N"],
            "dt": r["dt"],
            "rel_l2_error": err,
            "mass_error": r["mass_error"],
            "energy": r["energy"],
            "reference_case": temporal_ref["case"],
        })

    summary_df = pd.DataFrame(rows)

    # Observed orders (approximate)
    spatial = summary_df[summary_df["study"] == "spatial"].sort_values("h", ascending=False).reset_index(drop=True)
    spatial_orders = [np.nan]
    for i in range(1, len(spatial)):
        e1, e2 = spatial.loc[i-1, "rel_l2_error"], spatial.loc[i, "rel_l2_error"]
        h1, h2 = spatial.loc[i-1, "h"], spatial.loc[i, "h"]
        p = np.log(e1/e2) / np.log(h1/h2)
        spatial_orders.append(p)
    summary_df.loc[spatial.index, "observed_order"] = spatial_orders

    temporal = summary_df[summary_df["study"] == "temporal"].sort_values("dt", ascending=False).reset_index(drop=True)
    temporal_orders = [np.nan]
    for i in range(1, len(temporal)):
        e1, e2 = temporal.loc[i-1, "rel_l2_error"], temporal.loc[i, "rel_l2_error"]
        d1, d2 = temporal.loc[i-1, "dt"], temporal.loc[i, "dt"]
        p = np.log(e1/e2) / np.log(d1/d2)
        temporal_orders.append(p)
    temporal_idx = summary_df[summary_df["study"] == "temporal"].sort_values("dt", ascending=False).index
    summary_df.loc[temporal_idx, "observed_order"] = temporal_orders

    summary_path = CONV_DIR / "convergence_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("[4/4] Creating plots and workflow diagrams...")
    make_comparison_plots([r for r in spatial_results if r["case"] != spatial_ref["case"]],
                          [r for r in temporal_results if r["case"] != temporal_ref["case"]])
    make_convergence_plots(summary_df)
    make_stage1_workflow()

    print("\n[OK] Step 1B complete.")
    print(f"Summary CSV: {summary_path}")
    print(f"Plots/diagrams saved in: {CONV_DIR}")
    print(f"Step 1A plots/diagram saved in: {STEP1A_DIR}")


if __name__ == "__main__":
    main()
