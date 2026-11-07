"""SMTNO Step 2B.1 — paired closed-loop diagnostics.

Run from your Colab Notebooks directory:
 python step2b1_rollout_diagnostics.py

Alternatively:
 python step2b1_rollout_diagnostics.py --input /path/to/step2B --output /path/to/diagnostics

Reads the Step 2B CSV outputs only; does not train or re-run simulations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["no_closure", "clark_gradient", "memoryless", "short_history", "oracle_held"]


def find_input(root: Path) -> Path:
    candidates = [root / "results" / "step2B", root / "step2B", root]
    for p in candidates:
        if all((p / f).is_file() for f in ["rollout_summary.csv", "rollout_summary_by_case.csv", "rollout_errors_by_time.csv"]):
            return p
    raise FileNotFoundError("Could not find Step 2B CSV files. Supply --input pointing to the folder containing the 3 CSV files.")


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def paired_bootstrap_ci(diffs: np.ndarray, n_boot=5000, seed=2026):
    # Small-sample descriptive interval; NOT evidence of generalization across IC families.
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Folder with rollout CSV files")
    ap.add_argument("--output", default=None, help="Destination for diagnostics")
    args = ap.parse_args()
    script_root = Path(__file__).resolve().parent
    root = script_root
    input_dir = find_input(Path(args.input).expanduser() if args.input else root)
    output = Path(args.output).expanduser() if args.output else input_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(input_dir / "rollout_summary.csv")
    by = pd.read_csv(input_dir / "rollout_summary_by_case.csv")
    ts = pd.read_csv(input_dir / "rollout_errors_by_time.csv")
    expected_s = {"model", "mesh_N", "ell", "mean_final_rel_M_l2", "max_abs_mass_drift"}
    expected_by = {"case_id", "model", "N", "ell", "final_rel_M_l2"}
    expected_ts = {"case_id", "model", "N", "ell", "time", "rel_M_l2"}
    for df, need, name in [(s, expected_s, "summary"), (by, expected_by, "by_case"), (ts, expected_ts, "by_time")]:
        if not need.issubset(df.columns):
            raise ValueError(f"{name} is missing: {sorted(need-set(df.columns))}")

    summary_rows = []
    n_values = sorted(by.N.unique())
    ell_values = sorted(by.ell.unique())
    for N in n_values:
        for ell in ell_values:
            part = by[(by.N == N) & np.isclose(by.ell, ell)]
            pivot = part.pivot(index="case_id", columns="model", values="final_rel_M_l2")
            required = {"memoryless", "short_history", "oracle_held", "clark_gradient", "no_closure"}
            if not required.issubset(pivot.columns):
                raise ValueError(f"Missing methods at N={N}, ell={ell}")
            delta = (pivot["memoryless"] - pivot["short_history"]).to_numpy()
            ci_lo, ci_hi = paired_bootstrap_ci(delta)
            m = pivot["memoryless"].mean()
            h = pivot["short_history"].mean()
            oracle = pivot["oracle_held"].mean()
            summary_rows.append({
                "N": N,
                "ell": ell,
                "n_cases": len(delta),
                "memoryless_mean_error": m,
                "history_mean_error": h,
                "absolute_mean_improvement": m-h,
                "relative_improvement_percent": 100*(m-h)/m if m else np.nan,
                "history_wins_cases": int((delta > 0).sum()),
                "bootstrap_abs_improvement_ci_lower": ci_lo,
                "bootstrap_abs_improvement_ci_upper": ci_hi,
                "oracle_mean_error": oracle,
                "best_learned_mean_error": min(m,h),
                "learned_better_than_oracle": min(m,h) < oracle,
            })
    effects = pd.DataFrame(summary_rows)
    effects.to_csv(output / "paired_history_effects.csv", index=False)

    # Figure 1: history benefit by scale, with zero baseline.
    fig, ax = plt.subplots(figsize=(9,5))
    for N in n_values:
        p = effects[effects.N == N].sort_values("ell")
        ax.plot(p.ell, p.relative_improvement_percent, marker="o", label=f"N={N}")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Physical filter width ell")
    ax.set_ylabel("Error reduction from short history (%)")
    ax.set_title("Step 2B.1: Does history help autonomous rollouts?")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig(fig, output / "history_benefit_by_scale.png")

    # Figure 2: individual test-case paired improvements, avoiding hidden mean differences.
    fig, axes = plt.subplots(1, len(n_values), figsize=(12, 5), sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, N in zip(axes, n_values):
        for ell in ell_values:
            part = by[(by.N == N) & np.isclose(by.ell, ell)]
            pivot = part.pivot(index="case_id", columns="model", values="final_rel_M_l2")
            d = (pivot.memoryless-pivot.short_history).values
            ax.scatter(np.full(len(d), ell), d, label=f"ell={ell:g}")
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_title(f"N={N}")
        ax.set_xlabel("Filter width ell")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Memoryless error minus history error")
    fig.suptitle("Per-case history benefit; positive means history wins")
    savefig(fig, output / "history_paired_case_differences.png")

    # Figure 3: oracle comparison / potential numerical floor.
    fig, axes = plt.subplots(1, len(n_values), figsize=(12,4), sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, N in zip(axes, n_values):
        p = s[s.mesh_N == N]
        for method in ["memoryless", "short_history", "oracle_held", "clark_gradient"]:
            q = p[p.model == method].sort_values("ell")
            ax.plot(q.ell, 100*q.mean_final_rel_M_l2, marker="o", label=method)
        ax.set_title(f"N={N}")
        ax.set_xlabel("Filter width ell")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean final relative error (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Step 2B.1: Learned methods vs held-oracle diagnostic")
    savefig(fig, output / "learned_vs_held_oracle.png")

    # Figure 4: final vs time average error at ell=0.04.
    selected_ell = max(ell_values)
    fig, axes = plt.subplots(1, len(n_values), figsize=(12,4), sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax,N in zip(axes,n_values):
        q=s[(s.mesh_N==N)&np.isclose(s.ell, selected_ell)].set_index("model").reindex(MODELS)
        x=np.arange(len(q))
        ax.bar(x-.18,100*q.mean_final_rel_M_l2, .36,label="final")
        ax.bar(x+.18,100*q.mean_time_rel_M_l2, .36,label="time-mean")
        ax.set_xticks(x, ["none","Clark","no history","history","oracle"],rotation=30,ha="right")
        ax.set_title(f"N={N}, ell={selected_ell:g}")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Relative FEM error (%)")
    axes[0].legend()
    fig.suptitle("Final snapshot error versus error averaged across the rollout")
    savefig(fig, output / "time_mean_vs_final.png")

    # Figure 5: mass drift on logarithmic axis.
    fig, ax=plt.subplots(figsize=(10,4.5))
    drift=s.groupby("model")["max_abs_mass_drift"].max().reindex(MODELS)
    ax.bar(np.arange(len(drift)),drift.values)
    ax.set_xticks(np.arange(len(drift)),drift.index,rotation=20,ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Max absolute discrete mass drift")
    ax.set_title("Weak forcing and mass-conservation diagnostic")
    ax.grid(axis="y",alpha=.2)
    savefig(fig,output/"mass_drift_diagnostic.png")

    # Figure 6: auto-generated step diagram.
    stages = [
        "Validated 1F/1G\nfiltered FEM dataset",
        "2A: train memoryless\nand history FNOs",
        "2B: autonomous rollout\nat N=64 and N=128",
        "2B.1: paired tests,\nresidual floor and conservation",
        "Next: improve temporal coupling\nand test true scale switching",
    ]
    fig, ax=plt.subplots(figsize=(12,9))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    ycoords=np.linspace(.89,.13,len(stages))
    for i,(text,y) in enumerate(zip(stages,ycoords)):
        ax.text(.5,y,text,ha="center",va="center",fontsize=12,
                bbox=dict(boxstyle="round,pad=0.8",fc="white",ec="black",lw=1.5))
        if i<len(stages)-1:
            ax.annotate("",xy=(.5,ycoords[i+1]+.065),xytext=(.5,y-.065),
                        arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.set_title("SMTNO: from FEM verification to robust scale–memory evaluation",pad=15)
    savefig(fig,output/"step2B1_workflow.png")

    report={
        "paired_cases_per_condition": int(effects.n_cases.min()),
        "history_wins_all_conditions": bool((effects.history_wins_cases == effects.n_cases).all()),
        "history_improves_all_means": bool((effects.absolute_mean_improvement > 0).all()),
        "learned_beat_held_oracle_somewhere": bool(effects.learned_better_than_oracle.any()),
        "maximum_absolute_mass_drift": float(s.max_abs_mass_drift.max()),
        "cautions": [
            "Five held-out trajectories per condition are too few for conclusive memory claims.",
            "Oracle forcing held over saved intervals is not an exact fully consistent oracle trajectory.",
            "Closure predictions conserve mass to rounding tolerance, not exact arithmetic.",
            "These diagnostics do not validate Mori–Zwanzig memory or a joint scale-transition architecture.",
        ],
    }
    (output/"diagnostic_report.json").write_text(json.dumps(report,indent=2))
    print(effects[["N","ell","relative_improvement_percent","history_wins_cases","learned_better_than_oracle"]].to_string(index=False))
    print("\nSaved analysis to:",output)

if __name__ == "__main__":
    main()