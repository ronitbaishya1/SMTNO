#!/usr/bin/env python3
"""SMTNO Step 2C: a controlled physical-scale switching benchmark.

Uses existing Step 2A FNO checkpoints and Step 1F FEM dataset.
This is NOT yet a learned scale-memory transition architecture.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from step2b_closed_loop_rollout import (
    clark_weak, load_checkpoint, net_weak, periodic_step, relative_mass_l2
)

METHODS = [
    "no_closure", "clark_gradient", "memoryless",
    "history_transferred", "history_reset", "oracle_held"
]


def gaussian_transfer(u: np.ndarray, ell_from: float, ell_to: float) -> np.ndarray:
    """Transfer a periodic physical field to a wider Gaussian filter."""
    if ell_to < ell_from:
        raise ValueError("This pilot only supports coarsening: ell_to >= ell_from")
    n = u.shape[-1]
    k = np.fft.rfftfreq(n, d=1.0 / n)
    kernel = np.exp(-0.5 * (2.0 * np.pi * k) ** 2 * (ell_to**2 - ell_from**2))
    return np.fft.irfft(np.fft.rfft(u, axis=-1) * kernel, n=n, axis=-1)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def workflow_plot(output_dir: Path) -> None:
    labels = [
        "Fine FEM ensemble + trained Step 2A FNO checkpoints",
        "Autonomous coarse FEM evolution at physical filter ell=0.01",
        "At t=0.10, Gaussian-transfer resolved state to ell=0.04",
        "History: transfer past states OR reset past states",
        "Continue autonomous rollout at the new physical scale",
        "Compare reference, mass, transition error, and final rollout errors",
        "Next: learn a joint resolved-state + latent-memory scale transition"
    ]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.set_title("SMTNO Step 2C — physical-scale switching benchmark", pad=12)
    ys = np.linspace(9.2, 0.8, len(labels))
    for i, (y, label) in enumerate(zip(ys, labels)):
        ax.add_patch(FancyBboxPatch((0.65, y - .42), 8.7, .84,
                                    boxstyle="round,pad=.06", edgecolor="black",
                                    facecolor="white", linewidth=1.1))
        ax.text(5, y, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(5, ys[i+1] + .49), xytext=(5, y - .49),
                        arrowprops=dict(arrowstyle="->", linewidth=1.1))
    fig.savefig(output_dir / "step2C_workflow.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path,
                        default=Path("/content/drive/MyDrive/SMTNO/Colab Notebooks"))
    parser.add_argument("--switch-time", type=float, default=0.10)
    parser.add_argument("--ell-from", type=float, default=0.01)
    parser.add_argument("--ell-to", type=float, default=0.04)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=None,
                        help="Optional short run for debugging")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.substeps < 1 or args.nu <= 0 or args.max_cases < 1:
        raise ValueError("Invalid substeps, viscosity, or case count")
    if not args.ell_to > args.ell_from:
        raise ValueError("For this benchmark ell-to must be greater than ell-from")

    project = args.project_dir
    out = project / "results" / "step2C"
    out.mkdir(parents=True, exist_ok=True)
    data_path = project / "datasets" / "burgers_step1F_ensemble.npz"
    with np.load(data_path, allow_pickle=False) as npz:
        d = {name: npz[name] for name in (
            "ell", "times", "split", "case_ids", "coarse_N64", "coarse_N128",
            "S_tau_N64", "S_tau_N128"
        )}

    ells = np.asarray(d["ell"], float)
    times_full = np.asarray(d["times"], float)
    t_count = len(times_full) if args.max_snapshots is None else args.max_snapshots
    if t_count is None or t_count < 6 or t_count > len(times_full):
        raise ValueError("max-snapshots must leave at least six saved snapshots")
    times = times_full[:t_count]
    if not np.allclose(np.diff(times), np.diff(times)[0]):
        raise ValueError("Expected uniformly spaced snapshot times")
    switch_idx = int(np.argmin(abs(times - args.switch_time)))
    if not np.isclose(times[switch_idx], args.switch_time, atol=1e-9):
        raise ValueError(f"Switch time must match saved time; closest is {times[switch_idx]}")
    if not 4 <= switch_idx <= len(times)-2:
        raise ValueError("Switch time must follow warm-up and precede final time")
    i_from = int(np.argmin(abs(ells - args.ell_from)))
    i_to = int(np.argmin(abs(ells - args.ell_to)))
    if abs(ells[i_from] - args.ell_from) > 1e-8 or abs(ells[i_to] - args.ell_to) > 1e-8:
        raise ValueError("Requested physical filter widths are missing from dataset")
    held = np.flatnonzero(d["split"].astype(str) == "test")[:args.max_cases]
    if not len(held):
        raise ValueError("Test split is empty")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint_dir = project / "checkpoints" / "step2A"
    memless, xstd_0, ystd_0, h0 = load_checkpoint(
        "memoryless", checkpoint_dir, device, 1, ells
    )
    history_model, xstd_1, ystd_1, h1 = load_checkpoint(
        "short_history", checkpoint_dir, device, 4, ells
    )
    if h0 != 1 or h1 != 4:
        raise ValueError("Expected Step 2A models with 1 and 4 input snapshots")
    if not np.isclose(xstd_0, xstd_1) or not np.allclose(ystd_0, ystd_1):
        raise ValueError("Checkpoint normalizations disagree")
    if not np.isclose(max(ells), .04):
        raise ValueError("Step 2A scale normalization assumes max ell=0.04")

    first = h1 - 1
    dt = float(times[1] - times[0]) / args.substeps
    summary = []
    per_time = []
    transfer_diagnostics = []
    colors = {"no_closure": "tab:gray", "clark_gradient": "tab:orange",
              "memoryless": "tab:blue", "history_transferred": "tab:green",
              "history_reset": "tab:red", "oracle_held": "tab:purple"}

    for N in (64, 128):
        coarse = np.asarray(d[f"coarse_N{N}"], dtype=np.float64)[held]
        forcing = np.asarray(d[f"S_tau_N{N}"], dtype=np.float64)[held]
        source = coarse[:, i_from]
        target = coarse[:, i_to]
        ref = np.concatenate((source[:, first:switch_idx], target[:, switch_idx:len(times)]), axis=1)
        bsz = len(held)

        check_transfer = gaussian_transfer(source[:, switch_idx], args.ell_from, args.ell_to)
        transfer_l2 = relative_mass_l2(check_transfer, target[:, switch_idx])
        for case_idx, e in zip(held, transfer_l2):
            transfer_diagnostics.append({"case_id": int(d["case_ids"][case_idx]),
                                         "N": N, "switch_time": float(times[switch_idx]),
                                         "project_then_filter_vs_filter_then_project": float(e)})

        states = {name: source[:, first].copy() for name in METHODS}
        histories = {
            name: np.stack([source[:, first-k] for k in range(h1)], axis=1).copy()
            for name in METHODS
        }
        paths = {name: [states[name].copy()] for name in METHODS}

        for it in range(first, len(times)-1):
            if it == switch_idx:
                for name in METHODS:
                    states[name] = gaussian_transfer(states[name], args.ell_from, args.ell_to)
                    if name == "history_reset":
                        histories[name] = np.repeat(states[name][:, None, :], h1, axis=1)
                    else:
                        histories[name] = gaussian_transfer(
                            histories[name], args.ell_from, args.ell_to
                        )
                    histories[name][:, 0, :] = states[name]
                    paths[name][-1] = states[name].copy()

            ell_idx = i_from if it < switch_idx else i_to
            ell_current = float(ells[ell_idx])
            for name in METHODS:
                u = states[name]
                if name == "no_closure":
                    weak = np.zeros_like(u)
                elif name == "clark_gradient":
                    weak = clark_weak(u, ell_current)
                elif name == "oracle_held":
                    weak = forcing[:, ell_idx, it, :].copy()
                elif name == "memoryless":
                    weak = net_weak(memless, histories[name], ell_current,
                                    xstd_0, ystd_0[ell_idx], device, 1)
                else:
                    weak = net_weak(history_model, histories[name], ell_current,
                                    xstd_1, ystd_1[ell_idx], device, 4)
                weak = np.asarray(weak, dtype=np.float64)
                weak -= np.mean(weak, axis=-1, keepdims=True)
                for _ in range(args.substeps):
                    u = periodic_step(u, weak, dt, args.nu)
                if not np.all(np.isfinite(u)):
                    raise FloatingPointError(f"Nonfinite rollout for {name}, N={N}")
                states[name] = u
                histories[name] = np.concatenate((u[:, None, :], histories[name][:, :-1]), axis=1)
                paths[name].append(u.copy())

        mean_curves = {}
        for name in METHODS:
            pred = np.stack(paths[name], axis=1)
            errors = relative_mass_l2(pred, ref)
            mean_curves[name] = errors.mean(axis=0)
            for ci, case_idx in enumerate(held):
                old_mass = float(np.mean(source[ci, first]))
                drift = float(np.max(np.abs(np.mean(pred[ci], axis=-1) - old_mass)))
                summary.append({
                    "case_id": int(d["case_ids"][case_idx]), "N": N, "model": name,
                    "ell_from": args.ell_from, "ell_to": args.ell_to,
                    "final_rel_M_l2": float(errors[ci, -1]),
                    "post_switch_mean_rel_M_l2": float(np.mean(errors[ci, switch_idx-first:])),
                    "switch_rel_M_l2": float(errors[ci, switch_idx-first]),
                    "max_abs_mass_drift": drift
                })
                for j, t in enumerate(times[first:]):
                    per_time.append({"case_id": int(d["case_ids"][case_idx]),
                                     "N": N, "model": name, "time": float(t),
                                     "ell": float(args.ell_from if j + first < switch_idx else args.ell_to),
                                     "rel_M_l2": float(errors[ci, j])})
            print(f"N={N}, {name:20s}, mean final error={np.mean(errors[:, -1]):.6g}", flush=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        for name in METHODS:
            ax.plot(times[first:], mean_curves[name], label=name, color=colors[name])
        ax.axvline(float(times[switch_idx]), color="black", linestyle="--", alpha=.7)
        ax.set(title=f"Step 2C: physical scale switch N={N} ({args.ell_from} to {args.ell_to})",
               xlabel="Physical time", ylabel="Mean relative FEM mass-norm error")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"switch_rollout_error_N{N}.png", dpi=180)
        plt.close(fig)

    write_csv(out / "scale_switch_by_case.csv", summary)
    write_csv(out / "scale_switch_by_time.csv", per_time)
    write_csv(out / "scale_switch_transfer_defect.csv", transfer_diagnostics)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, N in zip(axes, (64, 128)):
        names = METHODS
        mean_final = [np.mean([r["final_rel_M_l2"] for r in summary
                              if r["N"] == N and r["model"] == m]) for m in names]
        ax.bar(np.arange(len(names)), mean_final, color=[colors[m] for m in names])
        ax.set_xticks(np.arange(len(names)), names, rotation=38, ha="right")
        ax.set_title(f"N={N}")
        ax.set_ylabel("Mean final relative FEM error")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Step 2C: errors after the physical filtering scale changes")
    fig.tight_layout()
    fig.savefig(out / "scale_switch_final_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N in (64, 128):
        yy = [r["project_then_filter_vs_filter_then_project"]
              for r in transfer_diagnostics if r["N"] == N]
        ax.scatter([N] * len(yy), yy, label=f"N={N}")
    ax.set(yscale="log", xlabel="FEM elements N",
           ylabel="Relative transfer defect at switch",
           title="Known Gaussian scale transfer: numerical projection defect")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "scale_transfer_projection_defect.png", dpi=180)
    plt.close(fig)

    workflow_plot(out)
    (out / "run_config.json").write_text(json.dumps({
        "device": str(device), "test_case_ids": [int(d["case_ids"][q]) for q in held],
        "ell_from": args.ell_from, "ell_to": args.ell_to,
        "switch_time": float(times[switch_idx]), "substeps": args.substeps,
        "note": "No learned scale-memory transfer yet. Gaussian coarsening is known; oracle uses future fine forcing."
    }, indent=2))
    print("\n[COMPLETE] Results:", out)


if __name__ == "__main__":
    main()