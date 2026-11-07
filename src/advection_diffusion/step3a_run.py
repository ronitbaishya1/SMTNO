from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def locate_freefem(preferred):
    candidates = [
        preferred,
        shutil.which("FreeFem++-nw"),
        shutil.which("FreeFem++"),
        "/Applications/FreeFem++.app/Contents/MacOS/FreeFem++-nw",
        "/Applications/FreeFem++.app/Contents/MacOS/FreeFem++",
        "/opt/homebrew/bin/FreeFem++",
        "/usr/local/bin/FreeFem++",
    ]
    for entry in candidates:
        if entry and (Path(entry).is_file() or shutil.which(str(entry))):
            return str(entry)
    raise SystemExit(
        "FreeFEM++ not found. Supply --freefem /full/path/to/FreeFem++. "
        "For the Mac app, examine /Applications/FreeFem++.app/Contents/MacOS/"
    )


def main():
    ap = argparse.ArgumentParser(description="SMTNO Step 3A: published 2D CDR FEM")
    ap.add_argument("--case", type=int, choices=[3, 5], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--method", choices=["galerkin", "supg"], default="supg")
    ap.add_argument("--freefem")
    ap.add_argument("--outroot", type=Path, default=HERE / "results" / "step3A")
    a = ap.parse_args()
    if a.n < 4 or a.dt <= 0 or a.eps <= 0 or a.save_every < 1:
        ap.error("n >= 4, dt > 0, eps > 0, save-every >= 1 are required")
    Tf = a.T if a.T is not None else (1.0 if a.case == 3 else 0.25)
    if Tf <= 0:
        ap.error("T must be > 0")
    src = HERE / "step3a_cdr2d.edp"
    if not src.exists():
        raise SystemExit(f"Missing {src}; keep .edp and .py in the same folder")
    out = a.outroot.expanduser().resolve() / (
        f"example{a.case}_{a.method}_n{a.n}_dt{a.dt:g}_eps{a.eps:g}_T{Tf:g}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ff = locate_freefem(a.freefem)
    cmd = [ff, "-nw", str(src),
           "-case", str(a.case), "-n", str(a.n),
           "-dt", str(a.dt), "-T", str(Tf), "-eps", str(a.eps),
           "-save", str(a.save_every),
           "-supg", "1" if a.method == "supg" else "0",
           "-out", str(out)]
    cfg = {
        "citation": "Cengizci, Ugur, Natesan, CPC 329 (2026) 110375, Examples 3/5",
        "case": a.case, "n": a.n, "dt_requested": a.dt, "T": Tf,
        "epsilon": a.eps, "method": a.method, "save_every": a.save_every,
        "method_description": "P1, backward Euler, standard Galerkin or SUPG; not full paper SUPG-YZbeta",
        "matches_paper_grid_time_epsilon": (a.n == 64 and abs(a.dt-0.001)<1e-12 and abs(a.eps-1e-8)<1e-16 and abs(Tf-(1.0 if a.case == 3 else 0.25))<1e-12),
        "command": cmd
    }
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print("Executing FreeFEM++", flush=True)
    print("Output directory:", out, flush=True)
    with (out / "run.log").open("w") as log:
        result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        lines = (out / "run.log").read_text(errors="replace").splitlines()
        print("\n".join(lines[-35:]))
        raise SystemExit(f"FreeFEM++ failed (exit {result.returncode}). Full log: {out / 'run.log'}")
    for fname in ("fields.csv", "metrics.csv"):
        if not (out / fname).exists():
            raise SystemExit(f"Missing {fname}; inspect {out / 'run.log'}")
    print("Finished successfully. Use step3a_plot.py to plot saved outputs.")


if __name__ == "__main__":
    main()