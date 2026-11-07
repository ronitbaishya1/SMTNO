from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def grid(frame, field, n):
    x = np.sort(frame["x"].unique())
    y = np.sort(frame["y"].unique())
    if x.size != n+1 or y.size != n+1:
        raise ValueError(f"Expected {(n+1)} x {(n+1)} grid, got {x.size} x {y.size}")
    z = frame.pivot(index="y", columns="x", values=field)
    z = z.reindex(index=y, columns=x)
    if z.isna().any().any():
        raise ValueError(f"Missing/duplicate sample coordinates for {field}")
    return x, y, z.to_numpy(dtype=float)


def plot_run(folder):
    cfg = json.loads((folder / "run_config.json").read_text())
    fields = pd.read_csv(folder / "fields.csv")
    metrics = pd.read_csv(folder / "metrics.csv")
    case, n = cfg["case"], cfg["n"]
    dest = folder / "figures"
    dest.mkdir(exist_ok=True)
    steps = sorted(fields["step"].unique().tolist())
    snaps, true_snaps, times = [], [], []
    for k in steps:
        subset = fields.loc[fields["step"] == k]
        x,y,u = grid(subset,"u",n)
        _,_,exact = grid(subset,"exact",n)
        snaps.append(u.astype("float32"))
        true_snaps.append(exact.astype("float32"))
        times.append(float(subset["time"].iloc[0]))
    snaps = np.stack(snaps)
    true_snaps = np.stack(true_snaps)
    times = np.asarray(times)
    np.savez_compressed(folder / "field_snapshots.npz",
                        u=snaps, exact=true_snaps if case==3 else np.empty(0),
                        x=x, y=y, t=times, n=n, eps=cfg["epsilon"], case=case)
    kw = dict(origin="lower",extent=[x[0],x[-1],y[0],y[-1]],aspect="equal",cmap="viridis")

    fig,axes = plt.subplots(1,4,figsize=(16,4))
    inds = np.unique(np.linspace(0,len(times)-1,4).astype(int))
    for ax, idx in zip(axes,inds):
        im=ax.imshow(snaps[idx],**kw)
        ax.set(title=f"t={times[idx]:.3g}",xlabel="x",ylabel="y")
    fig.colorbar(im,ax=list(axes),shrink=.72,label="u")
    fig.savefig(dest / "01_time_evolution.png",dpi=170,bbox_inches="tight")
    plt.close(fig)

    if case == 3:
        fig, axes = plt.subplots(1,3,figsize=(14,4))
        for ax,z,title,cmap in zip(axes,
                                  [snaps[-1],true_snaps[-1],np.abs(snaps[-1]-true_snaps[-1])],
                                  ["FEM","Exact","Absolute error"],
                                  ["viridis","viridis","magma"]):
            im=ax.imshow(z,origin="lower",extent=[0,1,0,1],aspect="equal",cmap=cmap)
            fig.colorbar(im,ax=ax,shrink=.76)
            ax.set(title=title,xlabel="x",ylabel="y")
        fig.suptitle(f"Example 3: epsilon={cfg['epsilon']:g}, N={n}")
        fig.tight_layout()
        fig.savefig(dest / "02_exact_vs_fem.png",dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(metrics["time"],metrics["l2_relative"],".-")
        ax.set(title="Exact-solution relative L2 error",xlabel="time",ylabel="relative L2")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(dest / "03_error_vs_time.png",dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(x,snaps[-1][n//2,:],label="FEM")
        ax.plot(x,true_snaps[-1][n//2,:],"--",label="exact at vertices")
        ax.set(title="Midline y=0.5",xlabel="x",ylabel="u")
        ax.legend();ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(dest / "04_midline.png",dpi=170)
        plt.close(fig)
    else:
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(metrics["time"],metrics["min_u"],label="min u")
        ax.plot(metrics["time"],metrics["max_u"],label="max u")
        ax.axhline(0,color="gray",ls="--");ax.axhline(1,color="gray",ls="--")
        ax.set(title="Example 5: boundedness diagnostic",xlabel="time",ylabel="extrema")
        ax.legend();ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(dest / "02_min_max.png",dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(x,np.diag(snaps[-1]),label="final diagonal u(x,x)")
        ax.set(xlabel="x=y",ylabel="u",title="Example 5: diagonal cut")
        ax.grid(alpha=.3);ax.legend()
        fig.tight_layout()
        fig.savefig(dest / "03_diagonal.png",dpi=170)
        plt.close(fig)

    final=metrics.iloc[-1]
    return {"folder":folder.name,"case":case,"method":cfg["method"],"n":n,
            "dt":cfg["dt_requested"],"epsilon":cfg["epsilon"],"T":cfg["T"],
            "final_relative_l2":float(final["l2_relative"]),
            "min_u":float(metrics["min_u"].min()),
            "max_u":float(metrics["max_u"].max()),
            "saved_snapshots":len(times)}


def summary_plot(df,root,case):
    if case != 3:return
    # Plot spatial convergence only where all other numerical inputs match.
    grouped=df.groupby(["method","epsilon","dt","T"])
    for key,g in grouped:
        if g["n"].nunique()<2:continue
        g=g.sort_values("n")
        fig,ax=plt.subplots(figsize=(6,4))
        ax.loglog(g["n"],g["final_relative_l2"],"o-")
        ax.set(xlabel="N intervals per direction",ylabel="final relative L2 error",
               title=f"Spatial study: {key[0]}, eps={key[1]:g}, dt={key[2]:g}")
        ax.grid(alpha=.3,which="both")
        fig.tight_layout()
        fig.savefig(root/f"spatial_{key[0]}_eps{key[1]:g}_dt{key[2]:g}_T{key[3]:g}.png",dpi=170)
        plt.close(fig)
    # Time convergence: fixed N and epsilon, varying dt.
    for key,g in df.groupby(["method","epsilon","n","T"]):
        if g["dt"].nunique()<2:continue
        g=g.sort_values("dt")
        fig,ax=plt.subplots(figsize=(6,4))
        ax.loglog(g["dt"],g["final_relative_l2"],"o-")
        ax.set(xlabel="requested dt",ylabel="final relative L2 error",
               title=f"Time study: {key[0]}, eps={key[1]:g}, N={key[2]}")
        ax.grid(alpha=.3,which="both")
        fig.tight_layout()
        fig.savefig(root/f"time_{key[0]}_eps{key[1]:g}_N{key[2]}_T{key[3]:g}.png",dpi=170)
        plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",type=int,choices=[3,5],required=True)
    ap.add_argument("--outroot",type=Path,default=HERE/"results"/"step3A")
    a=ap.parse_args()
    root=a.outroot.expanduser().resolve()
    folders=sorted(p for p in root.glob(f"example{a.case}_*") if (p/"metrics.csv").exists())
    if not folders:raise SystemExit(f"No completed Example {a.case} runs in {root}")
    rows=[plot_run(folder) for folder in folders]
    df=pd.DataFrame(rows).sort_values(["epsilon","method","n","dt"])
    df.to_csv(root/f"example{a.case}_summary.csv",index=False)
    summary_plot(df,root,a.case)
    print(df.to_string(index=False))
    print("Figures and .npz saved in each run folder.")

if __name__=="__main__":main()