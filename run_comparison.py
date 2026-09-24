"""
Main entry point -- does Reachy-2 Orbita beat the SRS arm?

Runs both project metrics and prints / plots the comparison:

  Metric 1  reachable workspace size    (workspace.py)  -> larger workspace?
  Metric 2  chest-workspace singularity (singularity.py) -> fewer singularities?

Outputs (in outputs/):
  workspace_compare.png       reachable clouds + normalised volume bar
  singularity_compare.png     chest grid coloured by worst-case sigma_min
  singularity_interactive.html rotatable 3-D viewer

Usage
-----
    python run_comparison.py
    python run_comparison.py --ws-samples 3000000 --no-show
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from robots import make_orbita, make_srs
from singularity import (CHEST_ROI, SIGMA_THRESHOLD, chest_grid,
                         compare_singularity, evaluate_grid)
from workspace import compare_workspace, measure_workspace

OUT_DIR = "outputs"


def _roi_box_edges():
    r = CHEST_ROI
    xs, ys, zs = [r["xmin"], r["xmax"]], [r["ymin"], r["ymax"]], [r["zmin"], r["zmax"]]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [(corners[[a, b], 0], corners[[a, b], 1], corners[[a, b], 2])
            for a, b in edges]


def _equal_3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_zlabel("z up")


def _plot_workspace(ws_results, path):
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Metric 1 — reachable workspace (dense joint-space sampling)",
                 fontsize=13)
    rng = np.random.default_rng(0)
    allpts = np.concatenate([r.tip for r in ws_results])

    for col, r in enumerate(ws_results):
        idx = rng.choice(len(r.tip), size=min(9000, len(r.tip)), replace=False)
        tip = r.tip[idx]
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.scatter(tip[:, 0], tip[:, 1], tip[:, 2], s=2, alpha=0.25,
                   color="steelblue" if col == 0 else "darkorange")
        ax.set_title(f"{r.name}\nV/L³ = {r.volume_norm:.3f}")
        _equal_3d(ax, allpts)

    ax = fig.add_subplot(1, 3, 3)
    names = [r.name.split()[0] for r in ws_results]
    vals = [r.volume_norm for r in ws_results]
    bars = ax.bar(names, vals, color=["steelblue", "darkorange"])
    ax.set_ylabel("normalised reachable volume  V / L³")
    ax.set_title("larger = reaches more space")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    print(f"saved: {path}")


def _plot_singularity(sg_results, threshold, path):
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(15, 6))
    mode_txt = ("task approach-cone" if sg_results[0].mode == "task"
                else "full-sphere")
    fig.suptitle("Metric 2 — chest-workspace worst-case singularity "
                 f"({mode_txt} orientations, σ_min < {threshold:.3f})",
                 fontsize=13)
    allpts = chest_grid()
    vmax = max(np.nanpercentile(r.worst_smin, 95) for r in sg_results)

    for col, r in enumerate(sg_results):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        p = ax.scatter(r.points[:, 0], r.points[:, 1], r.points[:, 2],
                       c=r.worst_smin, s=30, cmap="viridis_r",
                       vmin=0, vmax=vmax)
        near = r.near_mask
        ax.scatter(r.points[near, 0], r.points[near, 1], r.points[near, 2],
                   c="red", s=55, marker="x")
        for ex, ey, ez in _roi_box_edges():
            ax.plot(ex, ey, ez, color="0.5", lw=0.7)
        ax.set_title(f"{r.name}\nnear-singular positions: "
                     f"{100 * r.near_fraction:.1f}%")
        _equal_3d(ax, allpts)
        fig.colorbar(p, ax=ax, shrink=0.6, pad=0.02, label="worst-case σ_min")

    ax = fig.add_subplot(1, 3, 3)
    for r in sg_results:
        ax.hist(r.worst_smin, bins=40, histtype="step", label=r.name.split()[0])
    ax.axvline(threshold, color="red", ls="--", lw=1, label="threshold")
    ax.set_xlabel("worst-case σ_min per position")
    ax.set_ylabel("grid positions")
    ax.set_title("further right = safer from singularity")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    print(f"saved: {path}")


def _plot_interactive(sg_results, threshold, path):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed -> skipping interactive HTML")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = make_subplots(rows=1, cols=len(sg_results),
                        specs=[[{"type": "scene"}] * len(sg_results)],
                        subplot_titles=[r.name for r in sg_results])
    for col, r in enumerate(sg_results, start=1):
        near = r.near_mask
        fig.add_trace(go.Scatter3d(
            x=r.points[~near, 0], y=r.points[~near, 1], z=r.points[~near, 2],
            mode="markers", name="ok",
            marker=dict(size=4, color=r.worst_smin[~near], colorscale="Viridis",
                        cmin=0, colorbar=dict(title="σmin", x=1.0)
                        if col == len(sg_results) else None),
            showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=r.points[near, 0], y=r.points[near, 1], z=r.points[near, 2],
            mode="markers", name="near-singular",
            marker=dict(size=6, color="red", symbol="x"),
            showlegend=(col == 1),
        ), row=1, col=col)
        for ex, ey, ez in _roi_box_edges():
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                          line=dict(color="black", width=2), showlegend=False),
                          row=1, col=col)
    fig.update_layout(
        title="Chest-workspace worst-case singularity — drag to rotate",
        height=650)
    for s in ("scene", "scene2"):
        fig.update_layout({s: dict(aspectmode="data")})
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-samples", type=int, default=1500000,
                    help="joint-space samples per arm for the workspace volume")
    ap.add_argument("--n-orient", type=int, default=12,
                    help="max-span orientations per grid position")
    ap.add_argument("--threshold", type=float, default=SIGMA_THRESHOLD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    robots = [make_srs(), make_orbita()]

    # ---- Metric 1: workspace size ----
    ws_results = [measure_workspace(r, args.ws_samples, rng) for r in robots]
    print(compare_workspace(ws_results))
    print()

    # ---- Metric 2: chest singularity ----
    # (a) fair everyday benchmark: tool only needs to point into the work area
    sg_task = [evaluate_grid(r, rng, args.threshold, args.n_orient, mode="task")
               for r in robots]
    print(compare_singularity(sg_task))
    print()
    # (b) harsh stress test: orientations spread over the whole sphere
    sg_sphere = [evaluate_grid(r, rng, args.threshold, args.n_orient, mode="sphere")
                 for r in robots]
    print(compare_singularity(sg_sphere))

    os.makedirs(OUT_DIR, exist_ok=True)
    _plot_workspace(ws_results, os.path.join(OUT_DIR, "workspace_compare.png"))
    _plot_singularity(sg_task, args.threshold,
                      os.path.join(OUT_DIR, "singularity_compare.png"))
    _plot_interactive(sg_task, args.threshold,
                      os.path.join(OUT_DIR, "singularity_interactive.html"))

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
