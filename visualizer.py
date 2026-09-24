"""
Interactive dual-arm viewer -- browser edition (no matplotlib, never freezes).

Matplotlib's interactive 3-D is slow and locks up under rapid dragging, so this
viewer PRE-COMPUTES everything in Python and ships a single self-contained HTML
file driven by Plotly.js.  Dragging the sliders in the browser is a pure table
lookup -> instant response, zero freezing, and the file can be shared as-is.

What it does
------------
* Pre-solves best-conditioned 7-DOF position IK for BOTH arms on a dense 3-D
  grid of end-effector targets covering the chest workspace.
* Writes outputs/interactive_viewer.html: two side-by-side 3-D views (SRS and
  Orbita) sharing one target, a half-body for context, three X/Y/Z sliders, and
  a live readout of manipulability w, smallest singular value sigma_min, the
  condition number and a NEAR-SINGULAR flag for each arm.

    python visualizer.py                 # build and open in the browser
    python visualizer.py --no-open       # just build the HTML
    python visualizer.py --step 0.03     # finer grid (bigger file)
"""

from __future__ import annotations

import argparse
import json
import os
import webbrowser

import numpy as np

from robots import (SHOULDER_POS, make_orbita, make_srs, manipulability,
                    singular_values)
from singularity import CHEST_ROI, SIGMA_THRESHOLD

OUT = os.path.join("outputs", "interactive_viewer.html")


# ----------------------------------------------------------------------
# Pre-computation
# ----------------------------------------------------------------------
def _solve_grid(robot, grid, rng, restarts=8):
    """Best-conditioned position IK for every grid target.

    Returns chains (M,5,3): [shoulder mount, shoulder, elbow, wrist, tip],
    and per-target w, sigma_min, cond, reached.
    """
    M = robot.n
    G = grid.shape[0]
    best_q = np.zeros((G, M))
    best_smin = np.full(G, -np.inf)
    fallback_q = np.zeros((G, M))
    fallback_err = np.full(G, np.inf)
    any_reached = np.zeros(G, dtype=bool)

    for _ in range(restarts):
        q0 = robot.sample_joints(G, rng)
        q, reached, err = robot.ik_position_batch(grid, q0)
        _, _, J = robot.fk_jacobian(q)
        smin = singular_values(J)[:, -1]

        take = reached & (smin > best_smin)
        best_q[take] = q[take]
        best_smin[take] = smin[take]
        any_reached |= reached

        fb = err < fallback_err
        fallback_q[fb] = q[fb]
        fallback_err[fb] = err[fb]

    q = np.where(any_reached[:, None], best_q, fallback_q)
    tip, seg_pts, J = robot.fk_jacobian(q)
    sv = singular_values(J)
    smin = sv[:, -1]
    smax = sv[:, 0]
    with np.errstate(divide="ignore"):
        cond = np.where(smin > 0, smax / smin, np.inf)
    w = manipulability(J)

    mount = np.broadcast_to(SHOULDER_POS, (G, 3))
    chains = np.concatenate([mount[:, None, :], seg_pts, tip[:, None, :]], axis=1)
    return chains, w, smin, cond, any_reached


def _half_body_lines():
    """Static torso/neck/head as Plotly line segments (None-separated)."""
    y_out = float(SHOULDER_POS[1])
    y_in = 0.0
    xs = [-0.09, 0.09]
    ys = sorted([y_in, y_out + 0.02])
    z_top, z_bot = 0.05, -0.45
    corners = [[x, y, z] for x in xs for y in ys for z in [z_bot, z_top]]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    X, Y, Z = [], [], []
    for a, b in edges:
        X += [corners[a][0], corners[b][0], None]
        Y += [corners[a][1], corners[b][1], None]
        Z += [corners[a][2], corners[b][2], None]
    # neck
    X += [0.0, 0.0, None]; Y += [y_in, y_in, None]; Z += [z_top, z_top + 0.10, None]
    # head as two orthogonal circles
    t = np.linspace(0, 2 * np.pi, 24)
    hz = z_top + 0.20
    for axis in ("xz", "yz"):
        for k in range(len(t)):
            if axis == "xz":
                X.append(0.09 * np.cos(t[k])); Y.append(y_in); Z.append(hz + 0.09 * np.sin(t[k]))
            else:
                X.append(y_in); Y.append(0.09 * np.cos(t[k])); Z.append(hz + 0.09 * np.sin(t[k]))
        X.append(None); Y.append(None); Z.append(None)
    return X, Y, Z


def _roi_box_lines():
    r = CHEST_ROI
    xs, ys, zs = [r["xmin"], r["xmax"]], [r["ymin"], r["ymax"]], [r["zmin"], r["zmax"]]
    corners = [[x, y, z] for x in xs for y in ys for z in zs]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    X, Y, Z = [], [], []
    for a, b in edges:
        X += [corners[a][0], corners[b][0], None]
        Y += [corners[a][1], corners[b][1], None]
        Z += [corners[a][2], corners[b][2], None]
    return X, Y, Z


def build(step: float, seed: int = 0):
    r = CHEST_ROI
    xs = np.arange(r["xmin"], r["xmax"] + 1e-9, step)
    ys = np.arange(r["ymin"], r["ymax"] + 1e-9, step)
    zs = np.arange(r["zmin"], r["zmax"] + 1e-9, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    nx, ny, nz = len(xs), len(ys), len(zs)
    print(f"grid = {nx}x{ny}x{nz} = {len(grid)} targets/arm")

    rng = np.random.default_rng(seed)
    arms = []
    for mk, color in ((make_srs, "#1f77b4"), (make_orbita, "#ff7f0e")):
        robot = mk()
        chains, w, smin, cond, reached = _solve_grid(robot, grid, rng)
        arms.append(dict(
            name=robot.name, color=color,
            chains=np.round(chains, 4).reshape(len(grid), -1).tolist(),
            w=np.round(w, 5).tolist(),
            smin=np.round(smin, 5).tolist(),
            cond=np.round(np.clip(cond, 0, 9999), 2).tolist(),
            reached=reached.astype(int).tolist(),
        ))

    body_x, body_y, body_z = _half_body_lines()
    roi_x, roi_y, roi_z = _roi_box_lines()
    data = dict(
        nx=nx, ny=ny, nz=nz,
        xs=np.round(xs, 4).tolist(),
        ys=np.round(ys, 4).tolist(),
        zs=np.round(zs, 4).tolist(),
        threshold=SIGMA_THRESHOLD,
        arms=arms,
        body=dict(x=body_x, y=body_y, z=body_z),
        roi=dict(x=roi_x, y=roi_y, z=roi_z),
        init=[nx // 2, ny // 2, nz // 2],
    )

    os.makedirs("outputs", exist_ok=True)
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"saved: {OUT}")
    return os.path.abspath(OUT)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SRS vs Reachy Orbita - chest singularity viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body{font-family:system-ui,Arial,sans-serif;margin:0;padding:12px;background:#fafafa}
  h2{margin:4px 0 8px}
  #plots{display:flex;gap:8px}
  .plot{flex:1;height:60vh;background:#fff;border:1px solid #ddd;border-radius:6px}
  #controls{margin:10px 0;padding:10px;background:#fff;border:1px solid #ddd;border-radius:6px}
  .row{display:flex;align-items:center;gap:10px;margin:6px 0}
  .row label{width:140px;font-weight:600}
  .row input[type=range]{flex:1}
  .val{width:90px;text-align:right;font-family:monospace}
  #readout{display:flex;gap:24px;margin-top:8px;font-family:monospace;font-size:14px}
  .card{padding:8px 12px;border-radius:6px;border:1px solid #ddd;background:#fff;flex:1}
  .near{color:#d62728;font-weight:700}
  .ok{color:#2ca02c}
</style></head>
<body>
<h2>SRS (3-1-3, limited) vs Reachy 2 Orbita (2-2-3, real+cone) &mdash; chest workspace</h2>
<div id="plots">
  <div id="p0" class="plot"></div>
  <div id="p1" class="plot"></div>
</div>
<div id="controls">
  <div class="row"><label>X forward</label><input id="sx" type="range"><span id="vx" class="val"></span></div>
  <div class="row"><label>Y left</label><input id="sy" type="range"><span id="vy" class="val"></span></div>
  <div class="row"><label>Z up</label><input id="sz" type="range"><span id="vz" class="val"></span></div>
  <div id="readout"></div>
</div>
<script>
const D = __DATA__;
const NX=D.nx, NY=D.ny, NZ=D.nz, TH=D.threshold;
function idx(ix,iy,iz){return (ix*NY + iy)*NZ + iz;}

function chainXYZ(arm, k){
  const c = arm.chains[k]; // flat [5*3]
  const X=[],Y=[],Z=[];
  for(let i=0;i<c.length;i+=3){X.push(c[i]);Y.push(c[i+1]);Z.push(c[i+2]);}
  return {X,Y,Z};
}

function baseTraces(arm){
  return [
    {type:'scatter3d',mode:'lines',x:D.body.x,y:D.body.y,z:D.body.z,
     line:{color:'#999',width:3},opacity:0.5,hoverinfo:'skip',name:'body'},
    {type:'scatter3d',mode:'lines',x:D.roi.x,y:D.roi.y,z:D.roi.z,
     line:{color:'#bbb',width:2},hoverinfo:'skip',name:'chest ROI'},
    {type:'scatter3d',mode:'lines+markers',x:[],y:[],z:[],
     line:{color:arm.color,width:8},marker:{size:4,color:arm.color},name:'arm'},
    {type:'scatter3d',mode:'markers',x:[],y:[],z:[],
     marker:{size:6,color:'#000'},name:'tip'},
    {type:'scatter3d',mode:'markers',x:[],y:[],z:[],
     marker:{size:7,color:'#2ca02c',symbol:'x'},name:'target'},
  ];
}
function layout(arm){
  return {title:{text:arm.name,font:{size:14}},showlegend:false,
    margin:{l:0,r:0,t:30,b:0},
    scene:{aspectmode:'data',
      xaxis:{title:'x fwd'},yaxis:{title:'y left'},zaxis:{title:'z up'}}};
}

const arms=D.arms;
Plotly.newPlot('p0', baseTraces(arms[0]), layout(arms[0]), {responsive:true});
Plotly.newPlot('p1', baseTraces(arms[1]), layout(arms[1]), {responsive:true});

function fmt(x){return (x>=0?' ':'')+x.toFixed(3);}
function update(){
  const ix=+sx.value, iy=+sy.value, iz=+sz.value;
  const k=idx(ix,iy,iz);
  const tx=D.xs[ix], ty=D.ys[iy], tz=D.zs[iz];
  vx.textContent=tx.toFixed(3); vy.textContent=ty.toFixed(3); vz.textContent=tz.toFixed(3);

  let cards='';
  ['p0','p1'].forEach((pid,ai)=>{
    const arm=arms[ai];
    const ch=chainXYZ(arm,k);
    const near = arm.smin[k] < TH;
    const reached = arm.reached[k]===1;
    Plotly.restyle(pid,{x:[ch.X],y:[ch.Y],z:[ch.Z],
        'line.color':[near?'#d62728':arm.color],
        'marker.color':[near?'#d62728':arm.color]},[2]);
    Plotly.restyle(pid,{x:[[ch.X[ch.X.length-1]]],y:[[ch.Y[ch.Y.length-1]]],
        z:[[ch.Z[ch.Z.length-1]]],'marker.color':[reached?'#000':'#d62728']},[3]);
    Plotly.restyle(pid,{x:[[tx]],y:[[ty]],z:[[tz]]},[4]);
    const cls = near?'near':(reached?'ok':'');
    const rr = reached?'':' <span class="near">[OUT OF REACH]</span>';
    cards += `<div class="card"><b>${arm.name.split(' ')[0]}</b>${rr}<br>`+
      `w = ${arm.w[k].toFixed(4)}<br>`+
      `<span class="${cls}">&sigma;min = ${arm.smin[k].toFixed(4)}</span>`+
      ` (thr ${TH})<br>cond = ${arm.cond[k].toFixed(1)}<br>`+
      `${near?'<span class="near">*** NEAR SINGULAR ***</span>':'<span class="ok">well-conditioned</span>'}</div>`;
  });
  readout.innerHTML=cards;
}

for(const [el,n,init] of [[sx,NX,D.init[0]],[sy,NY,D.init[1]],[sz,NZ,D.init[2]]]){
  el.min=0; el.max=n-1; el.step=1; el.value=init;
  el.addEventListener('input',update);
}
update();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.04,
                    help="grid spacing in metres (smaller = smoother, bigger file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    path = build(args.step, args.seed)
    if not args.no_open:
        webbrowser.open("file://" + path)


if __name__ == "__main__":
    main()
