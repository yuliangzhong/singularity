# Does Reachy 2 (Orbita) beat a classic SRS arm?

A 3-D simulation that answers two concrete questions about the Reachy 2
`2-2-3` Orbita arm architecture versus a classic `3-1-3` SRS serial arm:

1. **Is the reachable workspace larger?**
2. **In the chest workspace, are there fewer singularities?**

Both arms are modelled with screw theory (product-of-exponentials) as serial
revolute chains, keeping only what matters for the comparison: the **DOF
grouping**, **realistic serial link lengths**, and the **joint travel limits**.

| | grouping | joints | limits |
|---|---|---|---|
| **SRS** (KUKA iiwa-like) | Spherical–Revolute–Spherical, `3-1-3` | 7 serial revolute | **real KUKA iiwa** travel (±170/120/170/120/170/120/175°) |
| **Orbita** (Reachy 2) | Orbita2d + Orbita2d + Orbita3d, `2-2-3` | 7 | **unlimited** rotation |

Both arms are given the **same total reach L** and realistic proportions
(shoulder module 0.05, upper arm 0.28, forearm 0.25, wrist module 0.06,
tool 0.07 m — every sub-joint separated by a short non-zero link), mounted at
the same shoulder with the Reachy-2 shoulder orientation (−15° roll, +10° yaw)
so the workspace sits in front of the chest.

## Metric 1 — reachable workspace size

[workspace.py](workspace.py) densely samples each arm's **joint space**
(uniformly inside the real limits), runs FK to get the reachable tool-tip
positions, and estimates the reachable **volume by voxel occupancy**, reported
normalised by the arm length cubed **V / L³** (dimensionless).

## Metric 2 — chest-workspace singularity

[singularity.py](singularity.py):

1. Down-sample the chest workspace on a **3-D equidistant grid** of target
   positions (`CHEST_ROI`, `GRID_STEP`).
2. Attach **N maximum-span orientations** to every grid position — approach
   directions spread as widely as possible over the sphere (Fibonacci lattice).
   Positions × orientations = the benchmark pose set ("指标集").
3. Solve full 6-DOF, **7-axis IK** for every benchmark pose. The arm is
   redundant, so a redundancy-aware planner uses the 7th DOF to steer *away*
   from singularities — we credit that by keeping the **best-conditioned** IK
   solution (largest σ_min) across several restarts.
4. For each grid position take the **worst orientation** it must serve — the
   smallest σ_min across its reachable orientations ("最差奇异情况"). A position
   is **near-singular** when even after redundancy optimisation its worst
   required orientation still has σ_min < threshold (default `0.02`).

Singularity proximity uses the smallest singular value of the `6×7` geometric
Jacobian, `σ_min(J)`, with the **same absolute threshold for both arms**.

## Usage

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt        # numpy matplotlib scipy plotly

python run_comparison.py                                  # both metrics + plots
python run_comparison.py --ws-samples 2000000 --n-orient 16 --no-show
```

Outputs (in `outputs/`):
- `workspace_compare.png` — reachable clouds + normalised-volume bar (Metric 1).
- `singularity_compare.png` — chest grid coloured by worst-case σ_min, near-singular positions marked; distribution histogram (Metric 2).
- `singularity_interactive.html` — rotatable 3-D Plotly viewer of the chest grid.

### Interactive dual-arm visualizer (browser, no matplotlib)

[visualizer.py](visualizer.py) **pre-computes** best-conditioned 7-DOF IK for
both arms on a dense grid of chest targets and writes a single self-contained
`outputs/interactive_viewer.html` driven by Plotly.js. Because dragging is a
pure browser-side table lookup, it responds instantly and **never freezes**
(unlike interactive matplotlib), and the HTML can be shared as-is.

```bash
python visualizer.py                 # build the HTML and open it in the browser
python visualizer.py --no-open       # just build the HTML
python visualizer.py --step 0.03     # finer grid (smoother, larger file)
```

- Two side-by-side 3-D views (SRS and Orbita) share **one target**, with the
  half-body drawn for context.
- Three **X / Y / Z sliders** move the shared end-effector target.
- Live readout per arm: manipulability `w`, smallest singular value `σ_min`,
  condition number, an **OUT OF REACH** flag, and a red **NEAR SINGULAR** flag
  (the arm turns red) when `σ_min < threshold`.

## Typical result

**Metric 1 — workspace size** (2 M joint samples/arm, voxel 28 mm):

| | SRS | Orbita |
|---|---|---|
| normalised V / L³ | **4.06** | 3.89 |
| relative | 100 % | 95.8 % |

The positional workspaces are almost identical (SRS marginally larger). The
unlimited wrist does **not** buy a bigger *positional* workspace — tip position
is set mainly by the shoulder + elbow, which both architectures span similarly.

**Metric 2 — chest singularity** (grid 50 mm, 16 orientations/point):

| | SRS | Orbita |
|---|---|---|
| reachable positions | 96.1 % | **100 %** |
| mean orientation reachability | 75.6 % | **96.1 %** |
| near-singular positions (worst orientation) | **2.4 %** | 12.4 % |
| mean worst-case σ_min | **0.082** | 0.070 |
| median worst-case σ_min | **0.081** | 0.076 |

The honest, nuanced answer:

- **Orbita wins big on orientation dexterity** — it reaches essentially every
  chest position and ~96 % of all required orientations, versus 76 % for the
  limited SRS wrist. This is exactly what the continuous-rotation Orbita
  actuators buy.
- **But Orbita has MORE near-singular positions in the worst case.** Because it
  can reach many hard orientations that the SRS wrist simply cannot, some of
  those newly-reachable orientations sit near the Orbita3d wrist's *intrinsic*
  gimbal singularities. The SRS is never "penalised" for those orientations —
  it just can't reach them, so they are excluded from its worst case.

So Reachy 2's advantage is **reachability / dexterity**, not a larger workspace
and not fewer singularities: the unlimited wrist trades a few intrinsic
worst-case singular spots for a large gain in achievable orientations.

## Files

- [robots.py](robots.py) — kinematics (FK, geometric Jacobian, position IK, batched 6-DOF pose IK) and the two robot definitions.
- [workspace.py](workspace.py) — Metric 1: dense joint-space sampling → V / L³.
- [singularity.py](singularity.py) — Metric 2: chest grid × max-span orientations → 7-DOF IK worst-case singularity.
- [run_comparison.py](run_comparison.py) — runs both metrics, prints tables, saves figures and the interactive HTML.
- [visualizer.py](visualizer.py) — builds the interactive browser viewer (pre-computed IK, Plotly.js; no matplotlib, never freezes).

## Assumptions & simplifications

- Both arms share the same reach L and realistic proportions; SRS uses the real
  KUKA iiwa joint limits, Orbita is unlimited (full `[-π, π]`, no stops).
- The Orbita2d/3d actuators are modelled as ideal serial revolute clusters (the
  internal parallel mechanism is ignored, as requested).
- Redundancy is credited: for each pose the best-conditioned 7-DOF IK solution
  found across restarts is used (a redundancy-aware planner avoiding
  singularities). The same rule is applied to both arms.
- Worst-case σ_min per position is taken over each arm's **own reachable**
  orientations, so the near-singular fraction must be read together with the
  orientation-reachability figure.
- Volume is a voxel-occupancy estimate; the normalised V / L³ ratio is the
  robust comparison, not the absolute m³.
