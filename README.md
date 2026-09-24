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
| **Orbita** (Reachy 2) | Orbita2d + Orbita2d + Orbita3d, `2-2-3` | 7 | **real Reachy 2 limits + parallel-wrist cone** |

Both arms are given the **same total reach L** and realistic proportions
(shoulder module 0.05, upper arm 0.28, forearm 0.25, wrist module 0.06,
tool 0.07 m — every sub-joint separated by a short non-zero link), mounted at
the same shoulder with the Reachy-2 shoulder orientation (−15° roll, +10° yaw)
so the workspace sits in front of the chest.

### Orbita joint limits — sources

The Orbita limits are taken from the **official Pollen Reachy 2 URDF**
(`reachy2_symbolic_ik/config_files/reachy2.urdf`) and cross-checked against the
**ByteDance ByteWrist paper** (arXiv:2509.18084), whose ByteMini robot uses the
same 7-DoF SRS arm with a 3-DoF parallel wrist:

- shoulder: two DOF, ≈ ±90° each
- elbow: yaw ±90°, pitch **−129° … +5.7°** (anthropomorphic, bends one way)
- wrist (Orbita3d): the two **tilt** DOF are **not** two free ±180° axes — they
  are confined to a **cone**. Reachy casts the wrist into a cone of half-angle
  **42.5°** (`limit_orbita3d`, `orbita3D_max_angle = radians(42.5)`); ByteWrist
  gives `β² + γ² < 0.72 rad²` (half-angle **≈ 48.6°**). We use **45°**, between
  the two. The wrist **roll** about the arm axis stays free (multi-turn).

The cone is enforced everywhere: joint sampling draws the two tilt DOF uniformly
inside the disk of radius 45°, and every IK iteration projects the wrist tilt
back onto the cone (`Robot._project_limits`).

## Metric 1 — reachable workspace size

[workspace.py](workspace.py) densely samples each arm's **joint space**
(uniformly inside the real limits), runs FK to get the reachable tool-tip
positions, and estimates the reachable **volume by voxel occupancy**, reported
normalised by the arm length cubed **V / L³** (dimensionless).

## Metric 2 — chest-workspace singularity

[singularity.py](singularity.py):

1. Down-sample the chest workspace on a **3-D equidistant grid** of target
   positions (`CHEST_ROI`, `GRID_STEP`).
2. Attach **N required orientations** to every grid position. Two orientation
   sets are supported (`evaluate_grid(mode=...)`): a **task approach-cone**
   (±90° about a forward-and-down direction — the fair everyday set) and a
   **full-sphere** Fibonacci spread (the harsh stress test).
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
| normalised V / L³ | **4.06** | 2.87 |
| relative | 100 % | 70.8 % |

With the **real** Reachy 2 limits (and the ±45° parallel-wrist cone), the Orbita
arm reaches a **noticeably smaller** positional workspace — about **71 %** of the
SRS arm. So the answer to Q1 is **no — Orbita does not reach a larger workspace**.

Decomposing the cause (position volume is set by shoulder + elbow, the wrist only
adds a small tool-length shell):

| Orbita variant | V / L³ | of SRS |
|---|---|---|
| real limits + 45° wrist cone | 2.75 | 71 % |
| real limits, wrist ±45° box (no cone) | 2.78 | 72 % |
| real shoulder/elbow, wrist widened to ±120° | 2.83 | 73 % |

So the **wrist cone is *not* the cause** of the workspace shrinkage — widening the
wrist to ±120° only recovers ~2 points. The envelope is dominated by the real
**shoulder ±90°** and the **one-way elbow bend** (−129°…+5.7°).

**Metric 2 — chest singularity.** Each grid position must serve a set of tool
orientations; we report the **worst** orientation per position (7-DOF IK,
best-conditioned solution). Two orientation sets are evaluated with the same code
([singularity.py](singularity.py), `evaluate_grid(mode=...)`):

- **Task approach-cone** (*fair, everyday*): orientations only inside a ±90°
  cone about a **forward-and-down** approach direction — the tool just has to
  point into the work area, which is all a chest-front manipulation task needs.
- **Full-sphere** (*harsh stress test*): orientations spread over the whole
  sphere, including tool-up / tool-back poses no chest task requires.

**(a) Task approach-cone — the fair everyday benchmark** (grid 50 mm, 16 orient/pt):

| | SRS | Orbita |
|---|---|---|
| reachable positions | **92.2 %** | 49.1 % |
| mean orientation reachability | **75.5 %** | 15.8 % |
| near-singular positions (worst orientation) | 1.3 % | **0.0 %** |
| mean worst-case σ_min | 0.086 | **0.111** |
| median worst-case σ_min | 0.085 | **0.112** |

**(b) Full-sphere — the harsh stress test** (grid 50 mm, 16 orient/pt):

| | SRS | Orbita |
|---|---|---|
| reachable positions | **95.8 %** | 58.8 % |
| mean orientation reachability | **76.1 %** | 12.6 % |
| near-singular positions (worst orientation) | 2.1 % | **0.1 %** |
| mean worst-case σ_min | 0.079 | **0.119** |
| median worst-case σ_min | 0.078 | **0.120** |

The honest, nuanced answer (holds in **both** benchmarks):

- The parallel wrist is **far more restricted in orientation**. Even on the fair
  ±90° approach-cone benchmark it serves only ~**16 %** of the required
  orientations (vs ~76 % for the serial SRS wrist) and reaches ~**49 %** of chest
  positions with *any* required orientation. A ±90° half-cone is a whole
  hemisphere, and a ±45° wrist cone still cannot cover it — so restricting the
  benchmark from full-sphere to a hemisphere barely helps Orbita.
- **But within the poses it can reach, Orbita stays markedly clearer of
  singularity**: its worst-case σ_min distribution is shifted well to the right
  (median ≈ 0.11 vs 0.08) and **~0 %** of positions are near-singular, vs 1–2 %
  for the SRS arm.

The low orientation figure is **physically consistent**, not a bug: a 45°
half-cone covers only `2π(1−cos45°) ≈ 14.6 %` of the sphere, so a cone wrist
cannot reach broadly spread orientations no matter the benchmark.

This matches the ByteWrist paper's thesis: the parallel wrist trades **orientation
range / workspace** for **compactness, stiffness and clean, singularity-free
motion** in the poses it does reach — which is exactly what makes it strong in
confined, chest-front dual-arm tasks even though its raw reachability is lower.

> **Caveat.** The worst-case σ_min per position is taken over each arm's *own*
> reachable orientations. Because Orbita reaches far fewer orientations, its
> "worst case" is over a smaller (easier) set, which flatters its near-singular
> fraction. The two metrics must therefore be read **together**: Orbita is
> cleaner *but* over a much smaller reachable set.

## Files

- [robots.py](robots.py) — kinematics (FK, geometric Jacobian, position IK, batched 6-DOF pose IK) and the two robot definitions.
- [workspace.py](workspace.py) — Metric 1: dense joint-space sampling → V / L³.
- [singularity.py](singularity.py) — Metric 2: chest grid × orientations (task approach-cone or full-sphere) → 7-DOF IK worst-case singularity.
- [run_comparison.py](run_comparison.py) — runs both metrics, prints tables, saves figures and the interactive HTML.
- [visualizer.py](visualizer.py) — builds the interactive browser viewer (pre-computed IK, Plotly.js; no matplotlib, never freezes).

## Assumptions & simplifications

- Both arms share the same reach L and realistic proportions. SRS uses the real
  KUKA iiwa joint limits; Orbita uses the real Reachy 2 URDF limits plus a ±45°
  parallel-wrist cone (Reachy 42.5° / ByteWrist 48.6°) with a free wrist roll.
- The Orbita2d/3d actuators are modelled as ideal serial revolute clusters with
  the correct travel/cone limits; the internal parallel linkage geometry itself
  is not simulated (only its motion range).
- Redundancy is credited: for each pose the best-conditioned 7-DOF IK solution
  found across restarts is used (a redundancy-aware planner avoiding
  singularities). The same rule is applied to both arms.
- Worst-case σ_min per position is taken over each arm's **own reachable**
  orientations, so the near-singular fraction must be read together with the
  orientation-reachability figure (see caveat above).
- Volume is a voxel-occupancy estimate; the normalised V / L³ ratio is the
  robust comparison, not the absolute m³.

## Sources

- Pollen Robotics — Reachy 2 URDF, `reachy2_symbolic_ik/config_files/reachy2.urdf` (joint limits, `limit_orbita3d` 42.5° wrist cone).
- Pollen Robotics — Reachy 2 hardware specification (shoulder −15° roll / +10° yaw, 2×Orbita2d + Orbita3d arm).
- J. Tian et al., "ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces," ByteDance Seed, arXiv:2509.18084 (7-DoF SRS arm, parallel wrist cone `β²+γ² < 0.72`, vs serial Kinova).
