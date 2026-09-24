# SRS vs Reachy 2 (Orbita): a fair kinematic comparison

A 3-D simulation comparing a classic **`3-1-3` SRS** 7-DOF serial arm against the
Reachy 2 **`2-2-3` Orbita** arm (2×Orbita2d shoulder/elbow + Orbita3d parallel
wrist). It was built to answer, honestly:

1. **Is the Orbita arm's reachable workspace larger?**
2. **In the chest workspace, does the Orbita arm have fewer singularities?**

**Short answer (this framework): no to both.** With realistic joint limits the
Orbita arm reaches a *smaller* workspace and *fewer* orientations, and once you
compare the two arms **on the same poses**, the SRS arm is at least as clear of
singularities. Orbita's real advantages (stiffness, compactness, precision,
confined-space clearance) are mechanical and lie **outside** a pure kinematic
singularity analysis — which is exactly what the ByteWrist paper actually claims.

## Model

Both arms are modelled with screw theory (product-of-exponentials) as serial
revolute chains, keeping what matters for the comparison: **DOF grouping**,
**realistic serial link lengths**, and **joint travel limits**.

| | grouping | joints | limits |
|---|---|---|---|
| **SRS** (KUKA iiwa-like) | Spherical–Revolute–Spherical, `3-1-3` | 7 serial revolute | **real KUKA iiwa** travel (±170/120/170/120/170/120/175°) |
| **Orbita** (Reachy 2) | Orbita2d + Orbita2d + Orbita3d, `2-2-3` | 7 | **real Reachy 2 limits + parallel-wrist cone** |

Both arms get the **same total reach L** and realistic proportions (shoulder
module 0.05, upper arm 0.28, forearm 0.25, wrist module 0.06, tool 0.07 m — every
sub-joint separated by a short non-zero link), mounted at the same shoulder with
the Reachy 2 shoulder orientation (−15° roll, +10° yaw) so the workspace sits in
front of the chest.

### Orbita joint limits — sources

Taken from the **official Pollen Reachy 2 URDF**
(`reachy2_symbolic_ik/config_files/reachy2.urdf`) and cross-checked against the
**ByteDance ByteWrist paper** (arXiv:2509.18084), whose ByteMini robot uses the
same 7-DoF SRS arm layout with a 3-DoF parallel wrist:

- shoulder: two DOF, ≈ ±90° each
- elbow: yaw ±90°, pitch **−129° … +5.7°** (anthropomorphic, bends one way).
  *Caveat:* this is a single-source URDF value and may be a conservative safety
  limit rather than the mechanical maximum.
- wrist (Orbita3d): the two **tilt** DOF are **not** two free ±180° axes — they
  are confined to a **cone**. Reachy casts the wrist into a cone of half-angle
  **42.5°** (`limit_orbita3d`, `orbita3D_max_angle = radians(42.5)`); ByteWrist
  gives `β² + γ² < 0.72 rad²` (half-angle **≈ 48.6°**). We use **45°**, between
  the two. The wrist **roll** about the arm axis stays free (multi-turn).

The cone is enforced everywhere: joint sampling draws the two tilt DOF uniformly
inside the 45° disk, and every IK iteration projects the wrist tilt back onto the
cone (`Robot._project_limits`).

## Two metrics

**Metric 1 — reachable workspace size** ([workspace.py](workspace.py)): densely
sample each arm's **joint space** inside the real limits, run FK, and estimate the
reachable **volume by voxel occupancy**, normalised by arm length cubed **V / L³**.

**Metric 2 — chest-workspace singularity** ([singularity.py](singularity.py)):

1. Down-sample the chest ROI on a **3-D equidistant grid** of target positions.
2. Attach **N required orientations** per position (`evaluate_grid(mode=...)`):
   a **task approach-cone** (±90° about a forward-and-down direction — the fair
   everyday set) or a **full-sphere** Fibonacci spread (harsh stress test).
3. Solve full 6-DOF, **7-axis IK** for every pose; credit redundancy by keeping
   the **best-conditioned** solution (largest σ_min) over several restarts.
4. Two readouts:
   - **2a (per-position worst orientation):** for each position take the smallest
     σ_min over its reachable orientations. *This is sensitive to how many
     orientations each arm can reach (see the artifact warning below).*
   - **2b (apples-to-apples per pose):** compare σ_min pose-by-pose, and over the
     **intersection of poses both arms can reach** — the bias-free test.

Singularity proximity is the smallest singular value of the `6×7` geometric
Jacobian, `σ_min(J)`, with the **same absolute threshold (0.02) for both arms**.

## Results

### Metric 1 — workspace size (2 M joint samples/arm, voxel 28 mm)

| | SRS | Orbita |
|---|---|---|
| normalised V / L³ | **4.06** | 2.87 |
| relative | 100 % | **70.8 %** |

Orbita reaches a **smaller** positional workspace (~71 % of SRS). Decomposing the
cause (position volume is set by shoulder + elbow; the wrist only adds a thin
tool-length shell):

| Orbita variant | V / L³ | of SRS |
|---|---|---|
| real limits + 45° wrist cone | 2.75 | 71 % |
| real limits, wrist ±45° box (no cone) | 2.78 | 72 % |
| real shoulder/elbow, wrist widened to ±120° | 2.83 | 73 % |

The **wrist cone is *not* the cause** — widening the wrist to ±120° recovers only
~2 points. The envelope is dominated by the real **shoulder ±90°** and the
**one-way elbow bend** (−129°…+5.7°). **Answer to Q1: no, not larger.**

### Metric 2a — per-position worst orientation (grid 50 mm, 16 orient/pt)

| task approach-cone (±90°) | SRS | Orbita |
|---|---|---|
| reachable positions | **92.2 %** | 49.1 % |
| mean orientation reachability | **75.5 %** | 15.8 % |
| near-singular positions | 1.3 % | **0.0 %** |
| median worst-case σ_min | 0.085 | **0.112** |

| full-sphere (stress test) | SRS | Orbita |
|---|---|---|
| reachable positions | **95.8 %** | 58.8 % |
| mean orientation reachability | **76.1 %** | 12.6 % |
| near-singular positions | 2.1 % | **0.1 %** |
| median worst-case σ_min | 0.078 | **0.120** |

> ⚠️ **The "Orbita 0.0 %" here is a selection artifact, not an advantage.**
> "Worst of the reachable orientations" is a *worst-of-many* vs *worst-of-few*
> statistic. SRS serves ~12 orientations per position, so its worst is easily a
> hard one; Orbita serves only ~2–3 easy orientations, so its worst is still
> easy. This *aggregation* flatters the arm that reaches fewer poses. The robust
> facts here are the top rows: Orbita reaches **far fewer** positions and
> orientations.

### Metric 2b — apples-to-apples (task orientations, per pose)

| | SRS | Orbita |
|---|---|---|
| reachable poses [count] | **12 775** | 1 446 |
| poses **both** arms reach | 1 321 | 1 321 |
| median σ_min — own reachable poses | **0.155** | 0.130 |
| near-singular — own reachable poses | 0.13 % | 0.00 % |
| median σ_min — **common** poses | **0.162** | 0.130 |
| near-singular — **common** poses | 0.15 % | 0.00 % |

On the **same** poses both arms can reach, they are **both extremely clean**
(near-singular ≪ 1 %), and the SRS arm actually has the **higher** median σ_min
(0.162 vs 0.130) — i.e. it sits *farther* from singularity, not closer. **Answer
to Q2: no, Orbita does not have fewer singularities on a fair comparison.**

## Discussion

**"If you can reach places others can't, aren't you necessarily closer to
singularity?"** The precise version is a theorem: the **boundary** of a
reachable set is always singular (the Jacobian drops rank at the reach limit). So
stretching to a far pose does push you toward singularity. But this does **not**
describe Orbita — Orbita reaches *fewer* poses, not more, so it is never the arm
"reaching farther."

**"In Orbita's own reachable region, how does SRS do?"** At least as well — see
Metric 2b: on the common set the SRS median σ_min is *higher*. Wherever Orbita can
go, SRS can go and is equally or better conditioned.

**"Can't SRS avoid singularities too?"** Yes. Both are **7R arms on a 6D task**,
so both have exactly a **1-D null space** — identical redundancy, identical
ability to steer around singularities. There is no difference in kinematic
singularity-avoidance capability.

**Then what is the essential difference?** It is *not* singularity avoidance. A
classic **serial** spherical wrist loses a DOF when its two roll axes align, but
in a **7-DOF arm the redundant shoulder/elbow compensate**, so the *arm* stays
conditioned even when the *wrist subchain* is singular:

| wrist pitch | 3-DOF wrist subchain σ_min | full 7-DOF arm σ_min |
|---|---|---|
| 0° | 1.00 | 0.154 |
| 90° (roll axes aligned) | **0.00 (rank loss)** | **0.155 (still healthy)** |

So the serial-wrist singularity that motivates a parallel wrist is **already
neutralised by redundancy** at the arm level. The true difference between the two
architectures is a **mechanical trade-off** that pure kinematics cannot see:

| | serial SRS wrist | parallel Orbita wrist |
|---|---|---|
| orientation range | **large** | small (±45° cone) |
| positional workspace | **larger** | smaller |
| stiffness | lower (cantilevered chain) | **high** (parallel + centre ball) |
| accumulated error | yes | **no** |
| compactness / clearance | bulkier, longer forearm | **compact** |

## Conclusion

Within this position + orientation + singularity kinematic framework, the Orbita
arm has **no defensible advantage** over SRS: smaller workspace, far fewer
reachable orientations, and — on a fair like-for-like pose set — no fewer
singularities (SRS is if anything better conditioned). The eye-catching
"0 % near-singular" is an aggregation artifact of Orbita reaching fewer poses.

Orbita's genuine benefits — **compactness, stiffness, precision, confined-space
clearance** — are mechanical and are **not measurable** by this analysis. This is
consistent with the ByteWrist paper, which wins its glove-box comparison against a
serial Kinova wrist through **short forearm and no collisions**, and never claims
a larger workspace or fewer singularities.

## Usage

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt        # numpy matplotlib scipy plotly

python run_comparison.py                                   # both metrics + 2a/2b
python run_comparison.py --ws-samples 2000000 --n-orient 16 --no-show
```

Outputs (in `outputs/`):
- `workspace_compare.png` — reachable clouds + normalised-volume bar (Metric 1).
- `singularity_compare.png` — chest grid coloured by worst-case σ_min (Metric 2a).
- `singularity_interactive.html` — rotatable 3-D Plotly viewer of the chest grid.

### Interactive dual-arm visualizer (browser, no matplotlib)

[visualizer.py](visualizer.py) **pre-computes** best-conditioned 7-DOF IK for both
arms on a dense grid of chest targets and writes a self-contained
`outputs/interactive_viewer.html` (Plotly.js). Dragging is a pure browser-side
table lookup, so it responds instantly and **never freezes**.

```bash
python visualizer.py                 # build the HTML and open it in the browser
python visualizer.py --no-open       # just build the HTML
python visualizer.py --step 0.03     # finer grid (smoother, larger file)
```

Two side-by-side 3-D views share one draggable target (X/Y/Z sliders), with the
half-body drawn for context and a live per-arm readout of `w`, `σ_min`, condition
number, an **OUT OF REACH** flag and a **NEAR SINGULAR** flag.

## Files

- [robots.py](robots.py) — kinematics (FK, geometric Jacobian, position IK, batched 6-DOF pose IK) and the two robot definitions.
- [workspace.py](workspace.py) — Metric 1: dense joint-space sampling → V / L³.
- [singularity.py](singularity.py) — Metric 2: chest grid × orientations (task-cone or full-sphere) → worst-case σ_min (2a), plus the apples-to-apples per-pose / common-set comparison (2b).
- [run_comparison.py](run_comparison.py) — runs both metrics (2a + 2b), prints tables, saves figures and the interactive HTML.
- [visualizer.py](visualizer.py) — builds the interactive browser viewer.

## Assumptions & simplifications

- Both arms share reach L and realistic proportions. SRS uses real KUKA iiwa
  limits; Orbita uses real Reachy 2 URDF limits + a ±45° wrist cone (Reachy 42.5°
  / ByteWrist 48.6°) with free wrist roll.
- Orbita2d/3d actuators are modelled as ideal serial revolute clusters with the
  correct travel/cone limits; the internal parallel linkage geometry itself is
  not simulated (only its motion range) — so stiffness/precision advantages are
  **out of scope**.
- Redundancy is credited equally for both arms (best-conditioned IK over
  restarts).
- Metric 2a is sensitive to orientation-reachability and must be read with 2b;
  volume is a voxel estimate, so the V / L³ ratio (not absolute m³) is the robust
  figure.

## Sources

- Pollen Robotics — Reachy 2 URDF, `reachy2_symbolic_ik/config_files/reachy2.urdf` (joint limits; `limit_orbita3d` 42.5° wrist cone).
- Pollen Robotics — Reachy 2 hardware specification (shoulder −15° roll / +10° yaw; 2×Orbita2d + Orbita3d arm).
- J. Tian et al., "ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces," ByteDance Seed, arXiv:2509.18084 (7-DoF SRS arm, parallel wrist cone `β²+γ² < 0.72`, glove-box comparison vs serial Kinova).
