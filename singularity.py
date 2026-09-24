"""
Metric 2 -- singularities in the chest workspace.

Pipeline (same standard for both arms):

1. Take the chest workspace (a box in front of the torso, CHEST_ROI) and
   DOWN-SAMPLE it on a 3-D equidistant grid of target positions.
2. For every grid position, attach N "maximum-span" tool orientations -- a set
   of approach directions spread as widely as possible over the sphere
   (Fibonacci sphere).  Grid positions x orientations form the benchmark
   pose set ("指标集") the arm is required to achieve.
3. For each benchmark pose solve full 6-DOF, 7-axis inverse kinematics.  The
   arm is redundant, so a redundancy-aware planner would use the 7th DOF to
   steer AWAY from singularities; we credit that by keeping the
   BEST-conditioned IK solution found across several restarts (largest
   sigma_min).  Being able to avoid a singularity is a genuine strength.
4. For each grid position we then take the WORST orientation it is required to
   serve -- the smallest sigma_min across its N reachable orientations
   ("最差奇异情况").  A position is "near-singular" when even after redundancy
   optimisation its worst required orientation still has sigma_min < THRESHOLD.

This answers project question (2): in the chest workspace, does the Reachy-2
Orbita architecture leave FEWER near-singular positions than the SRS arm?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots import Robot, singular_values

# same absolute singular-value threshold for both arms.
SIGMA_THRESHOLD = 0.02

# Chest workspace region of interest (world frame: X forward, Y left, Z up).
# The right-arm shoulder sits at y=-0.20; this box covers the area in front of
# the chest where a dual-arm robot actually manipulates objects.
CHEST_ROI = dict(xmin=0.08, xmax=0.45,
                 ymin=-0.45, ymax=0.12,
                 zmin=-0.25, zmax=0.32)

GRID_STEP = 0.05        # equidistant grid spacing [m]
N_ORIENT = 12           # max-span orientations per grid position
IK_RESTARTS = 6         # redundancy restarts, keep best-conditioned

# Task-oriented orientation set: instead of demanding orientations spread over
# the WHOLE sphere (which structurally penalises any cone-limited wrist), a
# fairer "everyday manipulation" benchmark only requires the tool to point
# roughly into the work area -- forward and slightly down -- within a cone.
APPROACH_DIR = np.array([1.0, 0.0, -1.0])   # forward (+X) and down (-Z)
APPROACH_CONE = np.deg2rad(90.0)            # half-angle of the required cone


@dataclass
class GridResult:
    name: str
    points: np.ndarray        # (G, 3) grid positions that are reachable
    worst_smin: np.ndarray    # (G,) worst-orientation sigma_min at each point
    orient_reach: np.ndarray  # (G,) fraction of the N orientations reachable
    near_mask: np.ndarray     # (G,) bool, worst_smin < threshold
    n_grid_total: int         # grid positions tested (incl. unreachable)
    n_orient: int
    threshold: float
    mode: str = "sphere"      # "sphere" (full-sphere) or "task" (approach cone)

    @property
    def pos_reach_fraction(self) -> float:
        return len(self.points) / max(self.n_grid_total, 1)

    @property
    def near_fraction(self) -> float:
        """Fraction of REACHABLE positions whose worst orientation is singular."""
        return float(self.near_mask.mean()) if len(self.near_mask) else 0.0


def chest_grid() -> np.ndarray:
    """Equidistant 3-D grid of positions inside the chest ROI.  (G0, 3)."""
    r = CHEST_ROI
    xs = np.arange(r["xmin"], r["xmax"] + 1e-9, GRID_STEP)
    ys = np.arange(r["ymin"], r["ymax"] + 1e-9, GRID_STEP)
    zs = np.arange(r["zmin"], r["zmax"] + 1e-9, GRID_STEP)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def _fibonacci_directions(n: int) -> np.ndarray:
    """N unit vectors maximally spread on the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)          # polar
    gold = np.pi * (1.0 + 5.0 ** 0.5)           # golden angle
    theta = gold * i                            # azimuth
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _cone_directions(n: int, axis: np.ndarray, half_angle: float) -> np.ndarray:
    """N unit vectors evenly spread inside a cone of `half_angle` about `axis`.

    Uses a Fibonacci cap so the directions are as uniform as possible over the
    spherical cap (the set of "acceptable" tool approach directions).
    """
    axis = axis / np.linalg.norm(axis)
    cos_h = np.cos(half_angle)
    i = np.arange(n) + 0.5
    # cos(polar) uniformly in [cos_h, 1] -> uniform area over the cap
    cz = 1.0 - (i / n) * (1.0 - cos_h)          # local z (about +Z)
    r = np.sqrt(np.clip(1.0 - cz * cz, 0.0, None))
    gold = np.pi * (1.0 + 5.0 ** 0.5)
    theta = gold * i
    local = np.stack([r * np.cos(theta), r * np.sin(theta), cz], axis=1)
    # rotate local +Z onto `axis`
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ axis)
    if c > 1 - 1e-8:
        return local
    if c < -1 + 1e-8:
        local[:, 2] *= -1
        return local
    v = np.cross(z, axis)
    s = np.linalg.norm(v)
    v /= s
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return local @ R.T


def _rot_x_to(d: np.ndarray) -> np.ndarray:
    """Rotation matrices mapping local +x axis onto each direction d.  (N,3,3)."""
    x = np.array([1.0, 0.0, 0.0])
    out = np.empty((len(d), 3, 3))
    for k, v in enumerate(d):
        v = v / np.linalg.norm(v)
        c = float(x @ v)
        if c > 1 - 1e-8:
            out[k] = np.eye(3)
            continue
        if c < -1 + 1e-8:
            out[k] = np.diag([-1.0, -1.0, 1.0])
            continue
        axis = np.cross(x, v)
        s = np.linalg.norm(axis)
        axis /= s
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        out[k] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return out


def _best_conditioned_ik(robot: Robot, pos: np.ndarray, Rt: np.ndarray,
                         rng: np.random.Generator, restarts: int):
    """Solve IK for a batch of poses, keeping the best-conditioned solution.

    Returns (reached, smin) each of shape (M,); smin is NaN where unreachable.
    """
    M = pos.shape[0]
    best_smin = np.full(M, -np.inf)
    any_reached = np.zeros(M, dtype=bool)
    for _ in range(restarts):
        q0 = robot.sample_joints(M, rng)
        q, reached, _, _ = robot.ik_pose_batch(pos, Rt, q0)
        if not reached.any():
            continue
        _, _, J = robot.fk_jacobian(q)
        smin = singular_values(J)[:, -1]
        take = reached & (smin > best_smin)
        best_smin[take] = smin[take]
        any_reached |= reached
    smin_out = np.where(any_reached, best_smin, np.nan)
    return any_reached, smin_out


def evaluate_grid(robot: Robot, rng: np.random.Generator,
                  threshold: float = SIGMA_THRESHOLD,
                  n_orient: int = N_ORIENT,
                  restarts: int = IK_RESTARTS,
                  mode: str = "sphere",
                  batch: int = 6000) -> GridResult:
    """Chest-grid worst-case singularity evaluation for one arm.

    mode = "sphere" : required orientations spread over the WHOLE sphere
                      (harsh stress test; penalises cone-limited wrists).
    mode = "task"   : required orientations only inside an APPROACH_CONE about
                      APPROACH_DIR (forward+down) -- a fair everyday-manipulation
                      benchmark where the tool just points into the work area.
    """
    grid = chest_grid()                          # (G0, 3)
    G0 = grid.shape[0]
    if mode == "task":
        dirs = _cone_directions(n_orient, APPROACH_DIR, APPROACH_CONE)
    else:
        dirs = _fibonacci_directions(n_orient)
    R_orient = _rot_x_to(dirs)                    # (N, 3, 3)

    # flatten all (grid position x orientation) benchmark poses
    pos_all = np.repeat(grid, n_orient, axis=0)                  # (G0*N, 3)
    R_all = np.tile(R_orient, (G0, 1, 1))                        # (G0*N, 3, 3)

    reached = np.empty(G0 * n_orient, dtype=bool)
    smin = np.empty(G0 * n_orient, dtype=float)
    for s in range(0, G0 * n_orient, batch):
        e = min(s + batch, G0 * n_orient)
        r, sm = _best_conditioned_ik(robot, pos_all[s:e], R_all[s:e], rng, restarts)
        reached[s:e] = r
        smin[s:e] = sm

    reached = reached.reshape(G0, n_orient)
    smin = smin.reshape(G0, n_orient)

    pos_reachable = reached.any(axis=1)          # position reachable if any orient
    orient_reach = reached.mean(axis=1)
    # worst (minimum) sigma_min across reachable orientations at each position
    worst = np.full(G0, np.nan)
    smin_masked = np.where(reached, smin, np.nan)
    worst[pos_reachable] = np.nanmin(smin_masked[pos_reachable], axis=1)

    sel = pos_reachable
    points = grid[sel]
    worst_smin = worst[sel]
    orient_reach_sel = orient_reach[sel]
    near_mask = worst_smin < threshold

    return GridResult(
        name=robot.name, points=points, worst_smin=worst_smin,
        orient_reach=orient_reach_sel, near_mask=near_mask,
        n_grid_total=G0, n_orient=n_orient, threshold=threshold, mode=mode,
    )


def compare_singularity(results: list[GridResult]) -> str:
    lines = []
    mode = results[0].mode
    title = ("FULL-SPHERE orientations (stress test)" if mode == "sphere"
             else f"TASK APPROACH-CONE orientations "
                  f"(+/-{int(np.degrees(APPROACH_CONE))} deg, everyday)")
    lines.append("=" * 78)
    lines.append(f"METRIC 2 -- CHEST-WORKSPACE SINGULARITY -- {title}")
    lines.append(f"grid step = {GRID_STEP * 1000:.0f} mm   "
                 f"orientations/point = {results[0].n_orient}   "
                 f"threshold sigma_min < {results[0].threshold:.3f}")
    lines.append(f"grid positions tested = {results[0].n_grid_total} "
                 f"(worst required orientation per point, 7-DOF IK)")
    lines.append("=" * 78)
    header = f"{'metric':<40}" + "".join(f"{r.name.split()[0]:>18}" for r in results)
    lines.append(header)
    lines.append("-" * 78)

    def row(label, fn, fmt="{:>18.4f}"):
        lines.append(f"{label:<40}" + "".join(fmt.format(fn(r)) for r in results))

    row("reachable positions [%]", lambda r: 100 * r.pos_reach_fraction)
    row("reachable positions [count]", lambda r: len(r.points), fmt="{:>18d}")
    row("mean orientation reachability [%]", lambda r: 100 * r.orient_reach.mean())
    row("near-singular positions [%]", lambda r: 100 * r.near_fraction)
    row("mean worst-case sigma_min", lambda r: np.nanmean(r.worst_smin))
    row("median worst-case sigma_min", lambda r: np.nanmedian(r.worst_smin))
    row("5th pctile worst-case sigma_min", lambda r: np.nanpercentile(r.worst_smin, 5))
    lines.append("=" * 78)
    return "\n".join(lines)


def _per_pose_smin(robot: Robot, pos_all: np.ndarray, R_all: np.ndarray,
                   rng: np.random.Generator, restarts: int = 6,
                   batch: int = 8000):
    """Best-conditioned sigma_min for every (position x orientation) pose.

    Returns (reached, smin), both shape (M,); smin is -1 where unreachable.
    No per-position 'worst orientation' aggregation -- one value per pose.
    """
    M = pos_all.shape[0]
    best = np.full(M, -1.0)
    reached = np.zeros(M, dtype=bool)
    for _ in range(restarts):
        q0 = robot.sample_joints(M, rng)
        q, rc, _, _ = robot.ik_pose_batch(pos_all, R_all, q0)
        _, _, J = robot.fk_jacobian(q)
        sm = singular_values(J)[:, -1]
        take = rc & (sm > best)
        best[take] = sm[take]
        reached |= rc
    return reached, best


def compare_common_poses(robots: list, rng: np.random.Generator,
                         threshold: float = SIGMA_THRESHOLD,
                         n_orient: int = N_ORIENT,
                         mode: str = "task") -> str:
    """Apples-to-apples: compare the arms on the SAME benchmark poses.

    Evaluates per-pose sigma_min (no 'worst per position' aggregation) and
    reports statistics over (a) each arm's own reachable poses and (b) the
    intersection of poses BOTH arms can reach.  This removes the selection
    bias that flatters the arm which reaches fewer (easier) poses.
    """
    grid = chest_grid()
    if mode == "task":
        dirs = _cone_directions(n_orient, APPROACH_DIR, APPROACH_CONE)
    else:
        dirs = _fibonacci_directions(n_orient)
    Ro = _rot_x_to(dirs)
    pos_all = np.repeat(grid, n_orient, axis=0)
    R_all = np.tile(Ro, (len(grid), 1, 1))

    res = [(_per_pose_smin(r, pos_all, R_all, rng)) for r in robots]
    reached = [x[0] for x in res]
    smin = [x[1] for x in res]
    both = reached[0] & reached[1]

    lines = []
    lines.append("=" * 78)
    lines.append("METRIC 2b -- PER-POSE sigma_min, APPLES-TO-APPLES "
                 f"({mode} orientations)")
    lines.append(f"total benchmark poses = {len(pos_all):,}   "
                 f"threshold sigma_min < {threshold:.3f}")
    lines.append("=" * 78)
    hdr = f"{'metric':<40}" + "".join(f"{r.name.split()[0]:>18}" for r in robots)
    lines.append(hdr)
    lines.append("-" * 78)

    def row(label, vals, fmt="{:>18.4f}"):
        lines.append(f"{label:<40}" + "".join(fmt.format(v) for v in vals))

    row("reachable poses [count]", [int(m.sum()) for m in reached], fmt="{:>18d}")
    lines.append(f"{'poses BOTH arms reach [count]':<40}{both.sum():>18d}")
    lines.append("-- over EACH arm's OWN reachable poses " + "-" * 39)
    row("median sigma_min", [np.median(smin[i][reached[i]]) for i in range(2)])
    row("near-singular [%]",
        [100 * np.mean(smin[i][reached[i]] < threshold) for i in range(2)])
    lines.append("-- over the COMMON poses both arms reach " + "-" * 37)
    row("median sigma_min", [np.median(smin[i][both]) for i in range(2)])
    row("near-singular [%]",
        [100 * np.mean(smin[i][both] < threshold) for i in range(2)])
    lines.append("=" * 78)
    return "\n".join(lines)
