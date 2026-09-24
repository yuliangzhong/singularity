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
                  batch: int = 6000) -> GridResult:
    """Chest-grid worst-case singularity evaluation for one arm."""
    grid = chest_grid()                          # (G0, 3)
    G0 = grid.shape[0]
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
        n_grid_total=G0, n_orient=n_orient, threshold=threshold,
    )


def compare_singularity(results: list[GridResult]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("METRIC 2 -- CHEST-WORKSPACE SINGULARITY (grid x max-span orientations)")
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
