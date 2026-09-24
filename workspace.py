"""
Metric 1 -- reachable workspace size.

Densely sample each arm's JOINT space (uniformly inside the real joint limits),
run forward kinematics to get the reachable tool-tip positions, and estimate
the reachable-workspace VOLUME by voxel occupancy.  The volume is reported
normalised by the arm length cubed (V / L^3), which is dimensionless and lets
us compare "how much space the architecture can reach" independently of the
absolute size.

This answers project question (1): does the Reachy-2 Orbita architecture reach
a LARGER workspace than the classic SRS arm?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots import Robot


@dataclass
class WorkspaceResult:
    name: str
    tip: np.ndarray          # (N, 3) reachable tool-tip positions
    volume: float            # reachable volume estimate [m^3]
    volume_norm: float       # V / L^3  (dimensionless)
    reach: float             # arm length L [m]
    voxel: float             # voxel edge used [m]
    n_samples: int


def _occupied_voxels(tip: np.ndarray, voxel: float) -> int:
    keys = np.floor(tip / voxel).astype(np.int64)
    # unique rows
    view = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1])))
    return len(np.unique(view))


def measure_workspace(robot: Robot, n_samples: int, rng: np.random.Generator,
                      voxel_ratio: float = 0.04, batch: int = 200000) -> WorkspaceResult:
    """Dense joint-space sampling -> reachable volume (voxel occupancy) / L^3.

    Parameters
    ----------
    n_samples   : number of joint configurations to sample.
    voxel_ratio : voxel edge as a fraction of the arm length L.
    """
    L = robot.reach
    voxel = voxel_ratio * L

    tips = []
    remaining = n_samples
    while remaining > 0:
        m = min(batch, remaining)
        q = robot.sample_joints(m, rng)
        tip, _, _ = robot.fk_jacobian(q)
        tips.append(tip.astype(np.float32))
        remaining -= m
    tip = np.concatenate(tips)

    n_vox = _occupied_voxels(tip, voxel)
    volume = n_vox * voxel ** 3
    return WorkspaceResult(
        name=robot.name, tip=tip, volume=volume,
        volume_norm=volume / L ** 3, reach=L, voxel=voxel,
        n_samples=n_samples,
    )


def compare_workspace(results: list[WorkspaceResult]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("METRIC 1 -- REACHABLE WORKSPACE SIZE (dense joint-space sampling)")
    lines.append(f"samples/arm = {results[0].n_samples:,}   "
                 f"voxel = {results[0].voxel * 1000:.0f} mm")
    lines.append("=" * 70)
    header = f"{'metric':<30}" + "".join(f"{r.name.split()[0]:>18}" for r in results)
    lines.append(header)
    lines.append("-" * 70)

    def row(label, fn, fmt="{:>18.4f}"):
        lines.append(f"{label:<30}" + "".join(fmt.format(fn(r)) for r in results))

    row("arm length L [m]", lambda r: r.reach)
    row("reachable volume [m^3]", lambda r: r.volume)
    row("normalised V / L^3", lambda r: r.volume_norm)
    base = results[0].volume_norm
    row("relative to SRS [%]", lambda r: 100 * r.volume_norm / base)
    lines.append("=" * 70)
    return "\n".join(lines)
