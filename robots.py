"""
Kinematic models for two 7-DOF arm architectures, built for a fair,
same-size, same-standard singularity comparison.

Both arms are modelled as serial revolute chains using screw theory
(product-of-exponentials).  We deliberately IGNORE the concrete mechanical
implementation and keep only what matters for singularity analysis:

  * the arrangement of the rotational degrees of freedom,
  * short but non-zero link lengths between the sub-joints (so the shoulder
    and wrist are real serial clusters, not single points), and
  * the joint travel limits.

------------------------------------------------------------------------
Architecture 1 -- "SRS" (Spherical - Revolute - Spherical)
    Classic anthropomorphic 7-DOF serial arm (e.g. KUKA LBR iiwa).
    DOF grouping: shoulder [3] + elbow [1] + wrist [3]  ->  3-1-3
    Every axis is a *finite-travel* revolute joint, and the three shoulder
    axes / three wrist axes are distributed along a short module length.

Architecture 2 -- "Orbita" (Reachy 2, Pollen Robotics)
    DOF grouping: shoulder Orbita2d [2] + elbow Orbita2d [2]
                  + wrist Orbita3d [3]                    ->  2-2-3
    Joint travel is taken from the official Pollen Reachy 2 URDF and
    cross-checked with the ByteDance ByteWrist paper (arXiv:2509.18084):
    the shoulder/elbow are finite revolute pairs and the parallel Orbita3d
    wrist tilts inside a CONE (half-angle ~45 deg) with a free roll about the
    arm axis -- NOT three independent unlimited axes.  See make_orbita.

Both arms share identical link lengths so the workspaces are directly
comparable, and both are mounted at the same shoulder location with the
Reachy-2 shoulder orientation (-15 deg roll, +10 deg yaw) so their natural
workspace sits in front of the chest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------
# Shared geometry (metres).  Both arms are given the SAME total reach L so the
# chest-workspace singularity comparison is apples-to-apples, and the reachable
# workspace volume is reported normalised by L^3 (dimensionless).  Proportions
# follow realistic anthropomorphic / Reachy-2 ratios and every sub-joint is
# separated by a short but non-zero serial link (no coincident axes).
# SRS additionally uses the real KUKA iiwa joint travel limits (see make_srs).
# ----------------------------------------------------------------------
D_SHOULDER = 0.05   # short shoulder-cluster module length
L_UPPER = 0.28      # upper arm  (shoulder cluster -> elbow)
L_FORE = 0.25       # forearm    (elbow -> wrist cluster)
D_WRIST = 0.06      # short wrist-cluster module length
L_TOOL = 0.07       # wrist -> tool tip

# total shoulder->tip length, identical for both arms
L_TOTAL = D_SHOULDER + L_UPPER + L_FORE + D_WRIST + L_TOOL

# tool tip in the arm-local frame (arm extends along +X at home)
TIP = np.array([L_TOTAL, 0.0, 0.0])

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])

# Right-shoulder mount, world frame: X forward, Y left, Z up.
SHOULDER_POS = np.array([0.0, -0.20, 0.0])
SHOULDER_ROLL = np.deg2rad(-15.0)   # Reachy-2 shoulder roll
SHOULDER_YAW = np.deg2rad(10.0)     # Reachy-2 shoulder yaw


# ----------------------------------------------------------------------
# SE(3) helpers
# ----------------------------------------------------------------------
def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _base_transform() -> np.ndarray:
    """Shoulder mount: place at SHOULDER_POS with Reachy-2 orientation.

    At home the arm hangs downward (local +X -> world -Z) then the Reachy
    shoulder yaw/roll tilt it slightly forward and inward.
    """
    R = _rot("z", SHOULDER_YAW) @ _rot("x", SHOULDER_ROLL) @ _rot("y", np.pi / 2)
    return se3(R, SHOULDER_POS)


@dataclass
class Robot:
    """A serial revolute chain described by screw axes at the home pose."""

    name: str
    omegas: np.ndarray          # (n, 3) unit rotation axes at home pose (local)
    points: np.ndarray          # (n, 3) a point on each axis at home (local)
    lower: np.ndarray           # (n,) lower joint limits [rad]
    upper: np.ndarray           # (n,) upper joint limits [rad]
    segment_idx: list[int]      # joint indices for shoulder/elbow/wrist markers
    base: np.ndarray = field(default_factory=_base_transform)
    unlimited: np.ndarray = field(default=None)  # (n,) bool, informational
    # Orbita-style parallel wrist: the two tilt DOFs (cone_idx) can only tilt
    # the wrist axis inside a cone of half-angle cone_angle, i.e.
    # sqrt(q[i]^2 + q[j]^2) <= cone_angle.  The wrist roll about the arm axis
    # stays free.  See make_orbita for sources (Reachy URDF, ByteWrist paper).
    cone_idx: tuple = None       # (i, j) joint indices forming the tilt cone
    cone_angle: float = 0.0      # cone half-angle [rad]

    @property
    def n(self) -> int:
        return self.omegas.shape[0]

    @property
    def reach(self) -> float:
        """Total shoulder->tip length L (used to normalise workspace volume)."""
        return float(TIP[0])

    def _project_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp box limits, wrap unlimited joints, project the wrist cone.

        Works for a single config (n,) or a batch (M, n).
        """
        unlimited = self.unlimited if self.unlimited is not None \
            else np.zeros(self.n, bool)
        q = np.where(unlimited,
                     (q + np.pi) % (2 * np.pi) - np.pi,
                     np.clip(q, self.lower, self.upper))
        if self.cone_idx is not None and self.cone_angle > 0:
            i, j = self.cone_idx
            qi = q[..., i]
            qj = q[..., j]
            r = np.hypot(qi, qj)
            scale = np.where(r > self.cone_angle,
                             self.cone_angle / np.maximum(r, 1e-9), 1.0)
            q[..., i] = qi * scale
            q[..., j] = qj * scale
        return q

    def sample_joints(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        q = rng.uniform(self.lower, self.upper, size=(n_samples, self.n))
        if self.cone_idx is not None and self.cone_angle > 0:
            # sample the two tilt DOFs uniformly inside the cone disk
            i, j = self.cone_idx
            r = self.cone_angle * np.sqrt(rng.uniform(0, 1, n_samples))
            phi = rng.uniform(0, 2 * np.pi, n_samples)
            q[:, i] = r * np.cos(phi)
            q[:, j] = r * np.sin(phi)
        return q

    # -- forward kinematics + geometric Jacobian, fully vectorised --------
    def fk_jacobian(self, q: np.ndarray):
        """Batched forward kinematics and geometric Jacobian.

        Parameters
        ----------
        q : (N, n) array of joint angles.

        Returns
        -------
        tip     : (N, 3) tool-tip position in the world frame.
        seg_pts : (N, S, 3) world positions of the segment joints
                  (shoulder, elbow, wrist) for drawing the arm.
        J       : (N, 6, n) geometric Jacobian, linear rows first.
        """
        tip, _, seg_pts, J = self.fk_full(q)
        return tip, seg_pts, J

    def fk_full(self, q: np.ndarray):
        """Batched FK returning tool position AND orientation.

        Returns
        -------
        tip     : (N, 3) tool-tip position in the world frame.
        R_tool  : (N, 3, 3) tool-frame orientation in the world frame.
        seg_pts : (N, S, 3) world positions of the segment joints.
        J       : (N, 6, n) geometric Jacobian, linear rows first.
        """
        q = np.atleast_2d(np.asarray(q, dtype=float))
        N = q.shape[0]
        n = self.n

        T = np.broadcast_to(self.base, (N, 4, 4)).copy()  # start at the mount
        zs = np.empty((N, n, 3))   # world axis of joint i
        ps = np.empty((N, n, 3))   # world point on axis of joint i

        for i in range(n):
            R_prev = T[:, :3, :3]
            zs[:, i, :] = R_prev @ self.omegas[i]
            ps[:, i, :] = R_prev @ self.points[i] + T[:, :3, 3]
            E = _joint_transform(self.omegas[i], self.points[i], q[:, i])
            T = T @ E

        R_tool = T[:, :3, :3]
        tip = (R_tool @ TIP) + T[:, :3, 3]

        J = np.empty((N, 6, n))
        diff = tip[:, None, :] - ps                      # (N, n, 3)
        J[:, :3, :] = np.cross(zs, diff).transpose(0, 2, 1)
        J[:, 3:, :] = zs.transpose(0, 2, 1)

        seg_pts = ps[:, self.segment_idx, :]             # (N, S, 3)
        return tip, R_tool, seg_pts, J

    # -- damped-least-squares position IK (for the interactive tool) ------
    def ik_position(self, target: np.ndarray, q0: np.ndarray | None = None,
                    iters: int = 300, damp: float = 0.06, tol: float = 2e-4):
        """Solve joints so the tool tip reaches `target` (position only).

        Returns (q, reached, err).  Unlimited joints wrap into [-pi, pi];
        limited joints are clamped to their travel range each iteration.
        """
        q = np.zeros(self.n) if q0 is None else np.array(q0, dtype=float)
        target = np.asarray(target, dtype=float)
        I3 = np.eye(3)
        err = np.inf
        for _ in range(iters):
            tip, _, J = self.fk_jacobian(q[None])
            e = target - tip[0]
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            Jv = J[0, :3, :]                          # (3, n)
            dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + damp**2 * I3, e)
            step = float(np.linalg.norm(dq))
            if step > 0.5:                            # cap the step for stability
                dq *= 0.5 / step
            q = self._project_limits(q + dq)
        return q, err < tol * 3, err

    # -- batched full 6-DOF pose IK (for task-space sampling) -------------
    def ik_pose_batch(self, target_pos: np.ndarray, target_R: np.ndarray,
                      q0: np.ndarray, iters: int = 80, damp: float = 0.08,
                      pos_tol: float = 5e-3, rot_tol: float = 0.05):
        """Vectorised damped-least-squares IK for full 6-DOF poses.

        Parameters
        ----------
        target_pos : (M, 3) desired tool positions.
        target_R   : (M, 3, 3) desired tool orientations.
        q0         : (M, n) initial joint configurations.

        Returns
        -------
        q          : (M, n) solved joint configurations.
        reached    : (M,) bool, position AND orientation within tolerance.
        pos_err    : (M,) final position error [m].
        rot_err    : (M,) final orientation error [rad].
        """
        q = np.array(q0, dtype=float)
        M = q.shape[0]
        I6 = np.eye(6)
        pos_err = np.full(M, np.inf)
        rot_err = np.full(M, np.inf)

        for _ in range(iters):
            tip, R_tool, _, J = self.fk_full(q)
            e_p = target_pos - tip                         # (M, 3)
            # orientation error: axis-angle of R_target @ R_current^T
            R_err = target_R @ R_tool.transpose(0, 2, 1)   # (M, 3, 3)
            e_o = _so3_log(R_err)                           # (M, 3)

            pos_err = np.linalg.norm(e_p, axis=1)
            rot_err = np.linalg.norm(e_o, axis=1)
            e = np.concatenate([e_p, e_o], axis=1)         # (M, 6)

            # dq = J^T (J J^T + damp^2 I)^-1 e  (batched)
            JJt = J @ J.transpose(0, 2, 1) + damp**2 * I6  # (M, 6, 6)
            y = np.linalg.solve(JJt, e[:, :, None])        # (M, 6, 1)
            dq = (J.transpose(0, 2, 1) @ y)[:, :, 0]       # (M, n)

            step = np.linalg.norm(dq, axis=1, keepdims=True)
            scale = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-9), 1.0)
            q = self._project_limits(q + dq * scale)

        reached = (pos_err < pos_tol) & (rot_err < rot_tol)
        return q, reached, pos_err, rot_err

    # -- batched position-only IK (for the precomputed interactive viewer)
    def ik_position_batch(self, target_pos: np.ndarray, q0: np.ndarray,
                          iters: int = 120, damp: float = 0.06,
                          tol: float = 3e-3):
        """Vectorised damped-least-squares position IK.

        Parameters
        ----------
        target_pos : (M, 3) desired tool positions.
        q0         : (M, n) initial joint configurations.

        Returns
        -------
        q       : (M, n) solved joint configurations.
        reached : (M,) bool, position within tolerance.
        pos_err : (M,) final position error [m].
        """
        q = np.array(q0, dtype=float)
        I3 = np.eye(3)
        pos_err = np.full(q.shape[0], np.inf)

        for _ in range(iters):
            tip, _, J = self.fk_jacobian(q)
            e = target_pos - tip                           # (M, 3)
            pos_err = np.linalg.norm(e, axis=1)
            Jv = J[:, :3, :]                               # (M, 3, n)
            JJt = Jv @ Jv.transpose(0, 2, 1) + damp**2 * I3
            y = np.linalg.solve(JJt, e[:, :, None])        # (M, 3, 1)
            dq = (Jv.transpose(0, 2, 1) @ y)[:, :, 0]      # (M, n)
            step = np.linalg.norm(dq, axis=1, keepdims=True)
            scale = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-9), 1.0)
            q = self._project_limits(q + dq * scale)

        reached = pos_err < tol
        return q, reached, pos_err



# ----------------------------------------------------------------------
# Screw / rotation helpers (vectorised over the sample dimension)
# ----------------------------------------------------------------------
def _so3_log(R: np.ndarray) -> np.ndarray:
    """Batched log map SO(3) -> R^3 (rotation vector).  (M,3,3) -> (M,3)."""
    tr = np.trace(R, axis1=1, axis2=2)
    cos_t = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)                              # (M,)
    vec = np.stack([R[:, 2, 1] - R[:, 1, 2],
                    R[:, 0, 2] - R[:, 2, 0],
                    R[:, 1, 0] - R[:, 0, 1]], axis=1)     # (M, 3)
    small = theta < 1e-6
    scale = np.where(small, 0.5, theta / (2.0 * np.sin(theta) + 1e-12))
    return vec * scale[:, None]


def _joint_transform(omega: np.ndarray, point: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """SE(3) of a revolute joint: Trans(p) . Rot(omega, theta) . Trans(-p)."""
    theta = np.atleast_1d(theta)
    N = theta.shape[0]
    c = np.cos(theta)[:, None, None]
    s = np.sin(theta)[:, None, None]
    K = _skew(omega)
    wwT = np.outer(omega, omega)
    eye = np.eye(3)

    R = c * eye + s * K + (1.0 - c) * wwT          # (N, 3, 3) Rodrigues
    T = np.zeros((N, 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = point - (R @ point)
    T[:, 3, 3] = 1.0
    return T


# ----------------------------------------------------------------------
# Robot factories
# ----------------------------------------------------------------------
def _deg(*vals: float) -> np.ndarray:
    return np.deg2rad(np.asarray(vals, dtype=float))


def make_srs() -> Robot:
    """7-DOF SRS serial arm, 3-1-3 grouping, finite travel on every axis.

    The three shoulder axes are distributed along a short module length, and
    the three wrist axes across a short wrist module, so the arm has real
    (short) serial links instead of coincident points.
    """
    x_sh0 = 0.0
    x_sh1 = D_SHOULDER * 0.5
    x_sh2 = D_SHOULDER
    x_elb = D_SHOULDER + L_UPPER
    x_wri = D_SHOULDER + L_UPPER + L_FORE
    x_wri2 = x_wri + D_WRIST

    omegas = np.array([
        _Z, _Y, _X,     # shoulder: yaw, pitch, humeral roll (spread along module)
        _Y,             # elbow: pitch
        _X, _Y, _Z,     # forearm roll, wrist pitch, wrist yaw
    ])
    points = np.array([
        [x_sh0, 0, 0], [x_sh1, 0, 0], [x_sh2, 0, 0],
        [x_elb, 0, 0],
        [x_elb, 0, 0], [x_wri, 0, 0], [x_wri2, 0, 0],
    ], dtype=float)
    # Real KUKA LBR iiwa 14 R820 joint travel limits
    # (A1..A7 = +/-170, 120, 170, 120, 170, 120, 175 deg).
    upper = _deg(170, 120, 170, 120, 170, 120, 175)
    lower = -upper
    return Robot(
        name="SRS (3-1-3, limited)",
        omegas=omegas, points=points,
        lower=lower, upper=upper,
        segment_idx=[0, 3, 5],
        unlimited=np.zeros(7, dtype=bool),
    )


def make_orbita() -> Robot:
    """Reachy 2 Orbita arm, 2-2-3 grouping, with REAL joint limits.

    Shoulder and elbow are compact Orbita2d actuators (2 intersecting axes
    each); the wrist is a compact Orbita3d spherical actuator.  A short wrist
    module separates the wrist centre from the tool tip.

    Joint travel is taken from the official Pollen Reachy 2 URDF
    (reachy2_symbolic_ik/config_files/reachy2.urdf) and cross-checked against
    the ByteDance ByteWrist paper (arXiv:2509.18084), which reports the same
    parallel-wrist architecture with a conical motion range:

      * shoulder  : two DOF, ~+/-90 deg travel each
      * elbow     : yaw +/-90 deg, pitch -129..+5.7 deg (anthropomorphic,
                    bends one way only)                 [URDF -2.25..0.1 rad]
      * wrist     : Orbita3d spherical actuator.  The two TILT DOF are NOT
                    two independent +/-180 axes -- they are limited to a CONE.
                    Reachy casts the wrist into a cone of half-angle 42.5 deg
                    (limit_orbita3d, orbita3D_max_angle = radians(42.5)); the
                    ByteWrist paper gives beta^2 + gamma^2 < 0.72 rad^2
                    (half-angle ~48.6 deg).  We use 45 deg, between the two.
                    The wrist roll about the arm axis stays free (multi-turn).
    """
    x_sh = 0.0
    x_elb = D_SHOULDER + L_UPPER
    x_wri = D_SHOULDER + L_UPPER + L_FORE

    omegas = np.array([
        _Y, _Z,         # shoulder Orbita2d: pitch, yaw
        _Y, _Z,         # elbow    Orbita2d: yaw, pitch
        _X, _Y, _Z,     # wrist    Orbita3d: roll (free), tilt-pitch, tilt-yaw
    ])
    points = np.array([
        [x_sh, 0, 0], [x_sh, 0, 0],
        [x_elb, 0, 0], [x_elb, 0, 0],
        [x_wri, 0, 0], [x_wri, 0, 0], [x_wri, 0, 0],
    ], dtype=float)

    #                shoulder      elbow        wrist
    #                pit   yaw     yaw   pit     roll  tiltY tiltZ
    upper = _deg(90,   90,     90,   5.7,    180,  45,   45)
    lower = _deg(-90, -90,    -90, -129.0,  -180, -45,  -45)

    unlimited = np.array([0, 0, 0, 0, 1, 0, 0], dtype=bool)  # only wrist roll
    return Robot(
        name="Orbita (2-2-3, real+cone)",
        omegas=omegas, points=points,
        lower=lower, upper=upper,
        segment_idx=[0, 2, 4],
        unlimited=unlimited,
        cone_idx=(5, 6),                 # wrist tilt-pitch / tilt-yaw
        cone_angle=np.deg2rad(45.0),     # half-cone (Reachy 42.5, ByteWrist 48.6)
    )



def manipulability(J: np.ndarray) -> np.ndarray:
    """Yoshikawa manipulability  w = sqrt(det(J J^T)).  Batched (N,6,n)->(N,)."""
    JJt = J @ J.transpose(0, 2, 1)
    det = np.linalg.det(JJt)
    return np.sqrt(np.clip(det, 0.0, None))


def singular_values(J: np.ndarray) -> np.ndarray:
    """Singular values of each Jacobian.  (N, 6, n) -> (N, 6) sorted desc."""
    return np.linalg.svd(J, compute_uv=False)
