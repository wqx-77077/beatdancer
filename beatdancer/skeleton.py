"""A 16-DOF servo humanoid: joint table, forward kinematics, limits.

The rig targets the kind of hobby humanoid you can actually build on site
(RDK X5 + serial-bus servos + 3D printed frame, ~45 cm tall). The joint names
are mapped to Unitree G1 names at the bottom so the same choreography can be
retargeted to a real humanoid later.

Axes: +X right, +Y forward, +Z up. Angles in radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

JOINTS = [
    "waist_yaw",
    "head_yaw",
    "shoulder_pitch_L", "shoulder_roll_L", "elbow_L",
    "shoulder_pitch_R", "shoulder_roll_R", "elbow_R",
    "hip_pitch_L", "hip_roll_L", "knee_L", "ankle_L",
    "hip_pitch_R", "hip_roll_R", "knee_R", "ankle_R",
]
N_JOINTS = len(JOINTS)
IDX = {name: i for i, name in enumerate(JOINTS)}

# (min, max) per joint, chosen so the motion never destroys a hobby servo
LIMITS = {
    "waist_yaw": (-0.9, 0.9),
    "head_yaw": (-1.0, 1.0),
    "shoulder_pitch_L": (-2.1, 2.1), "shoulder_pitch_R": (-2.1, 2.1),
    "shoulder_roll_L": (-0.25, 1.75), "shoulder_roll_R": (-1.75, 0.25),
    "elbow_L": (-2.2, 0.1), "elbow_R": (-2.2, 0.1),
    "hip_pitch_L": (-1.3, 1.1), "hip_pitch_R": (-1.3, 1.1),
    "hip_roll_L": (-0.45, 0.6), "hip_roll_R": (-0.6, 0.45),
    "knee_L": (-2.0, 0.05), "knee_R": (-2.0, 0.05),
    "ankle_L": (-0.7, 0.6), "ankle_R": (-0.7, 0.6),
}

# body dimensions in metres (a 45 cm desktop humanoid)
BONE = {
    "pelvis_z": 0.215,
    "hip_dx": 0.052,
    "thigh": 0.115,
    "shin": 0.110,
    "foot": 0.052,
    "torso": 0.150,
    "neck": 0.052,
    "head": 0.046,
    "shoulder_dx": 0.082,
    "shoulder_z": 0.130,   # above chest joint
    "upper_arm": 0.105,
    "forearm": 0.098,
}

G1_MAP = {
    "waist_yaw": ["waist_yaw_joint"],
    "head_yaw": ["head_yaw_joint"],
    "shoulder_pitch_L": ["left_shoulder_pitch_joint"],
    "shoulder_roll_L": ["left_shoulder_roll_joint"],
    "elbow_L": ["left_elbow_joint"],
    "shoulder_pitch_R": ["right_shoulder_pitch_joint"],
    "shoulder_roll_R": ["right_shoulder_roll_joint"],
    "elbow_R": ["right_elbow_joint"],
    "hip_pitch_L": ["left_hip_pitch_joint"],
    "hip_roll_L": ["left_hip_roll_joint"],
    "knee_L": ["left_knee_joint"],
    "ankle_L": ["left_ankle_pitch_joint"],
    "hip_pitch_R": ["right_hip_pitch_joint"],
    "hip_roll_R": ["right_hip_roll_joint"],
    "knee_R": ["right_knee_joint"],
    "ankle_R": ["right_ankle_pitch_joint"],
}


def rest_pose() -> np.ndarray:
    """A relaxed standing pose (arms slightly out, knees soft)."""
    q = np.zeros(N_JOINTS, dtype=np.float32)
    q[IDX["shoulder_roll_L"]] = 0.18
    q[IDX["shoulder_roll_R"]] = -0.18
    q[IDX["elbow_L"]] = -0.25
    q[IDX["elbow_R"]] = -0.25
    q[IDX["knee_L"]] = -0.08
    q[IDX["knee_R"]] = -0.08
    q[IDX["ankle_L"]] = 0.04
    q[IDX["ankle_R"]] = 0.04
    return q


def clamp(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float32, copy=True)
    for i, name in enumerate(JOINTS):
        lo, hi = LIMITS[name]
        out[i] = min(max(out[i], lo), hi)
    return out


# ------------------------------------------------------------------ rotations
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


@dataclass
class Frame:
    """World-space joints for one render frame."""
    points: dict[str, np.ndarray]
    root: np.ndarray
    yaw: float = 0.0


def fk(q: np.ndarray, root: np.ndarray | None = None,
       yaw: float = 0.0) -> Frame:
    """Forward kinematics for the 16-DOF rig."""
    b = BONE
    root = np.zeros(3) if root is None else np.asarray(root, dtype=np.float64)
    pelvis = root + np.array([0.0, 0.0, b["pelvis_z"]])

    Rz = rot_z(q[IDX["waist_yaw"]] + yaw)
    chest = pelvis + Rz @ np.array([0.0, 0.0, b["torso"]])
    neck = chest + Rz @ np.array([0.0, 0.0, b["neck"]])
    head = neck + Rz @ np.array([0.0, 0.0, b["head"]])

    pts: dict[str, np.ndarray] = {
        "pelvis": pelvis, "chest": chest, "neck": neck, "head": head,
    }

    # arms -------------------------------------------------------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh = chest + Rz @ np.array([sx * b["shoulder_dx"], 0.0, b["shoulder_z"]])
        pitch = q[IDX[f"shoulder_pitch_{side}"]]
        roll = q[IDX[f"shoulder_roll_{side}"]]
        elbow = q[IDX[f"elbow_{side}"]]

        Rarm = Rz @ rot_y(-sx * roll) @ rot_x(pitch)
        upper_dir = Rarm @ np.array([0.0, 0.0, -1.0])
        el = sh + upper_dir * b["upper_arm"]
        # elbow bends in the sagittal plane, always "inward"
        Rfore = Rarm @ rot_x(elbow)
        fore_dir = Rfore @ np.array([0.0, 0.0, -1.0])
        wr = el + fore_dir * b["forearm"]

        pts[f"shoulder_{side}"] = sh
        pts[f"elbow_{side}"] = el
        pts[f"wrist_{side}"] = wr

    # legs -------------------------------------------------------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        hip = pelvis + np.array([sx * b["hip_dx"], 0.0, 0.0])
        pitch = q[IDX[f"hip_pitch_{side}"]]
        roll = q[IDX[f"hip_roll_{side}"]]
        knee = q[IDX[f"knee_{side}"]]
        ankle = q[IDX[f"ankle_{side}"]]

        Rthigh = Rz @ rot_y(sx * roll) @ rot_x(pitch)
        thigh_dir = Rthigh @ np.array([0.0, 0.0, -1.0])
        kn = hip + thigh_dir * b["thigh"]

        Rshin = Rthigh @ rot_x(knee)
        shin_dir = Rshin @ np.array([0.0, 0.0, -1.0])
        an = kn + shin_dir * b["shin"]

        Rfoot = Rshin @ rot_x(ankle)
        foot_dir = Rfoot @ np.array([0.0, 1.0, -0.18])
        foot_dir = foot_dir / (np.linalg.norm(foot_dir) + 1e-9)
        toe = an + foot_dir * b["foot"] * 1.35

        pts[f"hip_{side}"] = hip
        pts[f"knee_{side}"] = kn
        pts[f"ankle_{side}"] = an
        pts[f"toe_{side}"] = toe

    return Frame(points=pts, root=pelvis.copy(), yaw=float(q[IDX["waist_yaw"]] + yaw))


BONES = [
    ("pelvis", "chest", 0.052, 0.046),
    ("chest", "neck", 0.030, 0.026),
    ("neck", "head", 0.030, 0.040),
    ("chest", "shoulder_L", 0.028, 0.024),
    ("shoulder_L", "elbow_L", 0.024, 0.020),
    ("elbow_L", "wrist_L", 0.020, 0.015),
    ("chest", "shoulder_R", 0.028, 0.024),
    ("shoulder_R", "elbow_R", 0.024, 0.020),
    ("elbow_R", "wrist_R", 0.020, 0.015),
    ("pelvis", "hip_L", 0.040, 0.034),
    ("hip_L", "knee_L", 0.034, 0.026),
    ("knee_L", "ankle_L", 0.026, 0.020),
    ("ankle_L", "toe_L", 0.020, 0.014),
    ("pelvis", "hip_R", 0.040, 0.034),
    ("hip_R", "knee_R", 0.034, 0.026),
    ("knee_R", "ankle_R", 0.026, 0.020),
    ("ankle_R", "toe_R", 0.020, 0.014),
]


def joint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute joint difference, the transition cost used by the composer."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def mirror(q: np.ndarray) -> np.ndarray:
    """Left/right mirrored version of a pose (used to vary a motif)."""
    m = np.array(q, dtype=np.float32, copy=True)
    for name in ("waist_yaw", "head_yaw"):
        m[IDX[name]] = -m[IDX[name]]
    swap = [
        ("shoulder_pitch_L", "shoulder_pitch_R"),
        ("elbow_L", "elbow_R"),
        ("hip_pitch_L", "hip_pitch_R"),
        ("knee_L", "knee_R"),
        ("ankle_L", "ankle_R"),
    ]
    for l, r in swap:
        m[IDX[l]], m[IDX[r]] = q[IDX[r]], q[IDX[l]]
    m[IDX["shoulder_roll_L"]] = -q[IDX["shoulder_roll_R"]]
    m[IDX["shoulder_roll_R"]] = -q[IDX["shoulder_roll_L"]]
    m[IDX["hip_roll_L"]] = -q[IDX["hip_roll_R"]]
    m[IDX["hip_roll_R"]] = -q[IDX["hip_roll_L"]]
    return m
