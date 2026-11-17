"""Exports: servo commands, BVH for Blender, Unitree G1 qpos, summary CSV.

Every format is a thin adapter over the same (T, 16) joint matrix, which is the
whole point of keeping the choreography in semantic joint space.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .choreo import Choreography
from .skeleton import G1_MAP, JOINTS, LIMITS, fk

# Feetech / Dynamixel style bus servos use 0..1000 (or 0..4095) ticks.
SERVO_TICKS = 1000
SERVO_CENTER = 500


def servo_csv(cho: Choreography, path: str, ticks: int = SERVO_TICKS,
              invert: dict[str, bool] | None = None) -> str:
    """Per-frame servo target in raw bus ticks, plus a header the driver reads."""
    invert = invert or {}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# beatdancer servo table", f"fps={cho.fps}", f"joints={len(JOINTS)}",
                    f"ticks={ticks}"])
        w.writerow(["t"] + JOINTS)
        for i, t in enumerate(cho.times):
            row = [f"{t:.3f}"]
            for j, name in enumerate(JOINTS):
                lo, hi = LIMITS[name]
                v = (cho.poses[i, j] - lo) / (hi - lo)          # 0..1
                if invert.get(name):
                    v = 1.0 - v
                row.append(str(int(round(v * (ticks - 1)))))
            w.writerow(row)
    return path


def servo_config(path: str) -> str:
    """Per-joint calibration file: id, centre offset, direction, limits."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for i, name in enumerate(JOINTS):
        rows.append({
            "joint": name, "servo_id": i + 1, "center": SERVO_CENTER,
            "offset": 0, "direction": -1 if "roll_R" in name or "yaw" in name else 1,
            "min_tick": 120, "max_tick": 880,
        })
    with open(path, "w") as f:
        json.dump({"bus": "/dev/ttyUSB0", "baud": 1000000, "protocol": "feetech_sts",
                   "joints": rows}, f, indent=2, ensure_ascii=False)
    return path


def bvh(cho: Choreography, path: str) -> str:
    """BVH export so the dance can be opened in Blender / MotionBuilder."""
    names = JOINTS
    order = ["waist_yaw", "head_yaw",
             "shoulder_pitch_L", "shoulder_roll_L", "elbow_L",
             "shoulder_pitch_R", "shoulder_roll_R", "elbow_R",
             "hip_pitch_L", "hip_roll_L", "knee_L", "ankle_L",
             "hip_pitch_R", "hip_roll_R", "knee_R", "ankle_R"]
    lines = ["HIERARCHY", "ROOT Hips", "{", "\tOFFSET 0 0 0", "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]
    # flat hierarchy: every joint hangs off Hips, offsets kept from the rest pose
    rest = fk(np.zeros(len(JOINTS), dtype=np.float32))
    for name in order:
        p = rest.points.get(name)
        if p is None:
            continue
        off = (p - rest.points["pelvis"]) * 1000.0
        lines += ["\tJOINT " + name, "\t{", f"\t\tOFFSET {off[0]:.3f} {off[1]:.3f} {off[2]:.3f}",
                  "\t\tCHANNELS 3 Zrotation Xrotation Yrotation", "\t}"]
    lines += ["}"]

    n = len(cho.times)
    lines += ["MOTION", f"Frames: {n}", f"Frame Time: {1.0 / cho.fps:.6f}"]
    for i in range(n):
        r = cho.root[i]
        vals = [r[0] * 1000, r[2] * 1000, r[1] * 1000, np.degrees(r[3]), 0, 0]
        for name in order:
            vals.append(np.degrees(cho.poses[i, JOINTS.index(name)]))
        lines.append(" ".join(f"{v:.4f}" for v in vals))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def g1_qpos(cho: Choreography, path: str) -> tuple[str, list[str]]:
    """Map our 16 joints onto Unitree G1 joint names (for mujoco / Isaac)."""
    cols, names = [], []
    for j, name in enumerate(JOINTS):
        cols.append(cho.poses[:, j])
        names.append(G1_MAP[name][0])
    data = np.stack(cols, axis=1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.column_stack([cho.times, data]), delimiter=",",
               header="time," + ",".join(names), comments="")
    return path, names


def summary_csv(cho: Choreography, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t0", "t1", "beats", "section", "move", "move_zh",
                    "motif", "mirrored", "energy"])
        for i, s in enumerate(cho.segments):
            w.writerow([i, f"{s.t0:.3f}", f"{s.t1:.3f}", s.n_beats, s.section,
                        s.name, s.zh, s.motif, int(s.mirrored), f"{s.energy:.2f}"])
    return path
