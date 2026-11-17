"""Play a BeatDancer trajectory on a Unitree G1 in MuJoCo (kinematic).

We deliberately do *not* run a balance controller here: the point is to see the
choreography on a real humanoid mesh before the hackathon, not to claim sim-to-
real. Joints are driven to the exported angles every step (`mj_forward` only),
which is exactly the motion the board will produce.

    python sim/g1_playback.py --csv dist/g1_qpos.csv --model unitree_g1/scene.xml

Get the model first:
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie
Then point --model at mujoco_menagerie/unitree_g1/scene.xml (BSD-3, see its
own LICENSE file).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np


def load_qpos(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(path) as f:
        header = f.readline().strip().lstrip("#").strip()
        names = header.split(",")[1:]
        rows = [r for r in csv.reader(f) if r]
    data = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    return data[:, 0], data[:, 1:], names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dist/g1_qpos.csv")
    ap.add_argument("--model", required=True, help="path to unitree_g1/scene.xml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    t, q, names = load_qpos(args.csv)
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    qadr = {}
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            qadr[n] = model.jnt_qposadr[jid]
    print(f"[sim] {len(qadr)}/{len(names)} joints mapped, {len(t)} frames")

    # G1 free joint: park the root at a sane height
    root = np.array([0.0, 0.0, 0.75])
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[:3] = root

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = 0
        t0 = time.perf_counter()
        while viewer.is_running():
            target = t[i] / max(args.speed, 1e-3)
            while time.perf_counter() - t0 < target:
                time.sleep(0.001)
            for k, n in enumerate(names):
                if n in qadr:
                    data.qpos[qadr[n]] = q[i, k]
            mujoco.mj_forward(model, data)
            viewer.sync()
            i += 1
            if i >= len(t):
                if not args.loop:
                    break
                i = 0
                t0 = time.perf_counter()
    return 0


if __name__ == "__main__":
    sys.exit(main())
