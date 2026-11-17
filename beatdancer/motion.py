"""The dance vocabulary: 18 procedural motion primitives.

Every primitive is a pure function phase -> joint vector, loopable
(q(0) == q(1)), with a musical length in beats and an intended energy.
This is the part that replaces "a human animator": instead of baking a
fixed dance, we describe *moves* and let the composer sequence them.

Why procedural and not mocap? (1) no dataset license to worry about,
(2) 16 numbers per frame is trivially streamable to hobby servos,
(3) we can retime a move to any tempo without foot sliding artefacts
that plague naive mocap playback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .skeleton import IDX, N_JOINTS, clamp as limit_clamp, rest_pose

TwoPi = 2.0 * math.pi


def _s(p: float, phase: float = 0.0) -> float:
    return math.sin(TwoPi * (p + phase))


def _c(p: float, phase: float = 0.0) -> float:
    return math.cos(TwoPi * (p + phase))


def _pulse(p: float, sharp: float = 3.0) -> float:
    """A decaying hit at phase ~0, used for staccato accents."""
    return math.exp(-sharp * ((p % 1.0))) * (1.0 - p % 1.0) ** 0.5


def _bounce(p: float, n: int = 2) -> float:
    """n bounces per cycle, 0..1, valley on the beat."""
    return 0.5 - 0.5 * math.cos(TwoPi * n * p)


@dataclass
class Primitive:
    name: str
    beats: float
    energy: float
    tags: tuple[str, ...]
    support: str                      # both | L | R | air
    fn: Callable[[float], np.ndarray]
    root: Callable[[float], tuple[float, float, float, float]] = field(
        default=lambda p: (0.0, 0.0, 0.0, 0.0))
    zh: str = ""

    def pose(self, p: float) -> np.ndarray:
        return limit_clamp(self.fn(p % 1.0))

    @property
    def entry(self) -> np.ndarray:
        return self.pose(0.0)

    @property
    def exit(self) -> np.ndarray:
        return self.pose(0.999)


def _base(**kw) -> np.ndarray:
    q = rest_pose()
    for k, v in kw.items():
        if k not in IDX:
            raise KeyError(k)
        q[IDX[k]] = v
    return q


def _with(q: np.ndarray, **kw) -> np.ndarray:
    out = q.copy()
    for k, v in kw.items():
        out[IDX[k]] = v
    return out


# ------------------------------------------------------------------ primitives
def _idle(p: float) -> np.ndarray:
    b = _bounce(p, 2)
    q = _base()
    q = _with(q, knee_L=-0.08 - 0.05 * b, knee_R=-0.08 - 0.05 * b,
              hip_pitch_L=0.03 * b, hip_pitch_R=0.03 * b,
              ankle_L=0.04 + 0.03 * b, ankle_R=0.04 + 0.03 * b,
              shoulder_pitch_L=0.05 * _s(p), shoulder_pitch_R=-0.05 * _s(p),
              waist_yaw=0.05 * _s(p, 0.25), head_yaw=0.12 * _s(p, 0.5))
    return q


def _step_touch(side: str) -> Callable[[float], np.ndarray]:
    sx = 1.0 if side == "L" else -1.0

    def f(p: float) -> np.ndarray:
        # weight on the "other" leg, tap on 2 and 4
        tap = max(0.0, math.sin(TwoPi * 2 * p))
        q = _base()
        q = _with(q,
                  hip_roll_L=0.10 * -sx * _s(p), hip_roll_R=0.10 * sx * _s(p),
                  hip_pitch_L=0.18 * tap * sx, hip_pitch_R=0.18 * tap * -sx,
                  knee_L=-0.12 - 0.30 * max(tap, 0) * sx,
                  knee_R=-0.12 - 0.30 * max(tap, 0) * -sx,
                  ankle_L=0.06, ankle_R=0.06,
                  shoulder_pitch_L=0.25 * _s(p) * sx, shoulder_pitch_R=-0.25 * _s(p) * sx,
                  elbow_L=-0.45 - 0.2 * _s(p), elbow_R=-0.45 + 0.2 * _s(p),
                  shoulder_roll_L=0.22 + 0.1 * sx, shoulder_roll_R=-0.22 + 0.1 * sx,
                  waist_yaw=0.16 * _s(p) * sx, head_yaw=0.10 * _s(p))
        return q
    return f


def _hip_sway(p: float) -> np.ndarray:
    s = _s(p)
    q = _base()
    q = _with(q, hip_roll_L=0.20 * s, hip_roll_R=0.20 * s,
              waist_yaw=-0.18 * s, head_yaw=0.14 * s,
              shoulder_roll_L=0.35 + 0.15 * s, shoulder_roll_R=-0.35 + 0.15 * s,
              shoulder_pitch_L=0.18 * _s(p, 0.25), shoulder_pitch_R=-0.18 * _s(p, 0.25),
              elbow_L=-0.7, elbow_R=-0.7,
              knee_L=-0.10 - 0.06 * abs(s), knee_R=-0.10 - 0.06 * abs(s),
              ankle_L=0.05, ankle_R=0.05)
    return q


def _arm_wave(side: str) -> Callable[[float], np.ndarray]:
    sx = 1.0 if side == "L" else -1.0

    def f(p: float) -> np.ndarray:
        q = _base()
        q = _with(q,
                  shoulder_roll_L=(1.55 if sx > 0 else 0.18),
                  shoulder_roll_R=(-1.55 if sx < 0 else -0.18),
                  shoulder_pitch_L=(-0.15 if sx > 0 else 0.1),
                  shoulder_pitch_R=(-0.15 if sx < 0 else 0.1),
                  elbow_L=(-0.35 + 0.75 * _s(p, 0.1)) if sx > 0 else -0.4,
                  elbow_R=(-0.35 + 0.75 * _s(p, 0.1)) if sx < 0 else -0.4,
                  waist_yaw=0.12 * sx * _s(p), head_yaw=0.18 * sx,
                  hip_roll_L=0.10 * sx, hip_roll_R=0.10 * sx,
                  knee_L=-0.12, knee_R=-0.12, ankle_L=0.05, ankle_R=0.05)
        return q
    return f


def _disco_point(side: str) -> Callable[[float], np.ndarray]:
    sx = 1.0 if side == "L" else -1.0

    def f(p: float) -> np.ndarray:
        hit = 0.5 + 0.5 * _c(p, 0.0)          # peaks on the beat
        q = _base()
        if sx > 0:
            q = _with(q, shoulder_pitch_R=1.55, shoulder_roll_R=-0.55, elbow_R=-0.05)
        else:
            q = _with(q, shoulder_pitch_L=1.55, shoulder_roll_L=0.55, elbow_L=-0.05)
        q = _with(q,
                  shoulder_roll_L=0.10 * sx, shoulder_roll_R=0.10 * sx,
                  elbow_L=-0.9 if sx > 0 else -0.05,
                  elbow_R=-0.9 if sx < 0 else -0.05,
                  hip_roll_L=0.16 * sx * hit, hip_roll_R=0.16 * sx * hit,
                  waist_yaw=-0.22 * sx, head_yaw=0.2 * sx,
                  knee_L=-0.14 - 0.1 * hit, knee_R=-0.14 - 0.1 * hit,
                  ankle_L=0.06, ankle_R=0.06)
        return q
    return f


def _clap(p: float) -> np.ndarray:
    hit = math.exp(-6.0 * (p % 0.5)) if (p % 0.5) < 0.3 else 0.0
    close = 0.55 + 0.45 * hit
    q = _base()
    q = _with(q,
              shoulder_pitch_L=1.15 * close, shoulder_pitch_R=1.15 * close,
              shoulder_roll_L=0.55 - 0.35 * hit, shoulder_roll_R=-0.55 + 0.35 * hit,
              elbow_L=-1.05 + 0.35 * hit, elbow_R=-1.05 + 0.35 * hit,
              waist_yaw=0.10 * _s(p, 0.5), head_yaw=0.08 * _s(p, 0.5),
              knee_L=-0.16 - 0.12 * hit, knee_R=-0.16 - 0.12 * hit,
              ankle_L=0.06, ankle_R=0.06)
    return q


def _body_roll(p: float) -> np.ndarray:
    q = _base()
    q = _with(q,
              waist_yaw=0.0,
              hip_roll_L=0.18 * _s(p), hip_roll_R=0.18 * _s(p),
              shoulder_roll_L=0.9 + 0.35 * _s(p, 0.25),
              shoulder_roll_R=-0.9 + 0.35 * _s(p, 0.25),
              shoulder_pitch_L=0.35 * _c(p), shoulder_pitch_R=0.35 * _c(p),
              elbow_L=-0.85, elbow_R=-0.85,
              knee_L=-0.10 - 0.08 * abs(_s(p)), knee_R=-0.10 - 0.08 * abs(_s(p)),
              ankle_L=0.05, ankle_R=0.05)
    return q


def _robot_arm(p: float) -> np.ndarray:
    # quantised into 4 poses per cycle: very "robot dance"
    step = int(p * 4) % 4
    poses = [
        dict(shoulder_pitch_L=1.5, elbow_L=0.0, shoulder_roll_L=0.15,
             shoulder_pitch_R=-0.2, elbow_R=-1.6, shoulder_roll_R=-0.2),
        dict(shoulder_pitch_L=0.2, elbow_L=-1.6, shoulder_roll_L=0.2,
             shoulder_pitch_R=1.5, elbow_R=0.0, shoulder_roll_R=-0.15),
        dict(shoulder_pitch_L=1.5, elbow_L=-1.6, shoulder_roll_L=1.0,
             shoulder_pitch_R=1.5, elbow_R=-1.6, shoulder_roll_R=-1.0),
        dict(shoulder_pitch_L=-0.1, elbow_L=-0.2, shoulder_roll_L=0.9,
             shoulder_pitch_R=-0.1, elbow_R=-0.2, shoulder_roll_R=-0.9),
    ]
    q = _base()
    q = _with(q, **poses[step],
              waist_yaw=0.25 * (1 if step in (1, 2) else -1),
              head_yaw=0.3 * (1 if step % 2 else -1),
              knee_L=-0.12, knee_R=-0.12, ankle_L=0.05, ankle_R=0.05)
    return q


def _spin(p: float) -> np.ndarray:
    q = _base()
    lift = 0.25 * math.sin(math.pi * p)
    q = _with(q,
              shoulder_roll_L=1.45, shoulder_roll_R=-1.45,
              shoulder_pitch_L=0.15, shoulder_pitch_R=0.15,
              elbow_L=-0.15, elbow_R=-0.15,
              hip_roll_L=-0.12 * _s(p, 0.25), hip_roll_R=-0.12 * _s(p, 0.25),
              knee_L=-0.18 - 0.15 * lift, knee_R=-0.10,
              ankle_L=0.06, ankle_R=0.06,
              head_yaw=0.25)
    return q


def _jump(p: float) -> np.ndarray:
    crouch = (math.exp(-8.0 * max(0.0, p - 0.55) ** 2) * (1.0 - p)) if p > 0.5 else 0.0
    air = max(0.0, math.sin(math.pi * min(1.0, p / 0.55))) if p < 0.55 else 0.0
    q = _base()
    q = _with(q,
              knee_L=-0.15 - 0.85 * crouch, knee_R=-0.15 - 0.85 * crouch,
              hip_pitch_L=0.25 * crouch, hip_pitch_R=0.25 * crouch,
              ankle_L=0.08 + 0.25 * crouch, ankle_R=0.08 + 0.25 * crouch,
              shoulder_pitch_L=-0.3 - 1.1 * air, shoulder_pitch_R=-0.3 - 1.1 * air,
              shoulder_roll_L=0.3 + 0.5 * air, shoulder_roll_R=-0.3 - 0.5 * air,
              elbow_L=-0.4, elbow_R=-0.4,
              waist_yaw=0.0, head_yaw=0.0)
    return q


def _jump_root(p: float) -> tuple[float, float, float, float]:
    air = max(0.0, math.sin(math.pi * min(1.0, p / 0.55))) if p < 0.55 else 0.0
    crouch = (math.exp(-8.0 * max(0.0, p - 0.55) ** 2) * (1.0 - p)) if p > 0.5 else 0.0
    return (0.0, 0.0, 0.055 * air - 0.045 * crouch, 0.0)


def _squat_pulse(p: float) -> np.ndarray:
    d = 0.5 + 0.5 * _c(p, 0.0)
    q = _base()
    q = _with(q,
              knee_L=-0.12 - 0.55 * d, knee_R=-0.12 - 0.55 * d,
              hip_pitch_L=0.55 * d, hip_pitch_R=0.55 * d,
              ankle_L=0.06 + 0.30 * d, ankle_R=0.06 + 0.30 * d,
              shoulder_pitch_L=0.9 * d - 0.3, shoulder_pitch_R=0.9 * d - 0.3,
              shoulder_roll_L=0.35, shoulder_roll_R=-0.35,
              elbow_L=-0.8, elbow_R=-0.8,
              waist_yaw=0.0, head_yaw=0.0)
    return q


def _squat_root(p: float) -> tuple[float, float, float, float]:
    d = 0.5 + 0.5 * _c(p, 0.0)
    return (0.0, 0.0, -0.035 * d, 0.0)


def _kick(side: str) -> Callable[[float], np.ndarray]:
    sx = 1.0 if side == "L" else -1.0

    def f(p: float) -> np.ndarray:
        kick = max(0.0, math.sin(math.pi * min(1.0, p / 0.6))) if p < 0.6 else 0.0
        q = _base()
        q = _with(q,
                  hip_pitch_L=1.15 * kick if sx > 0 else 0.0,
                  hip_pitch_R=1.15 * kick if sx < 0 else 0.0,
                  knee_L=-0.1 - 0.55 * kick if sx > 0 else -0.1,
                  knee_R=-0.1 - 0.55 * kick if sx < 0 else -0.1,
                  ankle_L=0.1 + 0.3 * kick if sx > 0 else 0.06,
                  ankle_R=0.1 + 0.3 * kick if sx < 0 else 0.06,
                  hip_roll_L=0.08 * sx, hip_roll_R=0.08 * sx,
                  shoulder_pitch_L=-0.5 * kick - 0.1, shoulder_pitch_R=-0.5 * kick - 0.1,
                  shoulder_roll_L=0.5, shoulder_roll_R=-0.5,
                  elbow_L=-0.6, elbow_R=-0.6,
                  waist_yaw=-0.12 * sx, head_yaw=0.1 * sx)
        return q
    return f


def _cross_step(p: float) -> np.ndarray:
    s = _s(p)
    q = _base()
    q = _with(q,
              hip_roll_L=0.22 * s, hip_roll_R=-0.10 * s,
              hip_pitch_L=0.25 * abs(s), hip_pitch_R=-0.15 * abs(s),
              knee_L=-0.15 - 0.25 * max(0.0, s), knee_R=-0.15 - 0.25 * max(0.0, -s),
              ankle_L=0.06, ankle_R=0.06,
              shoulder_pitch_L=-0.55 * s, shoulder_pitch_R=0.55 * s,
              shoulder_roll_L=0.4, shoulder_roll_R=-0.4,
              elbow_L=-1.0, elbow_R=-1.0,
              waist_yaw=0.22 * s, head_yaw=-0.1 * s)
    return q


def _wave_both(p: float) -> np.ndarray:
    w = _s(p, 0.25)
    q = _base()
    q = _with(q,
              shoulder_roll_L=1.45 + 0.12 * w, shoulder_roll_R=-1.45 + 0.12 * w,
              shoulder_pitch_L=0.1, shoulder_pitch_R=0.1,
              elbow_L=-0.5 + 0.55 * w, elbow_R=-0.5 - 0.55 * w,
              waist_yaw=0.0, head_yaw=0.22 * w,
              knee_L=-0.12, knee_R=-0.12, ankle_L=0.05, ankle_R=0.05)
    return q


def _heart_pose(p: float) -> np.ndarray:
    a = 0.5 - 0.5 * _c(p)          # slow in-out
    q = _base()
    q = _with(q,
              shoulder_pitch_L=1.35, shoulder_pitch_R=1.35,
              shoulder_roll_L=0.95, shoulder_roll_R=-0.95,
              elbow_L=-1.5 - 0.2 * a, elbow_R=-1.5 - 0.2 * a,
              waist_yaw=0.05 * _s(p), head_yaw=-0.12,
              knee_L=-0.10, knee_R=-0.10, ankle_L=0.05, ankle_R=0.05)
    return q


def _head_bob(p: float) -> np.ndarray:
    b = _bounce(p, 2)
    q = _base()
    q = _with(q,
              knee_L=-0.10 - 0.10 * b, knee_R=-0.10 - 0.10 * b,
              hip_pitch_L=0.05 * b, hip_pitch_R=0.05 * b,
              ankle_L=0.05, ankle_R=0.05,
              head_yaw=0.22 * _s(p), waist_yaw=0.08 * _s(p),
              shoulder_pitch_L=0.12 * _s(p, 0.25), shoulder_pitch_R=-0.12 * _s(p, 0.25),
              elbow_L=-0.5, elbow_R=-0.5,
              shoulder_roll_L=0.2, shoulder_roll_R=-0.2)
    return q


def _groove_low(p: float) -> np.ndarray:
    b = _bounce(p, 2)
    q = _base()
    q = _with(q,
              knee_L=-0.55 - 0.25 * b, knee_R=-0.55 - 0.25 * b,
              hip_pitch_L=0.35 + 0.15 * b, hip_pitch_R=0.35 + 0.15 * b,
              ankle_L=0.28, ankle_R=0.28,
              hip_roll_L=0.12 * _s(p), hip_roll_R=0.12 * _s(p),
              shoulder_pitch_L=0.55, shoulder_pitch_R=-0.35,
              shoulder_roll_L=0.45, shoulder_roll_R=-0.45,
              elbow_L=-1.25, elbow_R=-1.0,
              waist_yaw=0.18 * _s(p, 0.25), head_yaw=0.15 * _s(p, 0.25))
    return q


def _groove_low_root(p: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, -0.05 - 0.01 * _bounce(p, 2), 0.0)


def _spin_root(p: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, 0.012 * math.sin(math.pi * p), TwoPi * p)


LIBRARY: list[Primitive] = [
    Primitive("idle", 2, 0.10, ("calm",), "both", _idle, zh="待机律动"),
    Primitive("head_bob", 2, 0.18, ("calm", "groove"), "both", _head_bob, zh="点头"),
    Primitive("step_touch_L", 2, 0.32, ("light", "foot"), "R", _step_touch("L"), zh="左侧点步"),
    Primitive("step_touch_R", 2, 0.32, ("light", "foot"), "L", _step_touch("R"), zh="右侧点步"),
    Primitive("hip_sway", 4, 0.42, ("groove",), "both", _hip_sway, zh="摆胯"),
    Primitive("cross_step", 4, 0.46, ("groove", "foot"), "both", _cross_step, zh="交叉步"),
    Primitive("body_roll", 4, 0.50, ("groove", "wave"), "both", _body_roll, zh="身体波浪"),
    Primitive("arm_wave_L", 4, 0.40, ("wave", "arm"), "both", _arm_wave("L"), zh="左手挥手"),
    Primitive("arm_wave_R", 4, 0.40, ("wave", "arm"), "both", _arm_wave("R"), zh="右手挥手"),
    Primitive("wave_both", 4, 0.52, ("wave", "arm"), "both", _wave_both, zh="双手挥舞"),
    Primitive("clap", 2, 0.58, ("accent", "arm"), "both", _clap, zh="拍手"),
    Primitive("robot_arm", 4, 0.56, ("robot", "accent"), "both", _robot_arm, zh="机械臂"),
    Primitive("squat_pulse", 4, 0.62, ("power",), "both", _squat_pulse, _squat_root, zh="深蹲律动"),
    Primitive("groove_low", 4, 0.45, ("groove", "power"), "both", _groove_low, _groove_low_root, zh="低重心律动"),
    Primitive("kick_L", 4, 0.72, ("power", "foot"), "R", _kick("L"), zh="左踢腿"),
    Primitive("kick_R", 4, 0.72, ("power", "foot"), "L", _kick("R"), zh="右踢腿"),
    Primitive("disco_point_L", 2, 0.74, ("accent", "star"), "both", _disco_point("L"), zh="迪斯科指天(左)"),
    Primitive("disco_point_R", 2, 0.74, ("accent", "star"), "both", _disco_point("R"), zh="迪斯科指天(右)"),
    Primitive("jump", 2, 0.95, ("power", "air"), "air", _jump, _jump_root, zh="跳跃"),
    Primitive("spin", 8, 0.88, ("star", "air"), "both", _spin, _spin_root, zh="转体"),
    Primitive("heart_pose", 8, 0.26, ("calm", "wave"), "both", _heart_pose, zh="比心"),
]

BY_NAME = {p.name: p for p in LIBRARY}

# moves you can mirror without changing their meaning; the composer uses this
MIRRORABLE = {"step_touch_L", "step_touch_R", "arm_wave_L", "arm_wave_R",
              "disco_point_L", "disco_point_R", "kick_L", "kick_R"}
MIRROR_PAIRS = {
    "step_touch_L": "step_touch_R", "step_touch_R": "step_touch_L",
    "arm_wave_L": "arm_wave_R", "arm_wave_R": "arm_wave_L",
    "disco_point_L": "disco_point_R", "disco_point_R": "disco_point_L",
    "kick_L": "kick_R", "kick_R": "kick_L",
}


def family(name: str) -> str:
    """Group moves that feel alike, so a motif stays coherent."""
    p = BY_NAME[name]
    return p.tags[0]
