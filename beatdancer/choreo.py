"""The composer: turn a beat grid + sections into a dancing timeline.

Approach (this is the core idea of the project):

1. Cut the track into bars on the downbeat grid.
2. Give every bar a target intensity from the music (section label + beat energy).
3. For each *phrase* (2 bars for verses, 4 bars for choruses) pick a sequence of
   moves with dynamic programming. Cost = |move energy - target| + transition
   distance + repetition penalty; bonus for accents landing on strong beats.
4. Reuse the phrase motif across the section, mirrored every other phrase, so the
   dance has a recognisable structure instead of feeling random.
5. Sample the chosen moves at 50 Hz, cross-fade the seams, and rate-limit joints
   so a 25 kg*cm hobby servo is never asked to teleport.

The result is the same "motion graph + beat snapping" idea used by game
animation systems and dance-generation papers (e.g. EDGE, AIST++), but small
enough to run in a few hundred ms on a laptop and stream to servos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .audio import Analysis
from .motion import BY_NAME, LIBRARY, MIRROR_PAIRS, Primitive, family
from .skeleton import IDX, N_JOINTS, mirror

FPS = 50
VEL_LIMIT = 0.22          # rad per frame (~11 rad/s) - protects the servos

# how intense each section should feel
SECTION_TARGET = {
    "intro": 0.22,
    "verse": 0.45,
    "build": 0.62,
    "chorus": 0.82,
    "break": 0.30,
    "outro": 0.25,
    "main": 0.55,
}

# moves that look good on the first beat of a phrase
ACCENT = {"clap", "disco_point_L", "disco_point_R", "jump", "kick_L", "kick_R", "squat_pulse"}


@dataclass
class Segment:
    start_beat: int
    n_beats: int
    name: str
    mirrored: bool
    t0: float
    t1: float
    section: str
    motif: str = "A"
    energy: float = 0.5

    @property
    def zh(self) -> str:
        return BY_NAME[self.name].zh

    def to_dict(self) -> dict:
        return {
            "start_beat": self.start_beat, "n_beats": self.n_beats,
            "name": self.name, "zh": self.zh, "mirrored": self.mirrored,
            "t0": round(self.t0, 3), "t1": round(self.t1, 3),
            "section": self.section, "motif": self.motif,
        }


@dataclass
class Choreography:
    analysis: Analysis
    segments: list[Segment]
    fps: int
    times: np.ndarray
    poses: np.ndarray          # (T, N_JOINTS)
    root: np.ndarray           # (T, 4) -> x, y, z, yaw
    meta: dict = field(default_factory=dict)

    def to_dict(self, max_frames: int | None = None) -> dict:
        n = len(self.times)
        step = 1
        if max_frames and n > max_frames:
            step = int(math.ceil(n / max_frames))
        q = self.poses[::step]
        r = self.root[::step]
        t = self.times[::step]
        return {
            "meta": {
                **self.meta,
                "fps": self.fps / step,
                "bpm": round(float(self.analysis.bpm), 2),
                "duration": round(float(self.analysis.duration), 3),
                "n_frames": len(t),
            },
            "joints": _joint_names(),
            "times": [round(float(x), 4) for x in t],
            "q": [[round(float(v), 4) for v in row] for row in q],
            "root": [[round(float(v), 4) for v in row] for row in r],
            "segments": [s.to_dict() for s in self.segments],
            "sections": [
                {"start_beat": a, "end_beat": b, "label": lab}
                for a, b, lab in self.analysis.sections
            ],
            "beats": [round(float(x), 4) for x in self.analysis.beat_times],
            "downbeat": int(self.analysis.downbeat),
        }


def _joint_names() -> list[str]:
    from .skeleton import JOINTS
    return list(JOINTS)


# --------------------------------------------------------------------- planning
def _bars(analysis: Analysis) -> list[tuple[int, int]]:
    """Full bars as (start_beat, end_beat), aligned to the first downbeat."""
    meter = analysis.meter
    start = analysis.downbeat % meter
    if start < 0:
        start = 0
    bars = []
    b = 0 if start == 0 else start
    while b + meter <= analysis.n_beats:
        bars.append((b, b + meter))
        b += meter
    return bars


def _section_of(analysis: Analysis, beat: int) -> str:
    for a, b, label in analysis.sections:
        if a <= beat < b:
            return label
    return "main"


def _target_energy(analysis: Analysis, beat_a: int, beat_b: int) -> float:
    label = _section_of(analysis, beat_a)
    base = SECTION_TARGET.get(label, 0.55)
    if analysis.beat_energy is not None and len(analysis.beat_energy):
        lo = min(max(beat_a, 0), len(analysis.beat_energy) - 1)
        hi = min(max(beat_b, lo + 1), len(analysis.beat_energy))
        e = float(analysis.beat_energy[lo:hi].mean())
        base = 0.55 * base + 0.45 * (0.2 + 0.8 * e) * (base / 0.55)
    return float(np.clip(base, 0.05, 0.98))


def _transition(prev: Primitive, nxt: Primitive) -> float:
    return float(np.abs(prev.exit - nxt.entry).mean())


def compose_phrase(target_beats: int, target_energy: float, prev: Primitive | None,
                   rng: np.random.Generator, section: str,
                   allow_air: bool = True) -> list[Primitive]:
    """Dynamic programming over one phrase (exact, memoised).

    State = (beats filled so far, previous move). Transitions add the energy
    mismatch and the seam cost, so the cheapest path is a phrase whose moves
    match the music and flow into each other.
    """
    cands = []
    for p in LIBRARY:
        if p.beats > target_beats:
            continue
        if p.support == "air" and not allow_air:
            continue
        if p.beats == 8 and target_beats < 16:
            continue
        cands.append(p)

    def cost(p: Primitive, pos: int, prev_p: Primitive | None) -> float:
        c = 3.1 * abs(p.energy - target_energy)
        if prev_p is not None:
            c += 1.05 * _transition(prev_p, p)
            if prev_p.name == p.name:
                c += 0.9                      # don't repeat back to back
            elif family(prev_p.name) == family(p.name):
                c += 0.12                     # a little sameness is good
        if pos % 4 != 0 and p.name in ACCENT:
            c += 0.35                         # accents belong on the strong beat
        if pos % 4 == 0 and p.name in ACCENT:
            c -= 0.18
        if p.name == "spin" and target_energy < 0.6:
            c += 0.5
        if p.name == "jump" and target_energy < 0.7:
            c += 0.6
        if section in ("intro", "outro") and p.energy > 0.6:
            c += 0.5
        return c

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(pos: int, prev_name: str) -> tuple[float, tuple[str, ...]]:
        prev_p = BY_NAME[prev_name] if prev_name else None
        if pos == target_beats:
            return 0.0, ()
        best = (float("inf"), ())
        for p in cands:
            if pos + p.beats > target_beats:
                continue
            c = cost(p, pos, prev_p)
            rest_c, rest_seq = solve(pos + p.beats, p.name)
            if not np.isfinite(rest_c):
                continue
            if c + rest_c < best[0]:
                best = (c + rest_c, (p.name,) + rest_seq)
        return best

    _, seq = solve(0, prev.name if prev else "")
    if not seq:
        return [BY_NAME["idle"]]
    return [BY_NAME[n] for n in seq]


def _alternatives(name: str, energy: float, banned: set[str],
                  beats: int | None = None) -> list[str]:
    """Moves with the same musical length and a similar intensity."""
    out = []
    fam = family(name)
    want = beats if beats is not None else BY_NAME[name].beats
    for p in LIBRARY:
        if p.name == name or p.name in banned or p.beats != want:
            continue
        if family(p.name) == fam and abs(p.energy - energy) < 0.28:
            out.append(p.name)
    for p in LIBRARY:
        if p.name == name or p.name in banned or p.beats != want:
            continue
        if abs(p.energy - energy) < 0.15 and p.name not in out:
            out.append(p.name)
    return out


def plan(analysis: Analysis, seed: int = 0, phrase_bars: int = 2) -> list[Segment]:
    """Compose the whole track: motif per section + call/answer variation."""
    rng = np.random.default_rng(seed)
    bars = _bars(analysis)
    if not bars:
        return []

    segments: list[Segment] = []
    motif_cache: dict[str, list[Primitive]] = {}
    phrase_no: dict[str, int] = {}
    prev_name: str | None = None

    i = 0
    while i < len(bars):
        sec = _section_of(analysis, bars[i][0])
        j = min(len(bars), i + phrase_bars)
        while j > i and _section_of(analysis, bars[j - 1][0]) != sec:
            j -= 1
        if j <= i:
            i += 1
            continue

        beats_here = bars[j - 1][1] - bars[i][0]
        energy = _target_energy(analysis, bars[i][0], bars[j - 1][1])
        k = phrase_no.get(sec, 0)
        phrase_no[sec] = k + 1

        if sec not in motif_cache:
            seq = compose_phrase(beats_here, energy, BY_NAME.get(prev_name) if prev_name else None,
                                 rng, sec, allow_air=energy > 0.55)
            motif_cache[sec] = list(seq)
            motif = chr(ord("A") + len(motif_cache) - 1)
        else:
            base = motif_cache[sec]
            seq = list(base)
            if k % 2 == 1:   # answer phrase: mirror the motif
                seq = [BY_NAME[MIRROR_PAIRS.get(p.name, p.name)] for p in seq]
            if rng.random() < 0.55 and seq:   # swap one move for a cousin
                pos = int(rng.integers(0, len(seq)))
                alts = _alternatives(seq[pos].name, seq[pos].energy,
                                     {x.name for x in seq}, beats=int(seq[pos].beats))
                if alts:
                    seq[pos] = BY_NAME[alts[int(rng.integers(0, len(alts)))]]
            motif = chr(ord("A") + list(motif_cache).index(sec))

        # never repeat the same move across the phrase seam
        if prev_name and seq and seq[0].name == prev_name:
            alts = _alternatives(prev_name, seq[0].energy,
                                 {x.name for x in seq}, beats=int(seq[0].beats))
            if alts:
                seq[0] = BY_NAME[alts[0]]

        b = bars[i][0]
        placed = 0
        for p in seq:
            if b + p.beats > bars[j - 1][1]:
                break
            segments.append(Segment(start_beat=b, n_beats=int(p.beats), name=p.name,
                                    mirrored=k % 2 == 1, t0=0.0, t1=0.0, section=sec,
                                    motif=motif, energy=float(p.energy)))
            b += p.beats
            placed += 1
        if placed:
            prev_name = seq[placed - 1].name
        i = j

    # intro pickup before the first bar line
    if segments and segments[0].start_beat >= 2:
        segments.insert(0, Segment(0, segments[0].start_beat, "idle", False, 0, 0,
                                   "intro", "A", 0.1))

    # star moves are rationed: at most one per ~6 bars, otherwise it stops reading
    # as a special moment
    last_star = -999
    for idx, s in enumerate(segments):
        if s.name in ("spin", "jump"):
            if s.start_beat - last_star < 24:
                banned = {x.name for x in segments[max(0, idx - 2): idx + 2]}
                alts = _alternatives(s.name, s.energy, banned, beats=s.n_beats)
                if alts:
                    s.name = alts[0]
                continue
            last_star = s.start_beat

    # three moves of the same family in a row reads as a loop; vary the middle
    for idx in range(1, len(segments) - 1):
        a, b, c = segments[idx - 1], segments[idx], segments[idx + 1]
        if family(a.name) == family(b.name) == family(c.name) == family(
                segments[idx].name) and c.n_beats == b.n_beats:
            alts = _alternatives(b.name, b.energy,
                                 {x.name for x in segments[max(0, idx - 3): idx + 3]},
                                 beats=b.n_beats)
            if alts:
                b.name = alts[0]

    # no holes: if a phrase could not be filled exactly, stretch the last move
    for idx, s in enumerate(segments):
        if idx + 1 < len(segments) and segments[idx + 1].start_beat != s.start_beat + s.n_beats:
            gap = segments[idx + 1].start_beat - (s.start_beat + s.n_beats)
            if 0 < gap <= 4:
                s.n_beats += gap

    bt = analysis.beat_times
    for s in segments:
        s.t0 = float(bt[min(s.start_beat, len(bt) - 1)])
        end_beat = min(s.start_beat + s.n_beats, len(bt))
        s.t1 = float(bt[end_beat]) if end_beat < len(bt) else float(analysis.duration)
    return segments


# --------------------------------------------------------------------- sampling
def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def sample(analysis: Analysis, segments: list[Segment], fps: int = FPS,
           blend: float = 0.09) -> Choreography:
    """Render the plan into a 50 Hz joint trajectory."""
    dur = float(analysis.duration)
    n = int(dur * fps) + 1
    times = np.arange(n) / fps
    poses = np.zeros((n, N_JOINTS), dtype=np.float32)
    root = np.zeros((n, 4), dtype=np.float32)

    if not segments:
        return Choreography(analysis, [], fps, times, poses, root)

    # yaw accumulates across segments (spins add up)
    yaw_at = []
    yaw = 0.0
    for s in segments:
        yaw_at.append(yaw)
        prim = BY_NAME[s.name]
        _, _, _, dyaw = prim.root(0.999)
        yaw += dyaw

    for k, s in enumerate(segments):
        prim = BY_NAME[s.name]
        span = max(s.t1 - s.t0, 1e-3)
        i0 = int(round(s.t0 * fps))
        i1 = int(round(s.t1 * fps))
        i0 = max(0, min(i0, n - 1))
        i1 = max(i0 + 1, min(i1, n))
        for i in range(i0, i1):
            p = (times[i] - s.t0) / span
            p = min(max(p, 0.0), 0.999)
            q = prim.pose(p)
            if s.mirrored:
                q = mirror(q)
            poses[i] = q
            rx, ry, rz, dyaw = prim.root(p)
            if s.mirrored:
                rx, dyaw = -rx, -dyaw
            root[i] = (rx, ry, rz, yaw_at[k] + dyaw)

    # --- blend seams: for the first `blend` seconds of each segment, fade from
    # the previous segment's last pose. Primitives loop, so holding the previous
    # exit pose is the right thing to fade from.
    bw = max(1, int(round(blend * fps)))
    for k in range(1, len(segments)):
        i0 = int(round(segments[k].t0 * fps))
        if i0 <= 0 or i0 >= n:
            continue
        prev_exit = poses[max(0, i0 - 1)].copy()
        prev_root = root[max(0, i0 - 1)].copy()
        for j in range(min(bw, n - i0)):
            w = _smoothstep(j / bw)
            poses[i0 + j] = (1 - w) * prev_exit + w * poses[i0 + j]
            root[i0 + j] = (1 - w) * prev_root + w * root[i0 + j]

    # --- servo protection: limit angular velocity
    for i in range(1, n):
        d = poses[i] - poses[i - 1]
        np.clip(d, -VEL_LIMIT, VEL_LIMIT, out=d)
        poses[i] = poses[i - 1] + d

    # --- foot planting: never let a foot sink through the floor. Leg bends
    # should lift the body, not push the robot into the stage.
    root[:, 2] += _foot_lift(poses)

    # tiny low-pass to remove the 50 Hz stair-stepping
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    pad = np.pad(poses, ((1, 1), (0, 0)), mode="edge")
    poses = (pad[:-2] * k[0] + pad[1:-1] * k[1] + pad[2:] * k[2]).astype(np.float32)

    return Choreography(analysis, segments, fps, times, poses, root,
                        meta={"engine": "beatdancer-dp", "seed": 0})


def _foot_lift(poses: np.ndarray) -> np.ndarray:
    """Vertical offset that keeps the lowest foot on z=0."""
    from .skeleton import fk

    n = len(poses)
    lift = np.zeros(n, dtype=np.float32)
    for i in range(n):
        fr = fk(poses[i])
        zs = [fr.points[k][2] for k in ("toe_L", "toe_R", "ankle_L", "ankle_R")]
        m = min(zs)
        lift[i] = max(0.0, -m)
    if n > 3:
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        pad = np.pad(lift, 1, mode="edge")
        lift = pad[:-2] * k[0] + pad[1:-1] * k[1] + pad[2:] * k[2]
    return lift


def make_choreography(analysis: Analysis, seed: int = 0) -> Choreography:
    segs = plan(analysis, seed=seed)
    cho = sample(analysis, segs)
    cho.meta["seed"] = seed
    cho.meta["n_segments"] = len(segs)
    cho.meta["labels"] = sorted({s.section for s in segs})
    return cho
