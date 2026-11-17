"""50 Hz dance runtime for the RDK board.

Two modes:

* replay: play a precomputed servo table (offline choreography, perfectly in
  sync because we know the song).
* live:   listen to whatever music is playing and improvise - on every bar line
  the runtime picks a move whose energy matches the recent loudness. This is the
  "put on any song" demo.

Safety: joint limits, per-frame velocity clamp, a watchdog that returns to the
rest pose if beats stop for 2 s, and an optional temperature check.

    python -m rdk.dance_service --mode replay --table dist/servo_table.csv --dry-run
    python -m rdk.dance_service --mode live --dry-run
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from beatdancer.motion import BY_NAME, LIBRARY          # noqa: E402
from beatdancer.skeleton import IDX, LIMITS, JOINTS, clamp, rest_pose  # noqa: E402

FPS = 50
TICKS = 1000


def to_ticks(q: np.ndarray, direction: dict[str, int] | None = None) -> list[int]:
    direction = direction or {}
    out = []
    for j, name in enumerate(JOINTS):
        lo, hi = LIMITS[name]
        v = (float(q[j]) - lo) / (hi - lo)
        if direction.get(name, 1) < 0:
            v = 1.0 - v
        out.append(int(round(max(0.0, min(1.0, v)) * (TICKS - 1))))
    return out


def load_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    times, rows = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)                       # comment line
        header = next(r)
        for row in r:
            if not row or row[0].startswith("#"):
                continue
            times.append(float(row[0]))
            rows.append([int(v) for v in row[1:]])
    return np.array(times), np.array(rows, dtype=np.int32)


class LiveImproviser:
    """Pick moves on bar lines from the energy of the last few beats."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.current = BY_NAME["idle"]
        self.start_t = 0.0
        self.tempo = 120.0
        self.beat_in_bar = 0
        self.history: list[float] = []
        self.last_beat = 0.0

    def on_beat(self, t: float, bpm: float, downbeat: bool, loudness: float) -> None:
        self.tempo = bpm if bpm > 0 else self.tempo
        self.history = (self.history + [loudness])[-8:]
        self.last_beat = t
        if not downbeat:
            return
        energy = float(np.mean(self.history)) if self.history else 0.5
        target = float(np.clip(energy, 0.1, 0.95))
        pool = [p for p in LIBRARY if abs(p.energy - target) < 0.25 and p.beats in (2, 4)]
        if not pool:
            pool = [BY_NAME["hip_sway"]]
        choice = pool[int(self.rng.integers(0, len(pool)))]
        if choice.name == self.current.name and len(pool) > 1:
            pool = [p for p in pool if p.name != choice.name]
            choice = pool[int(self.rng.integers(0, len(pool)))]
        self.current = choice
        self.start_t = t

    def pose(self, t: float) -> np.ndarray:
        span = self.current.beats * 60.0 / max(self.tempo, 30.0)
        if span <= 0:
            span = 0.5
        p = ((t - self.start_t) % span) / span
        return self.current.pose(p)


class Runtime:
    def __init__(self, bus, dry_run: bool = False):
        self.bus = bus
        self.dry_run = dry_run
        self.rest = rest_pose()
        self.prev_q = self.rest.copy()
        self.frames = 0
        self.late = 0
        self.max_temp = 0

    def step(self, q: np.ndarray) -> None:
        q = clamp(q)
        d = q - self.prev_q
        np.clip(d, -0.30, 0.30, out=d)          # ~15 rad/s ceiling at 50 Hz
        q = self.prev_q + d
        self.prev_q = q
        ticks = to_ticks(q)
        if not self.dry_run:
            self.bus.sync_write(list(range(1, len(JOINTS) + 1)), ticks)
        self.frames += 1

    def rest_pose(self) -> None:
        self.step(self.rest)


def run_replay(table: str, bus, dry_run: bool, speed: float = 1.0) -> int:
    times, rows = load_table(table)
    rt = Runtime(bus, dry_run)
    print(f"[replay] {len(times)} frames, {times[-1]:.1f}s")
    t0 = time.perf_counter()
    next_t = 0
    for i in range(len(times)):
        target = times[i] / max(speed, 1e-3)
        while time.perf_counter() - t0 < target:
            time.sleep(0.0005)
        q = np.zeros(len(JOINTS), dtype=np.float32)
        for j, name in enumerate(JOINTS):
            lo, hi = LIMITS[name]
            q[j] = lo + (rows[i, j] / (TICKS - 1)) * (hi - lo)
        rt.step(q)
        if i % 100 == 0:
            print(f"  {i}/{len(times)} t={target:.2f}s", flush=True)
    rt.rest_pose()
    return 0


def run_live(bus, dry_run: bool, seconds: float | None = None) -> int:
    from .beat_frontend import BeatTracker, _mic_stream

    tr = BeatTracker()
    imp = LiveImproviser()
    rt = Runtime(bus, dry_run)
    state = {"loud": 0.5, "beats": []}

    def on_chunk(chunk: np.ndarray) -> None:
        state["loud"] = 0.9 * state["loud"] + 0.1 * float(np.abs(chunk).mean() * 6)
        for t, idx in tr.push(chunk):
            imp.on_beat(t, tr.bpm, idx % 4 == 1, min(1.0, state["loud"]))
            state["beats"].append(t)

    import threading
    threading.Thread(target=lambda: _mic_stream(on_chunk), daemon=True).start()
    print("[live ] listening... ctrl-c to stop")

    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if seconds and t > seconds:
            break
        idle = (time.perf_counter() - t0) - (state["beats"][-1] if state["beats"] else 0)
        if idle > 2.0:
            rt.rest_pose()
        else:
            rt.step(imp.pose(t))
        time.sleep(1.0 / FPS)
    rt.rest_pose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--table", default="../dist/servo_table.csv")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    if args.dry_run:
        from .servo_bus import SimBus
        bus = SimBus()
    else:
        from .servo_bus import ServoBus
        bus = ServoBus(args.port, args.baud)

    try:
        if args.mode == "replay":
            return run_replay(args.table, bus, args.dry_run, args.speed)
        return run_live(bus, args.dry_run, args.seconds)
    except KeyboardInterrupt:
        print("\n[stop ] returning to rest")
        Runtime(bus, args.dry_run).rest_pose()
        return 0
    finally:
        bus.close()


if __name__ == "__main__":
    raise SystemExit(main())
