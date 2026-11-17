"""Unit tests: the numbers quoted in the README must stay true.

    python -m unittest discover tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatdancer import audio, choreo, export  # noqa: E402
from beatdancer.motion import BY_NAME, LIBRARY  # noqa: E402
from beatdancer.skeleton import IDX, LIMITS, N_JOINTS, clamp, fk, rest_pose  # noqa: E402


def click_track(bpm: float, seconds: float = 12.0, sr: int = audio.SR_DEFAULT) -> np.ndarray:
    """Metronome with an accent on every 4th beat - a minimal 'song'."""
    n = int(seconds * sr)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    t = 0.0
    k = 0
    rng = np.random.RandomState(0)
    while t < seconds - period:
        i = int(t * sr)
        L = int(0.06 * sr)
        env = np.exp(-np.arange(L) / (0.012 * sr)).astype(np.float32)
        body = np.sin(2 * np.pi * 90 * np.arange(L) / sr).astype(np.float32)
        accent = 1.0 if k % 4 == 0 else 0.6
        seg = body * env * accent + 0.12 * rng.randn(L).astype(np.float32) * env
        x[i:i + L] += seg[: max(0, min(L, n - i))]
        t += period
        k += 1
    return x


class TestTempo(unittest.TestCase):
    def test_bpm_within_one_percent(self):
        for bpm in (100.0, 124.0, 140.0):
            x = click_track(bpm)
            est = audio.analyze(x, audio.SR_DEFAULT).bpm
            self.assertLess(abs(est - bpm) / bpm, 0.01, f"{bpm} -> {est}")

    def test_beat_grid_is_even(self):
        a = audio.analyze(click_track(120.0), audio.SR_DEFAULT)
        d = np.diff(a.beat_times)
        self.assertLess(np.std(d) / np.mean(d), 0.02)

    def test_demo_track_numbers(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "assets", "demo_track.wav")
        if not os.path.exists(path):
            self.skipTest("demo track missing; run tools/make_demo_track.py")
        x, sr = audio.read_wav(path)
        a = audio.analyze(x, sr)
        self.assertLess(abs(a.bpm - 124.0), 0.5)
        self.assertGreater(a.n_beats, 60)


class TestChoreography(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = click_track(124.0, seconds=16.0)
        cls.a = audio.analyze(x, audio.SR_DEFAULT)
        cls.cho = choreo.make_choreography(cls.a, seed=0)

    def test_every_beat_is_covered(self):
        covered = np.zeros(self.a.n_beats + 2, dtype=bool)
        for s in self.cho.segments:
            covered[s.start_beat:s.start_beat + s.n_beats] = True
        # all beats from the first move to the last must belong to some segment
        first = self.cho.segments[0].start_beat
        last = max(s.start_beat + s.n_beats for s in self.cho.segments)
        self.assertTrue(covered[first:last].all(), "choreography has holes")

    def test_poses_finite_and_clamped(self):
        q = self.cho.poses
        self.assertTrue(np.isfinite(q).all())
        from beatdancer.skeleton import JOINTS
        for j, name in enumerate(JOINTS):
            lo, hi = LIMITS[name]
            self.assertGreaterEqual(q[:, j].min(), lo - 1e-4, name)
            self.assertLessEqual(q[:, j].max(), hi + 1e-4, name)

    def test_velocity_limited(self):
        d = np.abs(np.diff(self.cho.poses, axis=0)).max()
        self.assertLessEqual(d, choreo.VEL_LIMIT + 0.06)   # small margin after smoothing

    def test_feet_stay_on_floor(self):
        zs = []
        for i in range(0, len(self.cho.times), 25):
            r = self.cho.root[i]
            fr = fk(self.cho.poses[i], root=np.array([r[0], r[1], r[2]]), yaw=float(r[3]))
            zs.append(min(fr.points["toe_L"][2], fr.points["toe_R"][2],
                          fr.points["ankle_L"][2], fr.points["ankle_R"][2]))
        self.assertGreater(min(zs), -0.006)

    def test_moves_are_known(self):
        for s in self.cho.segments:
            self.assertIn(s.name, BY_NAME)


class TestPrimitives(unittest.TestCase):
    def test_loop_continuity(self):
        for p in LIBRARY:
            if p.name in ("clap", "robot_arm"):
                continue          # intentional percussive snaps
            err = float(np.abs(p.pose(0.0) - p.pose(0.999)).max())
            self.assertLess(err, 0.12, f"{p.name} does not loop ({err:.3f})")

    def test_mirror_is_involution(self):
        q = BY_NAME["disco_point_L"].pose(0.3)
        from beatdancer.skeleton import mirror
        self.assertLess(float(np.abs(mirror(mirror(q)) - q).max()), 1e-5)


class TestExport(unittest.TestCase):
    def test_servo_ticks_in_range(self):
        x = click_track(120.0, seconds=6.0)
        cho = choreo.make_choreography(audio.analyze(x, audio.SR_DEFAULT))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = export.servo_csv(cho, os.path.join(d, "servo.csv"))
            with open(path) as f:
                lines = [l for l in f if not l.startswith("#")][1:]
            for line in lines[:50]:
                vals = [int(v) for v in line.strip().split(",")[1:]]
                self.assertEqual(len(vals), N_JOINTS)
                self.assertTrue(all(0 <= v < 1000 for v in vals))


if __name__ == "__main__":
    unittest.main()
