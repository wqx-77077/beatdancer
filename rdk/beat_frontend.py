"""Real-time beat front-end for RDK X5 / X3 (CPU only, no librosa).

Reads the board microphone (arecord) or a wav file, produces a stream of
`BeatEvent`s at ~86 Hz, and keeps a phase-locked estimate of the beat so the
dance can stay in sync even if the tempo drifts.

This is the same DSP as beatdancer.audio, but incremental: a ring buffer,
one STFT frame per hop, an autocorrelation tempo estimate refreshed every
2 seconds, and a comb-filter phase tracker.

Test on a laptop without a board:
    python -m rdk.beat_frontend --wav assets/demo_track.wav --realtime
"""
from __future__ import annotations

import argparse
import math
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000          # the board mic runs at 16 kHz; cheaper STFT, fine for beats
HOP = 160           # 10 ms
N_FFT = 512
N_MELS = 40
BPM_MIN, BPM_MAX = 60.0, 190.0


@dataclass
class BeatEvent:
    t: float            # seconds since start
    bpm: float
    beat_index: int     # running count, for bar counting
    is_downbeat: bool
    confidence: float


def mel_fb(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(35.0), hz2mel(sr / 2), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.clip(np.floor((n_fft + 1) * hz / sr).astype(int), 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        c = max(c, l + 1)
        r = max(r, c + 1)
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / (r - c)
    return fb


class BeatTracker:
    """Incremental onset -> tempo -> phase tracker."""

    def __init__(self, sr: int = SR, hop: int = HOP, history_s: float = 8.0):
        self.sr, self.hop = sr, hop
        self.fb = mel_fb(sr, N_FFT, N_MELS)
        self.win = np.hanning(N_FFT).astype(np.float32)
        self.buf = np.zeros(N_FFT, dtype=np.float32)
        self.prev_mel = None
        self.env = deque(maxlen=int(history_s * sr / hop))
        self.n_frames = 0
        self.bpm = 120.0
        self.phase = 0.0          # beat phase in [0, 1)
        self.last_beat_t = -1.0
        self.beat_index = 0
        self.conf = 0.0
        self.t0 = time.time()

    # -- audio in ---------------------------------------------------------
    def push(self, samples: np.ndarray) -> list[tuple[float, int]]:
        """Feed PCM; returns (time_s, beat_index) for every beat that fired."""
        fired = []
        self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
        while len(self.buf) >= N_FFT:
            frame = self.buf[:N_FFT] * self.win
            self.buf = self.buf[self.hop:]
            fluid = self._onset(frame)
            self.env.append(fluid)
            self.n_frames += 1
            if self.n_frames % 20 == 0:      # every 200 ms
                self._update_tempo()
            if self._tick():
                fired.append((self._now(), self.beat_index))
        return fired

    def _onset(self, frame: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(frame))[: N_FFT // 2 + 1] ** 2
        mel = self.fb @ spec
        db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        if self.prev_mel is None:
            self.prev_mel = db
            return 0.0
        d = np.maximum(db - self.prev_mel, 0.0).sum()
        self.prev_mel = db
        return float(d)

    def _update_tempo(self) -> None:
        e = np.asarray(self.env, dtype=np.float32)
        if len(e) < 64:
            return
        e = e - e.mean()
        fps = self.sr / self.hop
        ac = np.correlate(e, e, mode="full")[len(e) - 1:]
        ac /= (ac[0] + 1e-9)
        lo = int(fps * 60.0 / BPM_MAX)
        hi = int(fps * 60.0 / BPM_MIN)
        hi = min(hi, len(ac) - 1)
        if hi <= lo:
            return
        lags = np.arange(lo, hi + 1)
        bpms = 60.0 * fps / lags
        prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
        score = ac[lo:hi + 1] * prior
        best = int(np.argmax(score))
        new_bpm = float(bpms[best])
        self.conf = float(min(1.0, max(0.0, score[best])))
        # smooth tempo changes, the PLL should not jump on a single window
        self.bpm = 0.8 * self.bpm + 0.2 * new_bpm

    def _tick(self) -> bool:
        """Advance the phase; return True when a beat lands this frame."""
        dt = self.hop / self.sr
        self.phase += dt * self.bpm / 60.0
        if self.phase >= 1.0:
            self.phase -= 1.0
            self.beat_index += 1
            self.last_beat_t = self._now()
            return True
        return False

    def _now(self) -> float:
        return self.n_frames * self.hop / self.sr

    def nudge(self, amount: float) -> None:
        """Phase correction from an observed onset (soft, +/-0.06 beat)."""
        self.phase = (self.phase + max(-0.06, min(0.06, amount))) % 1.0


def _mic_stream(on_chunk):
    """Spawn arecord and pump 10 ms chunks. Board has ALSA, Mac has not."""
    proc = subprocess.Popen(
        ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE, bufsize=0)
    chunk = HOP * 2
    try:
        while True:
            raw = proc.stdout.read(chunk)
            if not raw or len(raw) < chunk:
                break
            on_chunk(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
    finally:
        proc.kill()


def _wav_stream(path: str, realtime: bool):
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x), n, endpoint=False),
                      np.arange(len(x)), x).astype(np.float32)
    step = HOP * 4
    for i in range(0, len(x) - step, step):
        yield x[i:i + step]
        if realtime:
            time.sleep(step / SR * 0.9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="replay a wav instead of the mic")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()

    tr = BeatTracker()
    q: queue.Queue = queue.Queue()

    if args.wav:
        def later():
            for chunk in _wav_stream(args.wav, args.realtime):
                for ev in tr.push(chunk):
                    q.put(ev)
            q.put(None)
        threading.Thread(target=later, daemon=True).start()
    else:
        def on_chunk(chunk):
            for ev in tr.push(chunk):
                q.put(ev)
        threading.Thread(target=lambda: _mic_stream(on_chunk), daemon=True).start()

    print(f"# listening at {SR} Hz, hop {HOP} ({1000 * HOP / SR:.1f} ms)")
    while True:
        try:
            item = q.get(timeout=5.0)
        except queue.Empty:
            print(f"bpm={tr.bpm:6.2f} conf={tr.conf:.2f} (idle)")
            continue
        if item is None:
            break
        t, idx = item
        down = idx % 4 == 1
        print(f"beat {idx:5d}  t={t:7.3f}s  bpm={tr.bpm:6.2f}  "
              f"conf={tr.conf:.2f}  {'DOWNBEAT' if down else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
