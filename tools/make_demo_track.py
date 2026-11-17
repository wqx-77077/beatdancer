"""Generate a license-clean demo track (no copyrighted audio in the repo).

The track is an original 124 BPM synth loop with a clear arrangement:
intro -> verse -> build -> drop -> break -> outro. Every sound is synthesised
from sine/noise generators, so the repo stays 100% redistributable.

usage:  python tools/make_demo_track.py assets/demo_track.wav
"""
from __future__ import annotations

import os
import sys
import wave

import numpy as np

SR = 22050
BPM = 124.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
N_BARS = 18


def env_ad(n: int, attack: float, decay: float) -> np.ndarray:
    """Attack/decay envelope, in seconds."""
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))
    return (a * d).astype(np.float32)


def sine(freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def kick(dur: float = 0.30) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    freq = 150.0 * np.exp(-t / 0.035) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase).astype(np.float32)
    amp = np.exp(-t / 0.11).astype(np.float32)
    click = (np.random.RandomState(0).randn(n).astype(np.float32) * np.exp(-t / 0.004)) * 0.25
    return (body * amp + click) * 0.9


def snare(dur: float = 0.22) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.random.RandomState(1).randn(n).astype(np.float32)
    # cheap first-order highpass so the noise sits above the bass
    noise = np.diff(noise, prepend=noise[:1])
    noise *= np.exp(-t / 0.07)
    tone = sine(185.0, n) * np.exp(-t / 0.05) * 0.35
    return (noise * 0.5 + tone) * 0.6


def hat(dur: float = 0.06) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.random.RandomState(2).randn(n).astype(np.float32)
    noise = np.diff(noise, prepend=noise[:1])
    return noise * np.exp(-t / 0.012) * 0.35


def open_hat(dur: float = 0.20) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = np.random.RandomState(3).randn(n).astype(np.float32)
    noise = np.diff(noise, prepend=noise[:1])
    return noise * np.exp(-t / 0.06) * 0.22


def bass(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    e = env_ad(n, 0.005, 0.10)
    sig = sine(freq, n) + 0.35 * sine(2 * freq, n) + 0.15 * sine(3 * freq, n)
    return (sig * e * 0.55).astype(np.float32)


def pad(freqs: list[float], dur: float, amp: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        for det in (-0.4, 0.0, 0.4):
            sig += sine(f + det, n, phase=np.random.rand() * 6.28)
    sig /= (len(freqs) * 3)
    e = np.minimum(t / 0.25, 1.0) * np.minimum((dur - t) / 0.35, 1.0)
    e = np.clip(e, 0, 1)
    return (sig * e * amp).astype(np.float32)


def lead(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    e = env_ad(n, 0.004, 0.09)
    sig = sine(freq, n) + 0.3 * sine(2 * freq, n) + 0.12 * sine(3 * freq, n)
    return (sig * e * 0.16).astype(np.float32)


def place(buf: np.ndarray, x: np.ndarray, at: float) -> None:
    i = int(round(at * SR))
    if i >= len(buf):
        return
    n = min(len(x), len(buf) - i)
    buf[i:i + n] += x[:n]


def note_hz(semi_from_a4: int) -> float:
    return 440.0 * 2 ** (semi_from_a4 / 12.0)


# A minor: Am F C G  (root, third, fifth)
CHORDS = {
    "Am": (-12, [-12, -9, -5]),
    "F": (-16, [-16, -12, -9]),
    "C": (-9, [-9, -5, -1]),
    "G": (-14, [-14, -10, -7]),
}
PROG = ["Am", "F", "C", "G"]


def build() -> np.ndarray:
    n = int(N_BARS * BAR * SR) + SR // 2
    buf = np.zeros(n, dtype=np.float32)
    rng = np.random.RandomState(7)

    for bar in range(N_BARS):
        t0 = bar * BAR
        chord = PROG[bar % 4]
        root, tones = CHORDS[chord]
        section = (
            "intro" if bar < 2 else
            "verse" if bar < 6 else
            "build" if bar < 8 else
            "drop" if bar < 14 else
            "break" if bar < 16 else "outro"
        )

        # --- pad: everywhere but the drop is thinner
        pad_amp = {"intro": 0.20, "verse": 0.10, "build": 0.16, "drop": 0.13,
                   "break": 0.22, "outro": 0.18}[section]
        place(buf, pad([note_hz(s) for s in tones], BAR * 1.02, pad_amp), t0)

        # --- drums
        if section in ("verse", "drop", "outro", "build"):
            for b in range(4):
                place(buf, kick(), t0 + b * BEAT)
        if section in ("verse", "drop"):
            for b in (1, 3):
                place(buf, snare(), t0 + b * BEAT)
        if section in ("verse", "drop", "build", "intro", "outro"):
            for e in range(8):
                amp = 1.0 if e % 2 == 0 else 0.55
                place(buf, hat() * amp, t0 + e * BEAT / 2)
        if section == "drop":
            for b in range(4):
                place(buf, open_hat(), t0 + b * BEAT + BEAT / 2)

        # --- build: accelerating snare roll
        if section == "build":
            steps = 16 if bar % 2 == 0 else 32
            for s in range(steps):
                at = t0 + s * BAR / steps
                place(buf, snare() * (0.35 + 0.65 * s / steps), at)

        # --- bass
        if section in ("verse", "drop", "outro", "build"):
            pattern = [0, 0, 7, 0, 0, 0, 5, 0] if section != "drop" else [0, 0, 7, 0, 12, 0, 7, 0]
            for e, semi in enumerate(pattern):
                if semi == 0 and e % 2 == 1:
                    continue
                place(buf, bass(note_hz(root + semi), BEAT / 2 * 0.9), t0 + e * BEAT / 2)

        # --- lead arp in the drop
        if section == "drop":
            arp = [0, 4, 7, 12, 7, 4, 0, 4]
            for e, semi in enumerate(arp):
                place(buf, lead(note_hz(root + 12 + semi), BEAT / 2 * 0.8), t0 + e * BEAT / 2)

        # --- breaks before the drop: strip the kick for half a bar
        if bar == 7:
            i = int((t0 + 2 * BEAT) * SR)
            buf[i:i + int(2 * BEAT * SR)] *= 0.25

    # sidechain pump: duck the pad/bass bus at every kick
    pump = np.ones(n, dtype=np.float32)
    for bar in range(N_BARS):
        for b in range(4):
            i = int((bar * BAR + b * BEAT) * SR)
            L = int(0.16 * SR)
            if i + L >= n:
                continue
            pump[i:i + L] *= (0.55 + 0.45 * np.linspace(0, 1, L)).astype(np.float32)
    buf *= (0.72 + 0.28 * pump)

    # gentle limiter + fade in/out
    buf = np.tanh(buf * 1.6) * 0.85
    fade = int(0.35 * SR)
    buf[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    buf[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return buf


def main(out: str) -> None:
    x = build()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    print(f"wrote {out}  {len(x) / SR:.1f}s  {BPM:.0f} BPM  (original synth, no samples)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/demo_track.wav")
