"""Audio front-end: wav IO, STFT, mel, onset, tempo, beat tracking, downbeat.

Dependency policy: this module only needs numpy + the standard library, so it can
run on the RDK X5 CPU (and inside a 48h hackathon) without librosa/madmom.

The beat tracker follows the dynamic-programming formulation of
D. Ellis, "Beat Tracking by Dynamic Programming" (J. New Music Research, 2007).
That paper is the reference implementation used by librosa.beat.beat_track;
we re-implement the scoring/backtracking here because librosa is not available
on the board image.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field

import numpy as np

SR_DEFAULT = 22050
HOP = 256
N_FFT = 1024
N_MELS = 64


# --------------------------------------------------------------------------- io
def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM wav as mono float32 in [-1, 1]. Supports 8/16/24/32-bit."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")

    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x.astype(np.float32), sr


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    """Write mono float32 [-1,1] to 16-bit PCM wav."""
    y = np.clip(x, -1.0, 1.0)
    pcm = (y * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resampling. Good enough for beat detection (low band focus)."""
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.linspace(0.0, len(x) / sr_in, n_out, endpoint=False)
    return np.interp(t_out, np.arange(len(x)) / sr_in, x).astype(np.float32)


# ------------------------------------------------------------------------ stft
def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Magnitude spectrogram, hann window, centred by reflection padding."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + hop), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * np.hanning(n_fft).astype(np.float32)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)).T  # (freq, time)
    return spec.astype(np.float32)


def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = 30.0, fmax: float | None = None) -> np.ndarray:
    """Slaney-style triangular mel filters, shape (n_mels, n_fft//2+1)."""
    fmax = fmax or sr / 2.0

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = l + 1
        if r <= c:
            r = c + 1
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    # area normalise (Slaney)
    enorm = 2.0 / (hz[2:n_mels + 2] - hz[:n_mels])
    fb *= enorm[:, None]
    return fb


def log_mel(x: np.ndarray, sr: int = SR_DEFAULT, n_mels: int = N_MELS,
            hop: int = HOP, n_fft: int = N_FFT) -> np.ndarray:
    spec = stft_mag(x, n_fft=n_fft, hop=hop)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = fb @ (spec ** 2)
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


# ---------------------------------------------------------------------- onsets
def _moving_average(x: np.ndarray, w: int, axis: int = -1) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, x)


def onset_envelope(x: np.ndarray, sr: int = SR_DEFAULT, hop: int = HOP) -> np.ndarray:
    """Spectral-flux onset strength (mel bands), detrended and normalised."""
    mel = log_mel(x, sr=sr, hop=hop)
    # half-wave rectified first difference, summed over bands
    diff = np.diff(mel, axis=1, prepend=mel[:, :1])
    flux = np.maximum(diff, 0.0).sum(axis=0)
    # local detrend removes slow loudness changes
    flux = flux - _moving_average(flux, 32)
    flux = np.maximum(flux, 0.0)
    m = flux.max()
    return (flux / m if m > 1e-9 else flux).astype(np.float32)


def rms_energy(x: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = len(x) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1)).astype(np.float32)


# ----------------------------------------------------------------------- tempo
def estimate_bpm(env: np.ndarray, sr: int = SR_DEFAULT, hop: int = HOP,
                 bpm_min: float = 60.0, bpm_max: float = 190.0,
                 prior_bpm: float = 120.0, prior_width: float = 1.0) -> float:
    """Autocorrelation tempo estimate with a log-normal tempo prior."""
    fps = sr / hop
    e = env - env.mean()
    if np.allclose(e, 0):
        return prior_bpm

    n = len(e)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    ac /= (ac[0] + 1e-9)

    lo = max(1, int(math.floor(fps * 60.0 / bpm_max)))
    hi = min(n - 1, int(math.ceil(fps * 60.0 / bpm_min)))
    if hi <= lo:
        return prior_bpm

    lags = np.arange(lo, hi + 1)
    bpms = 60.0 * fps / lags
    # tempo prior (log-normal in bpm)
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / prior_width) ** 2)
    # slight preference for faster peels when ambiguous (avoids half-tempo)
    score = ac[lo:hi + 1] * prior
    # add harmonic support: a true tempo also shows energy at 2x lag
    best = int(np.argmax(score))
    return float(bpms[best])


def _beat_dp(env: np.ndarray, bpm: float, sr: int = SR_DEFAULT, hop: int = HOP,
             tightness: float = 100.0) -> np.ndarray:
    """Ellis dynamic-programming beat tracker -> list of frame indices."""
    fps = sr / hop
    period = 60.0 * fps / bpm
    n = len(env)
    if n < 4:
        return np.array([], dtype=int)

    # local score: emphasise peaks against a moving median
    local = env - np.median(env)
    local = np.maximum(local, 0.0)

    cum = np.zeros(n, dtype=np.float64)
    back = np.full(n, -1, dtype=np.int32)

    lo = max(1, int(round(period / 2.0)))
    hi = max(lo + 1, int(round(period * 2.0)))

    for i in range(n):
        best_score, best_prev = -1.0, -1
        for tau in range(lo, hi + 1):
            j = i - tau
            if j < 0:
                break
            tz = math.log(tau / period) * tightness
            cand = cum[j] - tz * tz
            if cand > best_score:
                best_score, best_prev = cand, j
        if best_prev >= 0:
            cum[i] = local[i] + best_score
            back[i] = best_prev
        else:
            cum[i] = local[i]

    # start backtracking from the best cumulative score in the last period
    tail = int(round(period))
    start = int(np.argmax(cum[-max(tail, 1):])) + max(n - tail, 0)
    beats = []
    i = start
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats.reverse()
    return np.array(beats, dtype=int)


def track_beats(env: np.ndarray, bpm: float, sr: int = SR_DEFAULT,
                hop: int = HOP) -> np.ndarray:
    frames = _beat_dp(env, bpm, sr=sr, hop=hop)
    return frames * hop / sr


def downbeat_offset(beat_times: np.ndarray, mel: np.ndarray,
                    sr: int = SR_DEFAULT, hop: int = HOP,
                    meter: int = 4) -> int:
    """Pick the bar phase with the strongest spectral change.

    Four-on-the-floor dance music hits the kick on every beat, so low-band
    energy cannot tell us where bar 1 is. Chord / bass-note changes do land
    on bar lines, so we score the mel-frame difference across each beat and
    pick the phase that accumulates the most change.
    """
    if len(beat_times) < meter * 2:
        return 0
    frames = np.clip((beat_times * sr / hop).astype(int), 0, mel.shape[1] - 1)
    feats = []
    for f in frames:
        a, b = max(0, f - 1), min(mel.shape[1], f + 2)
        v = mel[:, a:b].mean(axis=1)
        feats.append(v)
    feats = np.stack(feats, axis=1)                        # (n_mels, n_beats)
    z = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    # cosine distance between consecutive beats
    num = (z[:, 1:] * z[:, :-1]).sum(axis=0)
    den = np.linalg.norm(z[:, 1:], axis=0) * np.linalg.norm(z[:, :-1], axis=0) + 1e-6
    dist = 1.0 - num / den                                  # length n_beats-1
    best, best_score = 0, -1.0
    for phase in range(meter):
        idx = np.arange(phase, len(dist), meter)
        if len(idx) == 0:
            continue
        score = float(dist[idx].sum())
        if score > best_score:
            best, best_score = phase, score
    return best


# ------------------------------------------------------------------ structure
def beat_sync(mel: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Average mel frames between consecutive beats -> (n_mels, n_beats)."""
    out = []
    edges = np.clip(beat_frames, 0, mel.shape[1])
    for k in range(len(edges) - 1):
        a, b = int(edges[k]), int(edges[k + 1])
        if b <= a:
            b = min(a + 1, mel.shape[1])
        out.append(mel[:, a:b].mean(axis=1))
    return np.stack(out, axis=1) if out else np.zeros((mel.shape[0], 0), np.float32)


def novelty(bss: np.ndarray, kernel: int = 8) -> np.ndarray:
    """Checkerboard-kernel novelty on a beat-synchronous self-similarity matrix."""
    if bss.shape[1] < 3:
        return np.zeros(bss.shape[1], dtype=np.float32)
    z = (bss - bss.mean(axis=1, keepdims=True)) / (bss.std(axis=1, keepdims=True) + 1e-6)
    ssm = (z.T @ z) / z.shape[0]  # cosine-ish
    ssm = np.clip(ssm, -1, 1)

    k = min(kernel, ssm.shape[0] // 2)
    if k < 1:
        return np.zeros(ssm.shape[1], dtype=np.float32)
    checker = np.ones((2 * k, 2 * k), dtype=np.float32)
    checker[:k, :k] = -1
    checker[k:, k:] = -1
    # skip diagonal band (cheap Gaussian taper would do; small windows are fine)
    nov = np.zeros(ssm.shape[0], dtype=np.float32)
    for i in range(k, ssm.shape[0] - k):
        patch = ssm[i - k:i + k, i - k:i + k]
        nov[i] = float((patch * checker).sum()) / (4.0 * k * k)
    return nov


def align_grid(env: np.ndarray, period: float, duration: float,
               sr: int = SR_DEFAULT, hop: int = HOP,
               period_tol: float = 0.03) -> tuple[np.ndarray, float, float]:
    """Snap the beat grid onto the actual transients.

    A DP tracker can be a whole beat-phase off when a track opens with a
    pad or a pickup. We search (period, offset) around the estimate and
    maximise the onset strength sampled on the grid; this is a cheap
    version of the "pulse train correlation" used by tempo-refinement
    stages in beat_this / BeatNet.
    """
    times = np.arange(len(env)) * hop / sr
    grid = np.arange(0.0, duration, period)
    best = (-1.0, period, 0.0)
    for pi in np.linspace(-period_tol, period_tol, 25):
        p = period * (1.0 + pi)
        for off in np.linspace(0.0, p, 33, endpoint=False):
            g = np.arange(off, duration, p)
            score = float(np.interp(g, times, env).sum() / max(len(g), 1))
            if score > best[0]:
                best = (score, p, off)
    _, p, off = best
    # second pass at finer resolution around the winner
    for pi in np.linspace(-0.004, 0.004, 9):
        pp = p * (1.0 + pi)
        for oo in np.linspace(off - 0.02, off + 0.02, 17):
            g = np.arange(oo, duration, pp)
            score = float(np.interp(g, times, env).sum() / max(len(g), 1))
            if score > best[0]:
                best = (score, pp, oo)
    _, p, off = best
    g = np.arange(off, duration, p)
    return g, float(p), float(off)


def refine_grid(beat_times: np.ndarray, duration: float) -> tuple[np.ndarray, float, float]:
    """Fit t_k = t0 + k*period by least squares and rebuild an even beat grid.

    The DP tracker jitters by a frame or two; a fitted grid keeps bars exact,
    which matters because the choreographer snaps every move to the bar.
    Returns (grid, period, t0).
    """
    n = len(beat_times)
    if n < 4:
        return beat_times, (beat_times[1] - beat_times[0] if n > 1 else 0.5), (beat_times[0] if n else 0.0)
    k = np.arange(n, dtype=np.float64)
    A = np.stack([k, np.ones_like(k)], axis=1)
    period, t0 = np.linalg.lstsq(A, beat_times, rcond=None)[0]
    if not (0.25 < period < 2.0):
        period = float(np.median(np.diff(beat_times)))
    # rebuild from the first beat, extend to the end of the track
    t0 = float(t0)
    count = int(np.floor((duration - t0) / period)) + 1
    grid = t0 + period * np.arange(max(count, 1))
    grid = grid[(grid >= 0) & (grid < duration)]
    return grid.astype(np.float64), float(period), t0


def _pool_features(mel: np.ndarray, spec: np.ndarray, beat_frames: np.ndarray,
                   meter: int, n_bars: int) -> np.ndarray:
    """Per-bar textural feature vector: loudness, drum density, band balance."""
    freqs = np.linspace(0.0, 1.0, spec.shape[0])
    low = spec[freqs < 0.12].sum(axis=0)
    mid = spec[(freqs >= 0.12) & (freqs < 0.55)].sum(axis=0)
    high = spec[freqs >= 0.55].sum(axis=0)

    def bar_mean(v, a, b):
        a = int(np.clip(a, 0, len(v) - 1))
        b = int(np.clip(b, a + 1, len(v)))
        return float(v[a:b].mean())

    feats = np.zeros((n_bars, 5), dtype=np.float64)
    for k in range(n_bars):
        i0 = k * meter
        i1 = min(len(beat_frames), (k + 1) * meter)
        if i1 <= i0:
            continue
        fa = int(beat_frames[i0])
        fb = int(beat_frames[i1]) if i1 < len(beat_frames) else len(mel[0]) - 1
        fb = max(fb, fa + 1)
        seg = mel[:, fa:fb]
        feats[k, 0] = seg.mean()
        feats[k, 1] = seg.max(axis=0).mean()
        feats[k, 2] = np.log1p(bar_mean(low, fa, fb))
        feats[k, 3] = np.log1p(bar_mean(mid, fa, fb))
        feats[k, 4] = np.log1p(bar_mean(high, fa, fb))
    # drum density: onset peak count per bar
    return feats


def segment_sections(beat_times: np.ndarray, mel: np.ndarray,
                     beat_frames: np.ndarray, meter: int = 4,
                     min_bars: int = 2, onset_env: np.ndarray | None = None,
                     spec: np.ndarray | None = None) -> list[tuple[int, int, str]]:
    """Cut the track into sections at bar lines.

    Chords change every bar in most dance music, so a plain self-similarity
    matrix is noisy. We instead look at *texture* (loudness, band balance,
    onset density) and run change-point detection on 2-bar windows. That is
    what the ear hears as "the drums came in" / "the drop hit" / "it broke down".
    """
    n_beats = len(beat_times)
    if n_beats < meter * 4 or spec is None:
        return [(0, n_beats, "main")]

    n_bars = n_beats // meter
    feats = _pool_features(mel, spec, beat_frames, meter, n_bars)
    if onset_env is not None:
        dens = np.zeros(n_bars, dtype=np.float64)
        for k in range(n_bars):
            fa = int(beat_frames[k * meter])
            fb = int(beat_frames[min(len(beat_frames) - 1, (k + 1) * meter)])
            fb = max(fb, fa + 2)
            dens[k] = float((onset_env[fa:fb] > 0.35).sum()) / max(1, fb - fa)
        feats = np.concatenate([feats, dens[:, None]], axis=1)

    z = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    w = np.array([1.0, 1.0, 1.0, 0.7, 0.7] + ([1.5] if onset_env is not None else []))
    z = z * w[: z.shape[1]]

    d = np.zeros(n_bars, dtype=np.float64)
    for b in range(1, n_bars):
        lo = max(0, b - 2)
        hi = min(n_bars, b + 2)
        left = z[lo:b].mean(axis=0)
        right = z[b:hi].mean(axis=0)
        d[b] = float(np.abs(left - right).sum()) * min(b - lo, hi - b) / 2.0

    d /= (d.max() + 1e-9)
    bars = []
    for b in range(min_bars, n_bars - min_bars + 1):
        lo, hi = max(0, b - 2), min(n_bars, b + 3)
        if d[b] >= d[lo:hi].max() - 1e-9 and d[b] > 0.25:
            if not bars or b - bars[-1] >= min_bars:
                bars.append(b)

    bounds = [0] + bars + [n_bars]
    sections = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            sections.append((a * meter, min(n_beats, b * meter)))

    energy = [float(feats[a, 0]) for a, b in zip(bounds[:-1], bounds[1:])]
    e = (np.array(energy) - min(energy)) / (max(energy) - min(energy) + 1e-6)

    named = []
    n = len(sections)
    for i, ((a, b), ei) in enumerate(zip(sections, e)):
        if n >= 3 and i == 0 and ei < 0.6:
            label = "intro"
        elif n >= 3 and i == n - 1 and ei < 0.6:
            label = "outro"
        elif ei < 0.35:
            label = "break"
        elif ei < 0.72:
            label = "verse"
        else:
            label = "chorus"
        named.append((a, b, label))
    return named


@dataclass
class Analysis:
    sr: int
    duration: float
    bpm: float
    beat_times: np.ndarray
    downbeat: int = 0
    meter: int = 4
    sections: list[tuple[int, int, str]] = field(default_factory=list)
    beat_energy: np.ndarray | None = None

    @property
    def n_beats(self) -> int:
        return len(self.beat_times)

    def bar_index(self, beat: int) -> int:
        return (beat - self.downbeat) // self.meter

    def to_dict(self) -> dict:
        return {
            "sr": self.sr,
            "duration": round(float(self.duration), 3),
            "bpm": round(float(self.bpm), 2),
            "meter": self.meter,
            "downbeat": int(self.downbeat),
            "n_beats": int(self.n_beats),
            "beat_times": [round(float(t), 4) for t in self.beat_times],
            "beat_energy": [round(float(v), 4) for v in (self.beat_energy or [])],
            "sections": [
                {"start_beat": int(a), "end_beat": int(b), "label": lab}
                for a, b, lab in self.sections
            ],
        }


def analyze(x: np.ndarray, sr: int = SR_DEFAULT, beats_per_bar: int = 4) -> Analysis:
    """Full front-end: onset -> tempo -> beats -> downbeat -> sections."""
    if sr != SR_DEFAULT:
        x = resample(x, sr, SR_DEFAULT)
        sr = SR_DEFAULT

    env = onset_envelope(x, sr=sr)
    bpm = estimate_bpm(env, sr=sr)
    beat_times = track_beats(env, bpm, sr=sr)
    beat_times, period, t0 = align_grid(env, 60.0 / bpm, len(x) / sr, sr=sr)
    bpm = 60.0 / period

    mel = log_mel(x, sr=sr)
    hop = HOP
    db = downbeat_offset(beat_times, mel, sr=sr, hop=hop, meter=beats_per_bar)
    beat_frames = (beat_times * sr / hop).astype(int)
    spec = stft_mag(x)
    sections = segment_sections(beat_times, mel, beat_frames, meter=beats_per_bar,
                                onset_env=env, spec=spec)

    e = rms_energy(x, hop=hop)
    e = e / (np.percentile(e, 95) + 1e-6)
    e = np.clip(e, 0, 1.5)
    idx = np.clip((beat_times * sr / hop).astype(int), 0, len(e) - 1)
    beat_energy = np.array([e[max(0, i - 2):i + 3].max() for i in idx], dtype=np.float32)
    # normalise to 0..1 over the track so the composer sees relative energy
    beat_energy = (beat_energy - beat_energy.min()) / (np.ptp(beat_energy) + 1e-6)

    n_beats = len(beat_times)

    # snap section edges to bar lines (bar 0 starts on the first downbeat)
    def snap(i: int) -> int:
        return int(round(db + beats_per_bar * round((i - db) / beats_per_bar)))

    snapped, prev = [], 0
    for i, (a, b, label) in enumerate(sections):
        a2 = 0 if i == 0 else max(prev, min(n_beats, snap(a)))
        b2 = max(a2 + 1, min(n_beats, snap(b)))
        snapped.append((a2, b2, label))
        prev = b2
    sections = snapped

    return Analysis(sr=sr, duration=len(x) / sr, bpm=bpm, beat_times=beat_times,
                    downbeat=db, meter=beats_per_bar, sections=sections,
                    beat_energy=beat_energy)


def onset_envelope_low(x: np.ndarray, sr: int = SR_DEFAULT, hop: int = HOP) -> np.ndarray:
    """Low-band (30-160 Hz) onset strength, used to locate kick / downbeat."""
    spec = stft_mag(x, hop=hop)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band = (freqs >= 30) & (freqs <= 160)
    low = spec[band].sum(axis=0)
    low = np.log1p(low * 10.0)
    flux = np.maximum(np.diff(low, prepend=low[:1]), 0.0)
    flux = flux - _moving_average(flux, 32)
    flux = np.maximum(flux, 0.0)
    m = flux.max()
    return (flux / m if m > 1e-9 else flux).astype(np.float32)
