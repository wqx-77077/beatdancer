"""Software renderer: draw the choreography as a neon stage performance.

No OpenGL, no browser, no GPU: numpy does the 3D maths and Pillow draws
capsules. That keeps the demo reproducible on any laptop (and lets us render
the README animation inside CI).

Outputs: PNG frames, MP4 (if ffmpeg is available), animated GIF preview,
and a poster frame.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .choreo import Choreography
from .skeleton import BONES, fk

W, H = 540, 720
SS = 2                     # supersampling factor

BG_TOP = (10, 13, 22)
BG_BOT = (20, 26, 44)
ACCENT = (94, 231, 255)
ACCENT2 = (167, 139, 250)
WARM = (255, 176, 102)


# ------------------------------------------------------------------ utilities
def _hex(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), int(a))


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _font(size: int, bold: bool = False):
    cands = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in cands:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


CAM_POS = (0.80, 1.06, 0.60)
CAM_TARGET = (0.0, 0.0, 0.265)
CAM_FOV = 40.0


class Camera:
    def __init__(self, pos, target, fov=38.0, aspect=W / H):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.fov = fov
        self.aspect = aspect

    def view(self):
        fwd = self.target - self.pos
        fwd /= np.linalg.norm(fwd) + 1e-9
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up2 = np.cross(right, fwd)
        R = np.stack([right, up2, fwd], axis=0)
        return R, self.pos

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R, p0 = self.view()
        rel = pts - p0
        cam = rel @ R.T
        z = cam[:, 2]
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        x = cam[:, 0] * f / np.maximum(z, 1e-3)
        y = cam[:, 1] * f / np.maximum(z, 1e-3)
        sx = (x / self.aspect * 0.5 + 0.5) * W * SS
        sy = (0.5 - y * 0.5) * H * SS
        return np.stack([sx, sy], axis=1), z


def _bg() -> Image.Image:
    top = np.array(BG_TOP, dtype=np.float32)
    bot = np.array(BG_BOT, dtype=np.float32)
    t = np.linspace(0, 1, H * SS, dtype=np.float32)[:, None]
    col = (top[None, :] * (1 - t) + bot[None, :] * t)          # (H,3)
    img = np.repeat(col[:, None, :], W * SS, axis=1)
    # radial vignette
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
    cx, cy = W * SS * 0.5, H * SS * 0.52
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (W * SS * 0.75)
    v = np.clip(1.15 - 0.65 * r, 0.35, 1.0)[:, :, None]
    img = np.clip(img * v, 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _stage(img: Image.Image, t: float, beat_phase: float, energy: float) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    cam = Camera(CAM_POS, CAM_TARGET, CAM_FOV)
    # floor grid in perspective
    for k in range(-4, 5):
        x = k * 0.16
        pts = np.array([[x, -0.55, 0.0], [x, 0.55, 0.0]])
        sp, z = cam.project(pts)
        a = int(26 + 10 * energy)
        d.line([tuple(sp[0]), tuple(sp[1])], fill=(90, 120, 170, a), width=max(1, SS))
    for j in range(-3, 4):
        y = j * 0.18
        pts = np.array([[-0.65, y, 0.0], [0.65, y, 0.0]])
        sp, z = cam.project(pts)
        a = int(22 + 8 * energy)
        d.line([tuple(sp[0]), tuple(sp[1])], fill=(90, 120, 170, a), width=max(1, SS))
    # light pool, anchored to the world origin so the robot stands in it
    o, _ = cam.project(np.array([[0.0, 0.0, 0.0]]))
    cx, cy = float(o[0][0]), float(o[0][1])
    for i in range(7):
        rr = int((60 + i * 26) * SS)
        a = int((16 - i * 2) * (0.6 + 0.8 * energy))
        d.ellipse([cx - rr * 1.5, cy - rr * 0.32, cx + rr * 1.5, cy + rr * 0.32],
                  fill=(70, 120, 210, max(a, 0)))
    # beat ring
    if 0.0 <= beat_phase < 1.0:
        rr = int((30 + 120 * beat_phase) * SS)
        a = int(90 * (1 - beat_phase) * (0.4 + energy))
        d.ellipse([cx - rr * 1.5, cy - rr * 0.32, cx + rr * 1.5, cy + rr * 0.32],
                  outline=(120, 220, 255, a), width=max(1, int(2 * SS)))


def _draw_body(img: Image.Image, frame, energy: float, glow_layer: Image.Image) -> None:
    cam = Camera(CAM_POS, CAM_TARGET, CAM_FOV)
    ppu = (H * SS) / (2.0 * math.tan(math.radians(CAM_FOV) / 2.0))
    d = ImageDraw.Draw(img, "RGBA")
    gd = ImageDraw.Draw(glow_layer, "RGBA")

    items = []
    for a, b, r0, r1 in BONES:
        sp, z = cam.project(np.array([frame.points[a], frame.points[b]]))
        if (z <= 0.08).any():
            continue
        items.append((float(0.5 * (z[0] + z[1])), a, b, r0, r1, sp))
    items.sort(key=lambda x: -x[0])      # farthest first

    # depth range for shading
    zs = [it[0] for it in items] or [1.0]
    zmin, zmax = min(zs), max(zs)

    for depth, a, b, r0, r1, sp in items:
        zz = depth
        near = 1.0 - (depth - zmin) / (zmax - zmin + 1e-6)
        shade = 0.55 + 0.45 * near
        col = _lerp(ACCENT2, ACCENT, 0.35 + 0.65 * near)
        c = tuple(int(v * shade) for v in col)

        w0 = max(2, int(2 * r0 * ppu / max(zz, 0.2)))
        w1 = max(2, int(2 * r1 * ppu / max(zz, 0.2)))

        gd.line([tuple(sp[0]), tuple(sp[1])], fill=_hex(col, 55),
                width=int(max(w0, w1) * 1.9))
        d.line([tuple(sp[0]), tuple(sp[1])], fill=_hex(c, 240),
               width=int(0.5 * (w0 + w1)))
        # highlight along the top edge -> reads as a rounded metal limb
        d.line([tuple(sp[0]), tuple(sp[1])], fill=(255, 255, 255, 40),
               width=max(1, int(0.16 * (w0 + w1))))

    # torso plate (neoprene vest) makes the silhouette read as a robot
    tp, tz = cam.project(np.array([frame.points["pelvis"], frame.points["chest"]]))
    if (tz > 0.08).all():
        zz = float(tz.mean())
        wa = 0.049 * ppu / max(zz, 0.2)
        wb = 0.064 * ppu / max(zz, 0.2)
        ax, ay = tp[0]
        bx, by = tp[1]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / L, dx / L
        quad = [(ax + nx * wa, ay + ny * wa), (bx + nx * wb, by + ny * wb),
                (bx - nx * wb, by - ny * wb), (ax - nx * wa, ay - ny * wa)]
        d.polygon(quad, fill=(20, 29, 48, 240), outline=_hex(ACCENT, 120))
        # collar bone highlight
        d.line([(bx + nx * wb * 0.9, by + ny * wb * 0.9),
                (bx - nx * wb * 0.9, by - ny * wb * 0.9)],
               fill=(255, 255, 255, 45), width=max(1, int(2 * SS)))
        # chest light
        cx_, cy_ = (ax + bx) / 2, (ay + by) / 2
        r = 0.016 * ppu / max(zz, 0.2)
        d.line([(cx_ - r, cy_), (cx_ + r, cy_)], fill=_hex(ACCENT, 240), width=max(2, int(r * 0.8)))

    # head + visor
    hp, z = cam.project(np.array([frame.points["head"]]))
    if z[0] > 0.08:
        hx, hy = hp[0]
        r = 0.050 * ppu / max(z[0], 0.2)
        gd.ellipse([hx - r * 1.5, hy - r * 1.5, hx + r * 1.5, hy + r * 1.5],
                   fill=_hex(ACCENT, 45))
        d.ellipse([hx - r, hy - r, hx + r, hy + r],
                  fill=_hex(_lerp(ACCENT, ACCENT2, 0.25), 250))
        d.ellipse([hx - r * 0.97, hy - r * 0.97, hx + r * 0.97, hy + r * 0.97],
                  outline=(255, 255, 255, 90), width=max(1, SS))
        d.rounded_rectangle([hx - r * 0.66, hy - r * 0.26, hx + r * 0.66, hy + r * 0.20],
                            radius=int(r * 0.2), fill=(10, 16, 30, 235))
        glint = 0.5 + 0.5 * math.sin(energy * 9.0)
        er = max(2, r * 0.13)
        d.ellipse([hx - r * 0.44 - er, hy - er * 0.6, hx - r * 0.44 + er, hy + er * 0.6],
                  fill=(160, 245, 255, int(150 + 90 * glint)))
        d.ellipse([hx + r * 0.44 - er, hy - er * 0.6, hx + r * 0.44 + er, hy + er * 0.6],
                  fill=(160, 245, 255, int(150 + 90 * glint)))

    # contact shadow under both feet
    for name in ("toe_L", "toe_R"):
        foot = frame.points[name]
        sp, z = cam.project(np.array([np.array([foot[0], foot[1], 0.0])]))
        if z[0] > 0.08:
            sx, sy = sp[0]
            rw = 0.06 * ppu / max(z[0], 0.3)
            d.ellipse([sx - rw, sy - rw * 0.24, sx + rw, sy + rw * 0.24], fill=(0, 0, 0, 110))


def _hud(draw: ImageDraw.ImageDraw, cho: Choreography, i: int, beat_idx: int,
         section: str, move: str, bpm: float, progress: float) -> None:
    f_big = _font(22)
    f_mid = _font(14)
    f_small = _font(12)

    draw.text((18, 16), "BEATDANCER", font=f_big, fill=(235, 245, 255, 235))
    draw.text((18, 44), "音乐驱动的机器人即兴编舞", font=f_small,
              fill=(150, 190, 230, 200))

    draw.text((W - 18, 18), f"{bpm:.1f} BPM", font=f_mid,
              fill=(120, 230, 255, 230), anchor="ra")
    draw.text((W - 18, 38), section.upper(), font=f_small,
              fill=(180, 160, 255, 220), anchor="ra")

    for k in range(4):
        x = W - (18 + (3 - k) * 13)
        y = 62
        on = k == beat_idx % 4
        draw.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5],
                     fill=(130, 240, 255, 255) if on else (70, 90, 120, 200))

    draw.rounded_rectangle([18, H - 66, 18 + 158, H - 36],
                           radius=9, fill=(255, 255, 255, 20),
                           outline=(120, 200, 240, 100), width=1)
    draw.text((30, H - 59), f"动作 · {move}", font=f_mid, fill=(215, 238, 255, 235))

    draw.text((W - 18, H - 58), f"beat {beat_idx + 1:03d}", font=f_small,
              fill=(140, 170, 210, 190), anchor="ra")

    y = H - 18
    draw.line([(18, y), (W - 18, y)], fill=(60, 80, 110, 160), width=2)
    draw.line([(18, y), (18 + (W - 36) * progress, y)], fill=(120, 230, 255, 230), width=3)


def render_frame(cho: Choreography, i: int, beat_idx: int, section: str,
                 move: str, beat_phase: float, energy: float) -> Image.Image:
    img = _bg()
    _stage(img, cho.times[i], beat_phase, energy)
    q = cho.poses[i]
    r = cho.root[i]
    frame = fk(q, root=np.array([r[0], r[1], r[2]]), yaw=float(r[3]))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _draw_body(img, frame, energy, glow)
    glow = glow.filter(ImageFilter.GaussianBlur(6 * SS))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")

    # downscale first, then draw crisp HUD text
    img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    _hud(d, cho, i, beat_idx, section, move, cho.analysis.bpm, i / max(1, len(cho.times) - 1))

    # downbeat flash
    if beat_idx % 4 == 0 and beat_phase < 0.35:
        overlay = Image.new("RGBA", (W, H), (120, 200, 255, int(26 * (1 - beat_phase / 0.35))))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img


def frame_meta(cho: Choreography, i: int):
    t = cho.times[i]
    bt = cho.analysis.beat_times
    beat_idx = int(np.searchsorted(bt, t, side="right") - 1)
    beat_idx = max(0, min(beat_idx, len(bt) - 1))
    period = (bt[min(beat_idx + 1, len(bt) - 1)] - bt[beat_idx]) if beat_idx + 1 < len(bt) else 0.5
    beat_phase = (t - bt[beat_idx]) / max(period, 1e-3)
    beat_phase = min(max(beat_phase, 0.0), 0.999)
    sec = "main"
    for a, b, label in cho.analysis.sections:
        if a <= beat_idx < b:
            sec = label
    seg = None
    for s in cho.segments:
        if s.t0 <= t < s.t1:
            seg = s
            break
    if seg is None and cho.segments:
        seg = cho.segments[-1]
    energy = float(cho.analysis.beat_energy[beat_idx]) if cho.analysis.beat_energy is not None else 0.5
    return beat_idx, beat_phase, sec, (seg.zh if seg else "-"), energy


def render_video(cho: Choreography, out_dir: str, fps: int = 30, max_seconds: float | None = None,
                 gif_seconds: float = 8.0, gif_width: int = 340) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    step = max(1, int(round(cho.fps / fps)))
    idx = list(range(0, len(cho.times), step))
    if max_seconds:
        idx = [i for i in idx if cho.times[i] <= max_seconds]

    frames = []
    for n, i in enumerate(idx):
        bi, bp, sec, move, energy = frame_meta(cho, i)
        img = render_frame(cho, i, bi, sec, move, bp, energy)
        frames.append(img)
        if n % 30 == 0:
            print(f"  frame {n}/{len(idx)}", flush=True)

    poster = f"{out_dir}/poster.png"
    frames[min(len(frames) - 1, int(len(frames) * 0.62))].save(poster)

    # GIF preview (loops, small file)
    gif_frames = []
    for n, i in enumerate(idx):
        if cho.times[i] > gif_seconds:
            break
        if n % 3:            # 10 fps for the gif
            continue
        gif_frames.append(frames[n].resize(
            (gif_width, int(gif_width * H / W)), Image.LANCZOS))
    gif_path = f"{out_dir}/preview.gif"
    if gif_frames:
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / 10), loop=0, optimize=True, disposal=2)

    mp4 = None
    ff = shutil.which("ffmpeg") or os.environ.get("FFMPEG_BIN")
    if ff:
        tmp = os.path.join(out_dir, "_frames")
        os.makedirs(tmp, exist_ok=True)
        for n, img in enumerate(frames):
            img.save(os.path.join(tmp, f"{n:05d}.png"))
        mp4 = f"{out_dir}/demo.mp4"
        subprocess.run([ff, "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                        "-i", "assets/demo_track.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", mp4],
                       check=False, capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(mp4) and os.path.getsize(mp4) < 1000:
            mp4 = None

    return {"frames": len(frames), "poster": poster, "gif": gif_path, "mp4": mp4}
