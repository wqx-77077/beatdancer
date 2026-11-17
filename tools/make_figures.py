"""Generate the two figures used in the project book (纯 PIL，无 matplotlib 依赖).

    python tools/make_figures.py
-> docs/fig_pipeline.png  (four-layer architecture)
-> docs/fig_timeline.png  (analysis + choreography on the real demo track)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beatdancer import audio, choreo  # noqa: E402

FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
BG = (12, 16, 26)
INK = (232, 240, 250)
DIM = (140, 165, 195)
ACC = (94, 231, 255)
ACC2 = (167, 139, 250)
WARM = (255, 176, 102)


def font(size: int, bold: bool = False):
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=0)
            except Exception:
                continue
    return ImageFont.load_default()


def rounded(d, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def arrow(d, p0, p1, color=ACC, width=3):
    d.line([p0, p1], fill=color, width=width)
    (x0, y0), (x1, y1) = p0, p1
    ang = np.arctan2(y1 - y0, x1 - x0)
    L = 12
    d.polygon([(x1, y1),
               (x1 - L * np.cos(ang - 0.4), y1 - L * np.sin(ang - 0.4)),
               (x1 - L * np.cos(ang + 0.4), y1 - L * np.sin(ang + 0.4))], fill=color)


def pipeline(path: str) -> str:
    W, H = 1680, 620
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title = font(34)
    f_box = font(26)
    f_body = font(19)
    f_small = font(17)

    d.text((48, 34), "BeatDancer · 四层流水线", font=f_title, fill=INK)
    d.text((48, 82), "输入一首歌，输出 16 个关节的 50 Hz 轨迹；中间每一层都可单独替换", font=f_body, fill=DIM)

    layers = [
        ("① 听觉层", "audio.py", ["对数梅尔谱", "谱通量 onset", "自相关测速", "DP 追拍 + 脉冲对齐", "小节 / 段落切分"], ACC),
        ("② 大脑层", "motion.py + choreo.py", ["20 个动作基元", "乐句级动态规划", "能量 / 衔接 / 重复代价", "动机 + 镜像呼应"], ACC2),
        ("③ 身体层", "skeleton.py", ["16 自由度 FK", "关节限位", "接缝交叉淡化", "速度限幅 + 足底贴地"], (255, 176, 102)),
        ("④ 执行层", "rdk/", ["Feetech 总线", "SYNC_WRITE 16 机同帧", "50 Hz 运行时", "断拍看门狗"], (120, 235, 180)),
    ]

    bw, bh = 372, 300
    gap = (W - 4 * bw - 96) / 3
    y0 = 150
    for i, (title, sub, items, col) in enumerate(layers):
        x = 48 + i * (bw + gap)
        rounded(d, [x, y0, x + bw, y0 + bh], 18, fill=(20, 27, 42), outline=col, width=2)
        d.text((x + 22, y0 + 18), title, font=f_box, fill=col)
        d.text((x + 22, y0 + 54), sub, font=f_small, fill=DIM)
        for k, it in enumerate(items):
            yy = y0 + 96 + k * 38
            d.ellipse([x + 24, yy + 7, x + 31, yy + 14], fill=col)
            d.text((x + 42, yy), it, font=f_body, fill=INK)
        if i < 3:
            arrow(d, (x + bw + 6, y0 + bh / 2), (x + bw + gap - 8, y0 + bh / 2))

    # io labels
    d.text((48, y0 + bh + 56), "输入：wav / 麦克风 16 kHz", font=f_small, fill=DIM)
    d.text((W - 48, y0 + bh + 34), "输出：网页 Demo · MP4 · servo_table.csv", font=f_small,
           fill=DIM, anchor="ra")
    d.text((W - 48, y0 + bh + 60), "BVH 动作文件 · Unitree G1 轨迹", font=f_small,
           fill=DIM, anchor="ra")
    arrow(d, (48, y0 + bh + 40), (W - 48, y0 + bh + 40), color=(70, 90, 120), width=2)
    img.save(path)
    return path


def timeline(path: str, wav: str) -> str:
    x, sr = audio.read_wav(wav)
    a = audio.analyze(x, sr)
    cho = choreo.make_choreography(a, seed=0)

    W, H = 1680, 640
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title = font(30)
    f = font(18)
    f_small = font(15)
    f_tiny = font(13)

    d.text((44, 28), "一首歌的解剖：拍点、段落与编舞", font=f_title, fill=INK)
    d.text((44, 68), f"demo_track.wav · {a.bpm:.1f} BPM · {a.duration:.1f}s · "
                     f"{a.n_beats} 拍 · {len(cho.segments)} 个动作段", font=f, fill=DIM)

    L, R = 90, W - 44
    span = a.duration

    def X(t):
        return L + (R - L) * t / span

    # 1) onset envelope
    env = audio.onset_envelope(x, sr=sr)
    et = np.arange(len(env)) * audio.HOP / sr
    y0, hgt = 120, 120
    d.text((44, y0 - 26), "onset 包络（每一次鼓点是一个峰）", font=f_small, fill=DIM)
    for k in range(0, len(env) - 1, 2):
        xx = X(et[k])
        yy = y0 + hgt - float(env[k]) * hgt
        d.line([(xx, y0 + hgt), (xx, yy)], fill=(60, 150, 190), width=2)

    # 2) beats
    yb = y0 + hgt + 26
    d.text((44, yb - 24), "拍点与小节线（青色 = 强拍 / 小节起点）", font=f_small, fill=DIM)
    for i, t in enumerate(a.beat_times):
        down = (i - a.downbeat) % a.meter == 0
        h = 16 if down else 8
        d.line([(X(t), yb), (X(t), yb + h)], fill=ACC if down else (70, 110, 140), width=2)

    # 3) sections
    ys = yb + 52
    d.text((44, ys - 26), "段落结构（变点检测 + 小节对齐）", font=f_small, fill=DIM)
    colors = {"intro": (70, 110, 150), "verse": (90, 150, 210), "chorus": (167, 139, 250),
              "break": (255, 176, 102), "outro": (110, 130, 160), "main": (90, 150, 210)}
    for (b0, b1, lab) in a.sections:
        t0 = a.beat_times[min(b0, a.n_beats - 1)]
        t1 = a.beat_times[min(b1, a.n_beats - 1)]
        rounded(d, [X(t0) + 1, ys, X(t1) - 1, ys + 34], 6,
                fill=colors.get(lab, (90, 150, 210)), outline=None, width=0)
        d.text(((X(t0) + X(t1)) / 2 - 18, ys + 8), lab, font=f_small, fill=(12, 16, 26))

    # 4) choreography blocks
    yc = ys + 66
    d.text((44, yc - 28), "编舞结果（每个方块是一个动作，长度 = 音乐长度）", font=f_small, fill=DIM)
    palette = [(94, 231, 255), (167, 139, 250), (255, 176, 102), (120, 235, 180),
               (255, 122, 160), (150, 200, 255)]
    names = {}
    for s in cho.segments:
        key = s.zh
        if key not in names:
            names[key] = palette[len(names) % len(palette)]
        c = names[key]
        x0, x1 = X(s.t0) + 1, X(s.t1) - 1
        rounded(d, [x0, yc, max(x1, x0 + 2), yc + 44], 7, fill=c, outline=None, width=0)
        if x1 - x0 > 46:
            d.text((x0 + 6, yc + 12), s.zh, font=f_tiny, fill=(10, 14, 22))

    # 5) energy
    ye = yc + 92
    d.text((44, ye - 26), "响度 / 能量曲线（编舞的目标强度就是跟着它走）", font=f_small, fill=DIM)
    e = a.beat_energy
    for i in range(len(e) - 1):
        t0, t1 = a.beat_times[i], a.beat_times[i + 1]
        d.line([(X(t0), ye + 60 - e[i] * 60), (X(t1), ye + 60 - e[i + 1] * 60)],
               fill=WARM, width=3)

    d.text((44, H - 42),
           "说明：段落标签由响度、频段平衡与 onset 密度共同决定；强拍位置用于对齐小节线。",
           font=f_tiny, fill=(110, 135, 165))
    img.save(path)
    return path


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(out, exist_ok=True)
    print(pipeline(os.path.join(out, "fig_pipeline.png")))
    print(timeline(os.path.join(out, "fig_timeline.png"),
                   os.path.join(os.path.dirname(out), "assets", "demo_track.wav")))
