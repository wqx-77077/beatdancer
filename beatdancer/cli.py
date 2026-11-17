"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import export
from .audio import analyze, read_wav
from .choreo import make_choreography


def _load(path: str):
    x, sr = read_wav(path)
    print(f"[audio] {path}  {len(x) / sr:.2f}s @ {sr} Hz")
    a = analyze(x, sr)
    print(f"[beat ] {a.bpm:.2f} BPM, {a.n_beats} beats, downbeat offset {a.downbeat}")
    for s, e, lab in a.sections:
        print(f"        section {lab:7s} beats {s:3d}-{e:3d}")
    return x, sr, a


def cmd_demo(args) -> int:
    x, sr, a = _load(args.audio)
    cho = make_choreography(a, seed=args.seed)
    print(f"[dance] {len(cho.segments)} moves, {len(cho.times)} frames @ {cho.fps} Hz")
    os.makedirs(args.out, exist_ok=True)

    data = cho.to_dict(max_frames=args.max_frames)
    with open(os.path.join(args.out, "choreo.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    export.summary_csv(cho, os.path.join(args.out, "choreo.csv"))
    export.servo_csv(cho, os.path.join(args.out, "servo_table.csv"))
    export.bvh(cho, os.path.join(args.out, "dance.bvh"))
    export.g1_qpos(cho, os.path.join(args.out, "g1_qpos.csv"))
    if args.servo_config:
        export.servo_config(os.path.join(args.out, "servo_config.json"))

    from .web import build
    html = build(data, args.audio, os.path.join(args.out, "beatdancer.html"))
    print(f"[web  ] {html}")

    if args.render:
        from .render import render_video
        print("[video] rendering frames ...")
        info = render_video(cho, os.path.join(args.out), fps=args.fps,
                            gif_seconds=args.gif_seconds)
        print(f"[video] {info}")
    return 0


def cmd_analyze(args) -> int:
    _, _, a = _load(args.audio)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(a.to_dict(), f, ensure_ascii=False, indent=1)
        print(f"[json ] {args.json}")
    return 0


def cmd_render(args) -> int:
    from .render import render_video
    x, sr, a = _load(args.audio)
    cho = make_choreography(a, seed=args.seed)
    info = render_video(cho, args.out, fps=args.fps, gif_seconds=args.gif_seconds,
                        max_seconds=args.seconds)
    print(info)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("beatdancer", description="music -> robot dance")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="run the whole pipeline")
    d.add_argument("audio")
    d.add_argument("--out", default="dist")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--fps", type=int, default=30)
    d.add_argument("--max-frames", type=int, default=1200)
    d.add_argument("--gif-seconds", type=float, default=8.0)
    d.add_argument("--render", action="store_true")
    d.add_argument("--servo-config", action="store_true")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", help="only analyse the song")
    a.add_argument("audio")
    a.add_argument("--json", default=None)
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("render", help="render a video / gif")
    r.add_argument("audio")
    r.add_argument("--out", default="dist")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--seconds", type=float, default=None)
    r.add_argument("--gif-seconds", type=float, default=8.0)
    r.set_defaults(func=cmd_render)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
