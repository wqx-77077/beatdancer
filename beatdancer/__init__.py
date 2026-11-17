"""beatdancer - music driven choreography for a 16-DOF servo humanoid.

A 48h-scale pipeline: listen -> understand the song -> compose a dance ->
render it -> stream joint targets to the board.

    python -m beatdancer demo assets/demo_track.wav --out dist

Stages can also be run one by one (analyze / compose / render / export / web).
"""
from __future__ import annotations

from .audio import Analysis, analyze, read_wav
from .choreo import Choreography, make_choreography, plan, sample

__all__ = ["analyze", "read_wav", "Analysis", "make_choreography",
           "Choreography", "plan", "sample"]
__version__ = "0.1.0"
