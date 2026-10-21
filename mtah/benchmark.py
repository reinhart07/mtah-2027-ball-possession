"""Occlusion stress test (needs no ground truth).

Hide the ball in random windows of a chosen length, re-run the pipeline, and compare with the
un-masked output. Measures how gracefully the decoder survives the 41% missing-ball problem.
The reference is our own un-masked output, NOT the truth: it measures robustness only.
"""
import argparse
import os

import numpy as np

from .config import Config
from .io import load_tactical
from .metrics import spell_f1
from .pipeline import predict


def mask_ball(ball, cfg, gap_frames, share, seed):
    rng = np.random.default_rng(seed)
    b = ball.copy()
    target = share * (~np.isnan(b[:, 0])).sum()
    hidden = 0
    while hidden < target:
        a, e = cfg.blocks[rng.integers(2)]
        s = int(rng.integers(a, e - gap_frames))
        hidden += int((~np.isnan(b[s:s + gap_frames, 0])).sum())
        b[s:s + gap_frames] = np.nan
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--share", type=float, default=0.3)
    a = ap.parse_args()
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    ref = predict(ball, P, team, cfg)
    for secs in (1, 2, 4, 8):
        lab = predict(mask_ball(ball, cfg, secs * cfg.fps, a.share, 0), P, team, cfg)
        g = np.concatenate([np.arange(x, y + 1) for x, y in cfg.blocks])
        print(f"gap {secs}s: spellF1 vs unmasked {spell_f1(lab, ref, cfg.blocks):.3f} | "
              f"frame agreement {(lab[g] == ref[g]).mean():.3f}")


if __name__ == "__main__":
    main()


def mask_ball_mix(ball, cfg, lens, share, seed):
    """Hide the ball in windows whose lengths follow the REAL gap-length distribution."""
    rng = np.random.default_rng(seed)
    b = ball.copy()
    target = share * (~np.isnan(b[:, 0])).sum()
    hidden = 0
    while hidden < target:
        L = int(rng.choice(lens))
        a, e = cfg.blocks[rng.integers(2)]
        if e - a <= L:
            continue
        s = int(rng.integers(a, e - L))
        hidden += int((~np.isnan(b[s:s + L, 0])).sum())
        b[s:s + L] = np.nan
    return b
