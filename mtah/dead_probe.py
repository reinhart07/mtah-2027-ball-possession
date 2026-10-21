"""What does the ATD look like while the corrected ground truth says the ball is DEAD?

    python -m mtah.dead_probe --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv

Prints per-stretch statistics of dead (D) vs live frames and a 1 s timeline around each D stretch.
Purpose: find which signals separate dead from live in THIS tracking, then build the detector on them.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

from .ball import clean_ball
from .config import Config
from .diagnose import rle


def player_activity(P, fps, win=25):
    """median over tracked players of displacement per second, centred on each frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.hypot(P[win:, :, 0] - P[:-win, :, 0], P[win:, :, 1] - P[:-win, :, 1]) / (win / fps)
        ms = np.nanmedian(d, axis=1)
    out = np.full(len(P), np.nan)
    out[win // 2:win // 2 + len(ms)] = ms
    return out


def stats(mask, ball, speed, act, ntrack, cfg):
    v = speed[mask]
    xy = ball[mask]
    oob = (xy[:, 0] < 0) | (xy[:, 0] > cfg.pitch_l) | (xy[:, 1] < 0) | (xy[:, 1] > cfg.pitch_w)
    ok = ~np.isnan(xy[:, 0])
    return dict(frames=int(mask.sum()),
                ball_nan=round(float((~ok).mean()), 2),
                ball_speed_med=round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else np.nan,
                share_static=round(float((v < cfg.static_speed).mean()), 2),
                share_oob=round(float((oob & ok).sum() / max(1, ok.sum())), 2),
                player_speed_med=round(float(np.nanmedian(act[mask])), 2),
                tracked_med=int(np.median(ntrack[mask])))


def probe(P, ball_raw, gt, cfg: Config, verbose=False):
    ball, speed = clean_ball(ball_raw, cfg)
    act = player_activity(P, cfg.fps)
    ntrack = (~np.isnan(P[:, :, 0])).sum(1)
    idx = np.where(gt != "X")[0]
    a, b = int(idx.min()), int(idx.max())
    win = np.zeros(len(gt), bool); win[a:b + 1] = True
    dead = win & (gt == "D"); live = win & (gt != "D")
    print(pd.DataFrame({"DEAD": stats(dead, ball, speed, act, ntrack, cfg),
                        "LIVE": stats(live, ball, speed, act, ntrack, cfg)}).T.to_string())
    print("\n== per D stretch ==")
    for s, e, l in rle(gt[a:b + 1], a):
        if l != "D":
            continue
        m = np.zeros(len(gt), bool); m[s:e] = True
        print(f"\nD {s}-{e} ({(e - s) / cfg.fps:.1f}s)", stats(m, ball, speed, act, ntrack, cfg))
        if not verbose:
            continue
        print("  t(s)  gt  ball_xy        ball_v  player_v  tracked")
        for f in range(max(a, s - 75), min(b, e + 50), 25):
            xy = "NaN" if np.isnan(ball[f, 0]) else f"({ball[f,0]:5.1f},{ball[f,1]:5.1f})"
            print(f"  {(f - s) / cfg.fps:5.1f}  {gt[f]}   {xy:14s} {speed[f]:6.1f}  {act[f]:7.1f}  {ntrack[f]:5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--verbose", action="store_true", help="LOCAL ONLY: per-second coordinates - never paste into a hosted LLM")
    a = ap.parse_args()
    from .io import load_tactical
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    probe(P, ball, gt, cfg, a.verbose)


if __name__ == "__main__":
    main()
