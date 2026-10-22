"""Pilot: does video-assisted ball recovery help? (truth = the tracker ball we hide)

Hides the tracker ball in N windows (known truth), detects candidates in the video every `step` frames, links them
(video_ball.link), and compares, INSIDE the windows, the possession labels obtained with (a) the hidden ball
(hold the state), (b) the recovered ball, against our own un-masked labels. Aggregates only.

    python -m mtah.video_pilot --data data --video /content/BAR-ATH_tactical.mp4 --n 24 --secs 12
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .ball import clean_ball
from .config import Config
from .io import load_tactical
from .pipeline import predict
from .video_ball import detect_stretch, fill_between, link, to_pitch

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def pick_windows(ball, hvalid, cfg, n, secs, seed):
    rng = np.random.default_rng(seed)
    L = int(secs * cfg.fps)
    ok = ~np.isnan(ball[:, 0])
    cand = []
    for a0, b0 in cfg.blocks:
        for s in range(a0 + 100, b0 - L - 100, 50):
            seg = slice(s - 25, s + L + 25)
            if ok[seg].mean() >= 0.9 and hvalid[s:s + L].all():
                cand.append(s)
    rng.shuffle(cand)
    chosen = []
    for s in cand:
        if all(abs(s - c) > L + 150 for c in chosen):
            chosen.append(s)
        if len(chosen) == n:
            break
    return sorted((s, s + L - 1) for s in chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="yolov8x.pt")
    a = ap.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    hom = pd.read_csv(f"{t}/BAR-ATH_tactical_homography.csv")
    Hm = np.full((cfg.n_frames, 9), np.nan); Hm[hom["frame"].to_numpy()] = hom[HCOLS].to_numpy()
    hvalid = np.zeros(cfg.n_frames, bool); hvalid[hom["frame"].to_numpy()] = hom["valid"].to_numpy() == 1
    ref = predict(ball, P, team, cfg)
    wins = pick_windows(ball, hvalid, cfg, a.n, a.secs, a.seed)
    print(f"[pilot] {len(wins)} windows of {a.secs:.0f}s, detection every {a.step} frames")
    cap = cv2.VideoCapture(a.video)
    model = YOLO(a.weights)
    half = torch.cuda.is_available()
    b_hold, b_vid = ball.copy(), ball.copy()
    rec_frames = prec_ok = tot_frames = 0
    errs = []
    t0 = time.time()
    for (s, e) in wins:
        f0, f1 = s - 1, e + 1
        p0, p1 = ball[f0].copy(), ball[f1].copy()
        truth = ball[s:e + 1].copy()
        b_hold[s:e + 1] = np.nan; b_vid[s:e + 1] = np.nan
        samples = []
        for f, xy, cf in detect_stretch(cap, model, s, e, a.step, half=half):
            if len(cf) == 0:
                continue
            pos = to_pitch(Hm[f], xy, cfg)
            d = np.hypot(P[f, :, 0][None, :] - pos[:, 0:1], P[f, :, 1][None, :] - pos[:, 1:2])
            dm = np.nanmin(np.where(np.isnan(d), 99.0, d), axis=1)
            samples.append((f, pos, cf, dm))
        path = link(samples, f0, p0, f1, p1, cfg.fps)
        b_vid = fill_between(b_vid, path, f0, p0, f1, p1)
        got = ~np.isnan(b_vid[s:e + 1, 0])
        err = np.hypot(*(b_vid[s:e + 1] - truth).T)
        rec_frames += int(got.sum()); tot_frames += len(got)
        prec_ok += int((got & (err <= 3.0)).sum()); errs += list(err[got])
    print(f"[pilot] detection+linking took {(time.time() - t0) / 60:.1f} min")
    print(f"[pilot] frames with a recovered ball: {rec_frames / tot_frames:.2f} of window frames; "
          f"within 3 m of the truth: {prec_ok / max(1, rec_frames):.2f} of those (median error {np.median(errs):.1f} m)")
    lab_hold = predict(b_hold, P, team, cfg)
    lab_vid = predict(b_vid, P, team, cfg)
    dh, dv = [], []
    for s, e in wins:
        r = ref[s:e + 1]
        dh.append(float((lab_hold[s:e + 1] == r).mean())); dv.append(float((lab_vid[s:e + 1] == r).mean()))
    d = np.array(dv) - np.array(dh)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"[pilot] label agreement with the un-masked output inside the windows: hold {np.mean(dh):.3f}, "
          f"video-assisted {np.mean(dv):.3f}; paired difference {d.mean():+.3f} +- {se:.3f} (SE); "
          f"better in {int((d > 0.005).sum())}, worse in {int((d < -0.005).sum())} of {len(d)} windows")
    print("[pilot] go / no-go: continue only if the difference is >= +0.03 AND > 2 SE")


if __name__ == "__main__":
    main()