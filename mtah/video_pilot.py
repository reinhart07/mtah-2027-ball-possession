"""Pilot: does video-assisted ball recovery help? (truth = the tracker ball we hide)

Hides the tracker ball in N windows (known truth), detects candidates in the video every `step` frames ONCE, then
links them under one or several linker settings (--w-conf) and compares, INSIDE the windows, the possession labels
from (a) the hidden ball (hold the state) and (b) the recovered ball, against our own un-masked labels.
Aggregates only.

    python -m mtah.video_pilot --data data --video /content/BAR-ATH_tactical.mp4 --n 24 --secs 12
    python -m mtah.video_pilot ... --weights <best.pt> --tiled --min-frame 168000 --w-conf 2 1 0.5 0.25
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .config import Config
from .io import load_tactical
from .pipeline import predict
from .video_ball import detect_stretch, fill_between, inside_pitch, link, to_pitch

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def pick_windows(ball, hvalid, cfg, n, secs, seed, min_frame=0, min_ok=0.9):
    rng = np.random.default_rng(seed)
    L = int(secs * cfg.fps)
    ok = ~np.isnan(ball[:, 0])
    cand = []
    for a0, b0 in cfg.blocks:
        for s in range(max(a0 + 100, min_frame), b0 - L - 100, 50):
            seg = slice(s - 25, s + L + 25)
            if ok[seg].mean() >= min_ok and hvalid[s:s + L].all():
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
    ap.add_argument("--min-frame", type=int, default=0, help="only use windows from here on (held-out period for a fine-tuned detector)")
    ap.add_argument("--min-ok", type=float, default=0.9, help="share of tracker-visible ball frames required around a window")
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--tiled", action="store_true", help="tiled inference for a fine-tuned single-class ball detector")
    ap.add_argument("--w-conf", type=float, nargs="+", default=[2.0], help="linker weight on detector confidence (lower = trust weak detections more)")
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
    wins = pick_windows(ball, hvalid, cfg, a.n, a.secs, a.seed, a.min_frame, a.min_ok)
    print(f"[pilot] {len(wins)} windows of {a.secs:.0f}s, detection every {a.step} frames")
    cap = cv2.VideoCapture(a.video)
    model = YOLO(a.weights)
    half = torch.cuda.is_available()
    t0 = time.time()
    store = []                                           # detect once, link under every setting
    n_cand = n_samp = 0
    for (s, e) in wins:
        samples = []
        for f, xy, cf in detect_stretch(cap, model, s, e, a.step, half=half, tiled=a.tiled):
            n_samp += 1
            if len(cf) == 0:
                continue
            pos = to_pitch(Hm[f], xy, cfg)
            if a.tiled:
                keep = inside_pitch(pos, cfg); pos, cf = pos[keep], cf[keep]
                if len(cf) == 0:
                    continue
            d = np.hypot(P[f, :, 0][None, :] - pos[:, 0:1], P[f, :, 1][None, :] - pos[:, 1:2])
            dm = np.nanmin(np.where(np.isnan(d), 99.0, d), axis=1)
            samples.append((f, pos, cf, dm)); n_cand += len(cf)
        store.append((s, e, ball[s - 1].copy(), ball[e + 1].copy(), ball[s:e + 1].copy(), samples))
    print(f"[pilot] detection took {(time.time() - t0) / 60:.1f} min; {sum(len(w[5]) for w in store)} of {n_samp} sample frames "
          f"have a candidate ({n_cand / max(1, n_samp):.1f} per sample frame); median candidate confidence "
          f"{np.median(np.concatenate([w[5][i][2] for w in store for i in range(len(w[5]))])) if n_cand else float('nan'):.2f}")
    b_hold = ball.copy()
    for s, e, *_ in store:
        b_hold[s:e + 1] = np.nan
    lab_hold = predict(b_hold, P, team, cfg)
    dh = np.array([float((lab_hold[s:e + 1] == ref[s:e + 1]).mean()) for s, e, *_ in store])
    for wc in a.w_conf:
        b_vid = b_hold.copy()
        rec = prec = tot = 0
        errs = []
        for s, e, p0, p1, truth, samples in store:
            path = link(samples, s - 1, p0, e + 1, p1, cfg.fps, w_conf=wc)
            b_vid = fill_between(b_vid, path, s - 1, p0, e + 1, p1)
            got = ~np.isnan(b_vid[s:e + 1, 0])
            err = np.hypot(*(b_vid[s:e + 1] - truth).T)
            rec += int(got.sum()); tot += len(got)
            prec += int((got & (err <= 3.0)).sum()); errs += list(err[got & ~np.isnan(err)])
        lab_vid = predict(b_vid, P, team, cfg)
        dv = np.array([float((lab_vid[s:e + 1] == ref[s:e + 1]).mean()) for s, e, *_ in store])
        d = dv - dh
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"[pilot] w_conf={wc}: recovered {rec / max(1, tot):.2f} of window frames, within 3 m {prec / max(1, rec):.2f} "
              f"(median error {np.nanmedian(errs) if errs else float('nan'):.1f} m); label agreement hold {dh.mean():.3f} -> "
              f"video {dv.mean():.3f}, difference {d.mean():+.3f} +- {se:.3f} SE; better in {int((d > 0.005).sum())}, "
              f"worse in {int((d < -0.005).sum())} of {len(d)}")
    print("[pilot] go / no-go: continue only if the difference is >= +0.03 AND > 2 SE")


if __name__ == "__main__":
    main()
