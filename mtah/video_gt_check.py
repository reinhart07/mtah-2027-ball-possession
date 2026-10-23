"""Truth-based check of video-assisted ball recovery on the PUBLIC 5 minutes.

The public window is the only place with true labels. We hide the tracker ball there in realistic gaps (lengths drawn from
the REAL gap-length distribution), then compare hold-the-state vs video-assisted labels against the GROUND TRUTH (not
against our own output). Detection runs once; masks differ per seed; differences are paired per seed.
Caveat: a few of these frames may have been among the detector's training crops (mild optimism).

    python -m mtah.video_gt_check --data data --video /content/BAR-ATH_tactical.mp4 --weights <best.pt> --tiled \
        --gt /content/org/evaluation/gt/BAR-ATH_possession_gt_public.csv --w-conf 0.5 --seeds 0 1 2 3 4 5 6 7
"""
import argparse
import os

import numpy as np
import pandas as pd

from .benchmark_gap import real_gap_stats
from .config import Config
from .io import load_tactical
from .metrics import spell_f1
from .pipeline import predict
from .sweep import macro_f1
from .video_ball import detect_stretch, fill_between, inside_pitch, link, to_pitch

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def pick_masks(ball, hvalid, ga, gb, lens, share, rng, tries=3000):
    visible = ~np.isnan(ball[:, 0])
    target = share * visible[ga:gb + 1].sum()
    masks, hidden = [], 0
    for _ in range(tries):
        L = int(rng.choice(lens))
        if L > (gb - ga) // 3:
            continue
        s = int(rng.integers(ga + 60, gb - L - 60)); e = s + L - 1
        if not (visible[s - 1] and visible[e + 1] and hvalid[s:e + 1].all()):
            continue
        if any(not (e + 100 < ms or s - 100 > me) for ms, me in masks):
            continue
        masks.append((s, e)); hidden += int(visible[s:e + 1].sum())
        if hidden >= target:
            break
    return sorted(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--w-conf", type=float, default=0.5)
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
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
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    idx = np.where(gt != "X")[0]; ga, gb = int(idx.min()), int(idx.max())
    lens = real_gap_stats(ball, cfg)
    lens = lens[lens >= cfg.gap_apply_min]
    cap = cv2.VideoCapture(a.video)
    model = YOLO(a.weights)
    samples = []
    for f, xy, cf in detect_stretch(cap, model, ga, gb, a.step, half=torch.cuda.is_available(), tiled=a.tiled):
        if len(cf) == 0:
            continue
        pos = to_pitch(Hm[f], xy, cfg)
        if a.tiled:
            keep = inside_pitch(pos, cfg); pos, cf = pos[keep], cf[keep]
            if len(cf) == 0:
                continue
        d = np.hypot(P[f, :, 0][None, :] - pos[:, 0:1], P[f, :, 1][None, :] - pos[:, 1:2])
        samples.append((f, pos, cf, np.nanmin(np.where(np.isnan(d), 99.0, d), axis=1)))
    ref = predict(ball, P, team, cfg)
    ref_s, ref_m = spell_f1(ref, gt, [(ga, gb)]), macro_f1(ref, gt, ga, gb)
    print(f"[gt] window {ga}..{gb}; un-masked output: spellF1 {ref_s:.3f}, macroF1 {ref_m:.3f}")
    ds, dm_ = [], []
    for sd in a.seeds:
        rng = np.random.default_rng(sd)
        masks = pick_masks(ball, hvalid, ga, gb, lens, a.share, rng)
        b_hold, b_vid = ball.copy(), ball.copy()
        for s, e in masks:
            b_hold[s:e + 1] = np.nan; b_vid[s:e + 1] = np.nan
        for s, e in masks:
            ss = [x for x in samples if s <= x[0] <= e]
            path = link(ss, s - 1, ball[s - 1], e + 1, ball[e + 1], cfg.fps, w_conf=a.w_conf)
            b_vid = fill_between(b_vid, path, s - 1, ball[s - 1], e + 1, ball[e + 1])
        lh, lv = predict(b_hold, P, team, cfg), predict(b_vid, P, team, cfg)
        sh, sv = spell_f1(lh, gt, [(ga, gb)]), spell_f1(lv, gt, [(ga, gb)])
        mh, mv = macro_f1(lh, gt, ga, gb), macro_f1(lv, gt, ga, gb)
        ds.append(sv - sh); dm_.append(mv - mh)
        print(f"[gt] seed {sd}: {len(masks)} masked gaps; spellF1 hold {sh:.3f} -> video {sv:.3f}; macroF1 hold {mh:.3f} -> video {mv:.3f}")
    ds, dm_ = np.array(ds), np.array(dm_)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    print(f"[gt] paired video - hold vs the GROUND TRUTH: spellF1 {ds.mean():+.3f} +- {se(ds):.3f} SE, macroF1 {dm_.mean():+.3f} +- {se(dm_):.3f} SE "
          f"({len(ds)} seeds; better in {int((ds > 0.005).sum())}, worse in {int((ds < -0.005).sum())})")
    print("[gt] go / no-go (last check): spellF1 difference >= +0.03 AND > 2 SE, and macroF1 difference >= 0")


if __name__ == "__main__":
    main()
