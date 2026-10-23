"""Fill the real ball-less gaps with video-assisted ball positions (resumable, one cache file per gap).

    python -m mtah.video_fill --data data --video /content/BAR-ATH_tactical.mp4 --out /content/drive/MyDrive/mtah/video_cache

Detection every `step` frames (YOLOv8x, class 'sports ball'), candidates projected to the pitch with the inverse
homography, then linked between the tracker's last ball BEFORE the gap and first ball AFTER it (video_ball.link).
The cache holds derived frame-level data: it must NOT be committed or published (competition rules).
Then:  python -m mtah.run --data data --out Rei_JR_video.csv --video-cache <dir>
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .ball import clean_ball, runs
from .config import Config
from .io import load_tactical
from .video_ball import detect_stretch, fill_between, inside_pitch, link, to_pitch

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def list_gaps(b, cfg: Config, min_len=20):
    """(start, end_exclusive) of ball-less runs inside the graded blocks with a known ball on both sides."""
    gaps = []
    for a0, b0 in cfg.blocks:
        for s, e in runs(np.isnan(b[a0:b0 + 1, 0])):
            s, e = s + a0, e + a0
            if e - s >= min_len and s > a0 and e <= b0:
                gaps.append((s, e))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--w-conf", type=float, default=2.0, help="linker weight on detector confidence")
    ap.add_argument("--tiled", action="store_true", help="tiled inference for a fine-tuned single-class ball detector")
    a = ap.parse_args()
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    os.makedirs(a.out, exist_ok=True)
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    b, _ = clean_ball(ball, cfg)
    hom = pd.read_csv(f"{t}/BAR-ATH_tactical_homography.csv")
    Hm = np.full((cfg.n_frames, 9), np.nan); Hm[hom["frame"].to_numpy()] = hom[HCOLS].to_numpy()
    gaps = list_gaps(b, cfg, a.min_len)
    todo = [g for g in gaps if not os.path.exists(os.path.join(a.out, f"gap_{g[0]}_{g[1]}.npz"))]
    frames_todo = sum(e - s for s, e in todo)
    print(f"[fill] {len(gaps)} gaps ({sum(e - s for s, e in gaps)} frames); {len(todo)} still to do "
          f"({frames_todo} frames, about {frames_todo / a.step} detections)")
    json.dump({"ultralytics": ultralytics.__version__, "torch": torch.__version__, "weights": a.weights,
               "step": a.step, "min_len": a.min_len, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              open(os.path.join(a.out, "meta.json"), "w"))
    cap = cv2.VideoCapture(a.video)
    model = YOLO(a.weights)
    half = torch.cuda.is_available()
    t0 = time.time()
    for n, (s, e) in enumerate(todo, 1):
        f0, f1 = s - 1, e
        p0, p1 = b[f0].copy(), b[f1].copy()
        samples = []
        for f, xy, cf in detect_stretch(cap, model, s, e - 1, a.step, half=half, tiled=a.tiled):
            if len(cf) == 0 or np.isnan(Hm[f, 0]):
                continue
            pos = to_pitch(Hm[f], xy, cfg)
            if a.tiled:
                keep = inside_pitch(pos, cfg); pos, cf = pos[keep], cf[keep]
                if len(cf) == 0:
                    continue
            d = np.hypot(P[f, :, 0][None, :] - pos[:, 0:1], P[f, :, 1][None, :] - pos[:, 1:2])
            dm = np.nanmin(np.where(np.isnan(d), 99.0, d), axis=1)
            samples.append((f, pos, cf, dm))
        path = link(samples, f0, p0, f1, p1, cfg.fps, w_conf=a.w_conf)
        tmp = np.full((f1 - f0 + 1, 2), np.nan)
        tmp = fill_between(tmp, [(f - f0, p) for f, p in path], 0, p0, f1 - f0, p1)
        np.savez_compressed(os.path.join(a.out, f"gap_{s}_{e}.npz"), frames=np.arange(s, e), xy=tmp[1:f1 - f0])
        if n % 10 == 0 or n == len(todo):
            el = (time.time() - t0) / 60
            print(f"[fill] {n}/{len(todo)} gaps, {el:.1f} min elapsed, about {el / n * (len(todo) - n):.1f} min left")
    print("[fill] done. Next: python -m mtah.run --data data --out Rei_JR_video.csv --video-cache", a.out)


if __name__ == "__main__":
    main()
