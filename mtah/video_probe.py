"""Feasibility probe: can a pretrained detector find the ball in the TACTICAL video?

Truth = the tracker's own ball (projected to pixels with the provided homography), so this prints aggregate
recall only. Two strata: random graded frames, and frames just BEFORE a ball-less gap (the situations where the
tracker is about to lose the ball). Needs a GPU runtime, opencv-python-headless and ultralytics.

    python -m mtah.video_probe --data data --video /content/BAR-ATH_tactical.mp4 --n 300
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from .ball import runs
from .config import Config

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def project(H, xn, yn):
    """normalised pitch (0-1) -> pixel."""
    v = H.reshape(3, 3) @ np.array([xn, yn, 1.0])
    return v[0] / v[2], v[1] / v[2]


def sample_frames(ball, hvalid, cfg: Config, n, seed=0, pre=75, min_frame=0):
    """(random_frames, pre_gap_frames): graded frames with a tracker ball and a valid homography."""
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(ball[:, 0]) & hvalid
    graded = np.zeros(len(ball), bool)
    for a, b in cfg.blocks:
        graded[a:b + 1] = True
    ok = ok & (np.arange(len(ball)) >= min_frame)
    pool = np.where(ok & graded)[0]
    pre_pool = []
    for a, b in cfg.blocks:
        for s, e in runs(np.isnan(ball[a:b + 1, 0])):
            if e - s >= 75 and a + s - pre >= a:
                pre_pool += [f for f in range(a + s - pre, a + s) if ok[f]]
    k = n // 2
    return (rng.choice(pool, min(k, len(pool)), replace=False),
            rng.choice(np.array(pre_pool), min(k, len(pre_pool)), replace=False) if pre_pool else np.array([], int))


def score_frame(centers, confs, truth_xy, radius):
    """(any detection within radius, best-confidence detection within radius, number of detections)"""
    if len(centers) == 0:
        return False, False, 0
    d = np.hypot(centers[:, 0] - truth_xy[0], centers[:, 1] - truth_xy[1])
    return bool((d <= radius).any()), bool(d[int(np.argmax(confs))] <= radius), len(centers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--radius", type=float, default=20.0, help="pixels")
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--min-frame", type=int, default=0, help="only sample frames from here on (held-out period)")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    from .sync_broadcast import load_ball
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball = load_ball(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg, cfg.n_frames)
    hom = pd.read_csv(f"{t}/BAR-ATH_tactical_homography.csv")
    Hm = np.full((cfg.n_frames, 9), np.nan)
    Hm[hom["frame"].to_numpy()] = hom[HCOLS].to_numpy()
    hvalid = np.zeros(cfg.n_frames, bool)
    hvalid[hom["frame"].to_numpy()] = hom["valid"].to_numpy() == 1
    rnd, pre = sample_frames(ball, hvalid, cfg, a.n, min_frame=a.min_frame)
    cap = cv2.VideoCapture(a.video)
    print(f"[probe] video frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} (expected {cfg.n_frames}); "
          f"frame size {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    model = YOLO(a.weights)
    for name, frames in (("random graded frames", rnd), ("just before a ball-less gap", pre)):
        hit = top1 = nd = tot = 0
        t0 = time.time()
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            tx, ty = project(Hm[f], ball[f, 0] / cfg.pitch_l, ball[f, 1] / cfg.pitch_w)
            if a.tiled:
                from .video_ball import detect_tiled
                xy, cf = detect_tiled(model, [img])[0]
            else:
                r = model.predict(img, imgsz=a.imgsz, conf=0.03, classes=[32], verbose=False)[0]
                xy = r.boxes.xywh.cpu().numpy()[:, :2] if len(r.boxes) else np.zeros((0, 2))
                cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            h, t1, k = score_frame(xy, cf, (tx, ty), a.radius)
            hit += h; top1 += t1; nd += k; tot += 1
        if tot:
            print(f"[probe] {name}: n={tot}  any-detection-within-{a.radius:.0f}px {hit / tot:.2f}  "
                  f"best-confidence-correct {top1 / tot:.2f}  detections/frame {nd / tot:.1f}  "
                  f"{(time.time() - t0) / tot:.2f}s/frame")
    print("[probe] go / no-go rule: continue only if best-confidence-correct >= 0.40 overall AND >= 0.25 before gaps")


if __name__ == "__main__":
    main()
