"""Auto-labelled fine-tuning set for a single-class BALL detector (no hand labelling).

Label = the tracker's ball, projected to pixels with the provided homography. Training samples are 640x640 native-
resolution crops (the ball stays ~10 px; no downscaling): one POSITIVE crop with the ball at a random place, and one
NEGATIVE crop (usually around players, so the model learns that feet, heads and pitch marks are not the ball).
Validation frames come from a held-out LATE period of the match, never from the training period.

    python -m mtah.ft_dataset --data data --video /content/BAR-ATH_tactical.mp4 --out /content/ftdata
The output is derived from licensed video: keep it OUT of the repository and never publish it.
"""
import argparse
import os

import numpy as np
import pandas as pd

from .config import Config

HCOLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def project_m(Hrow, x_m, y_m, cfg: Config):
    v = np.asarray(Hrow, float).reshape(3, 3) @ np.array([x_m / cfg.pitch_l, y_m / cfg.pitch_w, 1.0])
    return v[0] / v[2], v[1] / v[2]


def make_crops(img, bpx, players_px, rng, size=640, box=16):
    """(positive crop, label line, negative crop or None); None if the ball is outside the frame."""
    H, W = img.shape[:2]
    bx, by = bpx
    if not (0 <= bx < W and 0 <= by < H):
        return None
    m = 48
    ox, oy = int(rng.integers(m, size - m)), int(rng.integers(m, size - m))
    x0 = int(np.clip(round(bx - ox), 0, W - size)); y0 = int(np.clip(round(by - oy), 0, H - size))
    pos = img[y0:y0 + size, x0:x0 + size].copy()
    lab = f"0 {(bx - x0) / size:.6f} {(by - y0) / size:.6f} {box / size:.6f} {box / size:.6f}\n"
    neg = None
    for _ in range(12):
        if len(players_px) and rng.random() < 0.8:
            px, py = players_px[int(rng.integers(len(players_px)))]
            nx = int(np.clip(round(px + rng.normal(0, 150) - size / 2), 0, W - size))
            ny = int(np.clip(round(py + rng.normal(0, 150) - size / 2), 0, H - size))
        else:
            nx, ny = int(rng.integers(0, W - size + 1)), int(rng.integers(0, H - size + 1))
        if not (nx - 24 <= bx <= nx + size + 24 and ny - 24 <= by <= ny + size + 24):
            neg = img[ny:ny + size, nx:nx + size].copy()
            break
    return pos, lab, neg


def build(cap, ball, P, Hm, hvalid, cfg: Config, out, n_train, n_val, val_start, seed=0, size=640, box=16):
    import cv2
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ball))
    graded = np.zeros(len(ball), bool)
    for a, b in cfg.blocks:
        graded[a:b + 1] = True
    ok = ~np.isnan(ball[:, 0]) & hvalid & graded
    tr_pool, va_pool = np.where(ok & (idx < val_start))[0], np.where(ok & (idx >= val_start))[0]
    tr = rng.choice(tr_pool, min(n_train, len(tr_pool)), replace=False)
    va = rng.choice(va_pool, min(n_val, len(va_pool)), replace=False)
    split = {int(f): "train" for f in tr}
    split.update({int(f): "val" for f in va})
    for s in ("train", "val"):
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    frames = sorted(split)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    cur, n = frames[0], {"train": 0, "val": 0}
    for f in frames:
        while cur < f:
            cap.grab(); cur += 1
        got, img = cap.read(); cur += 1
        if not got:
            continue
        bpx = project_m(Hm[f], ball[f, 0], ball[f, 1], cfg)
        pl = P[f][~np.isnan(P[f, :, 0])]
        players_px = [project_m(Hm[f], x, y, cfg) for x, y in pl]
        res = make_crops(img, bpx, players_px, rng, size, box)
        if res is None:
            continue
        pos, lab, neg = res
        s = split[f]
        for tag, crop in (("p", pos), ("n", neg)):
            if crop is None:
                continue
            name = f"f{f:06d}_{tag}"
            cv2.imwrite(os.path.join(out, "images", s, name + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(out, "labels", s, name + ".txt"), "w") as fh:
                fh.write(lab if tag == "p" else "")
            n[s] += 1
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--val-start", type=int, default=168000, help="frames from here on are validation only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import cv2
    from .io import load_tactical
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    hom = pd.read_csv(f"{t}/BAR-ATH_tactical_homography.csv")
    Hm = np.full((cfg.n_frames, 9), np.nan); Hm[hom["frame"].to_numpy()] = hom[HCOLS].to_numpy()
    hvalid = np.zeros(cfg.n_frames, bool); hvalid[hom["frame"].to_numpy()] = hom["valid"].to_numpy() == 1
    n = build(cv2.VideoCapture(a.video), ball, P, Hm, hvalid, cfg, a.out, a.n_train, a.n_val, a.val_start, a.seed)
    print(f"[ft] wrote {n['train']} train and {n['val']} val crops to {a.out}; data.yaml ready")


if __name__ == "__main__":
    main()
