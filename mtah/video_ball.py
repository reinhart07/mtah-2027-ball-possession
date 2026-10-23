"""Ball candidates from the tactical video + temporal linking inside a ball-less gap.

A pretrained detector finds the ball somewhere in the frame about half of the time but returns ~7 candidates, so a
single detection is not trustworthy. The tracker DOES know where the ball was just before and just after a gap, and
the ball cannot teleport: we pick, per gap, the candidate path that starts at the last known position, ends at the
first known position, moves at plausible speeds and sits near players, allowing missed detections (skip edges).
"""
import glob
import logging
import os

import numpy as np

from .config import Config


def to_pitch(Hrow, xy_px, cfg: Config):
    """pixels -> pitch metres through the inverse of the pitch(0-1)->pixel homography."""
    Hi = np.linalg.inv(np.asarray(Hrow, float).reshape(3, 3))
    pts = np.c_[xy_px, np.ones(len(xy_px))] @ Hi.T
    return np.c_[pts[:, 0] / pts[:, 2] * cfg.pitch_l, pts[:, 1] / pts[:, 2] * cfg.pitch_w]


def tile_starts(length, tile, overlap):
    stride = tile - overlap
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def _merge(dets, radius):
    """greedy distance-NMS over (x, y, conf) tuples."""
    keep = []
    for x, y, c in sorted(dets, key=lambda d: -d[2]):
        if all(np.hypot(x - kx, y - ky) > radius for kx, ky, _ in keep):
            keep.append((x, y, c))
    if not keep:
        return np.zeros((0, 2)), np.zeros(0)
    k = np.array(keep)
    return k[:, :2], k[:, 2]


def detect_tiled(model, imgs, tile=640, overlap=128, conf=0.03, half=True, merge_px=8):
    """single-class detector run on overlapping native-resolution tiles (the ball stays ~10 px, no downscaling)."""
    tiles, meta = [], []
    for fi, img in enumerate(imgs):
        H, W = img.shape[:2]
        for y0 in tile_starts(H, tile, overlap):
            for x0 in tile_starts(W, tile, overlap):
                tiles.append(img[y0:y0 + tile, x0:x0 + tile]); meta.append((fi, x0, y0))
    res = model.predict(tiles, imgsz=tile, conf=conf, verbose=False, half=half)
    per = [[] for _ in imgs]
    for (fi, x0, y0), r in zip(meta, res):
        if len(r.boxes):
            xy = r.boxes.xywh.cpu().numpy()[:, :2]
            cf = r.boxes.conf.cpu().numpy()
            per[fi] += [(float(p[0] + x0), float(p[1] + y0), float(c)) for p, c in zip(xy, cf)]
    return [_merge(d, merge_px) for d in per]


def inside_pitch(pos, cfg: Config, margin=5.0):
    return ((pos[:, 0] >= -margin) & (pos[:, 0] <= cfg.pitch_l + margin) &
            (pos[:, 1] >= -margin) & (pos[:, 1] <= cfg.pitch_w + margin))


def detect_stretch(cap, model, a, b, step, imgsz=1920, conf=0.03, half=True, batch=8, tiled=False):
    """candidates (pixels, confidences) every `step` frames in [a, b]; sequential read, no random seeks."""
    import cv2
    logging.getLogger("ultralytics").setLevel(logging.ERROR)      # the "half is deprecated" warning repeats per call
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
    frames, imgs, out = [], [], []

    def flush():
        if not imgs:
            return
        if tiled:
            for f, (xy, cf) in zip(frames, detect_tiled(model, imgs, conf=conf, half=half)):
                out.append((f, xy, cf))
        else:
            res = model.predict(imgs, imgsz=imgsz, conf=conf, classes=[32], verbose=False, half=half)
            for f, r in zip(frames, res):
                xy = r.boxes.xywh.cpu().numpy()[:, :2] if len(r.boxes) else np.zeros((0, 2))
                cf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                out.append((f, xy, cf))
        frames.clear(); imgs.clear()

    for f in range(a, b + 1):
        if (f - a) % step == 0:
            ok, img = cap.read()
            if ok:
                frames.append(f); imgs.append(img)
                if len(imgs) == batch:
                    flush()
        else:
            cap.grab()
    flush()
    return out


def link(samples, f0, p0, f1, p1, fps=25, vmax=35.0, skip=5, w_miss=1.0, w_conf=2.0, w_player=1.0):
    """samples: list of (frame, pos_m (K,2), conf (K,), dmin_player (K,)) in time order.
    f0/p0: frame and position (m) of the last known ball BEFORE the gap; f1/p1: first known ball AFTER it.
    Returns [(frame, (x, y))] of the chosen candidates, or [] when 'no detection at all' is the cheapest story."""
    n = len(samples)
    nodes = []                                    # (sample index, candidate index)
    for i, (_, pos, cf, dm) in enumerate(samples):
        for k in range(len(pos)):
            nodes.append((i, k))
    if not nodes:
        return []

    def move(fa, pa, fb, pb, miss):
        dt = (fb - fa) / fps
        d = float(np.hypot(*(np.asarray(pb) - np.asarray(pa))))
        if d > vmax * dt + 1.0:
            return np.inf
        return (max(0.0, d - 0.5 * vmax * dt) / 3.0) ** 2 + w_miss * miss

    def node_cost(i, k):
        _, pos, cf, dm = samples[i]
        return w_conf * (1.0 - float(cf[k])) + w_player * min(float(dm[k]), 15.0) / 15.0

    best = {}
    back = {}
    for (i, k) in nodes:
        fi, pi = samples[i][0], samples[i][1][k]
        c0 = move(f0, p0, fi, pi, i)              # all earlier samples counted as misses
        bc, bp = c0, "start"
        for j in range(max(0, i - skip), i):
            for kj in range(len(samples[j][1])):
                if (j, kj) in best:
                    c = best[(j, kj)] + move(samples[j][0], samples[j][1][kj], fi, pi, i - j - 1)
                    if c < bc:
                        bc, bp = c, (j, kj)
        if np.isfinite(bc):
            best[(i, k)] = bc + node_cost(i, k)
            back[(i, k)] = bp
    end_best, end_node = w_miss * n + (0.0 if np.isfinite(move(f0, p0, f1, p1, 0)) else 1e9), None
    for (i, k), c in best.items():
        tot = c + move(samples[i][0], samples[i][1][k], f1, p1, n - 1 - i)
        if tot < end_best:
            end_best, end_node = tot, (i, k)
    if end_node is None:
        return []
    path, cur = [], end_node
    while cur != "start":
        i, k = cur
        path.append((samples[i][0], tuple(samples[i][1][k])))
        cur = back[cur]
    return path[::-1]


def fill_between(ball, path, f0, p0, f1, p1, max_gap=25):
    """write the linked path into `ball`, interpolating linearly between consecutive anchors closer than max_gap frames."""
    pts = [(f0, np.asarray(p0, float))] + [(f, np.asarray(p, float)) for f, p in path] + [(f1, np.asarray(p1, float))]
    for (fa, pa), (fb, pb) in zip(pts[:-1], pts[1:]):
        if fb - fa <= max_gap:
            for f in range(fa + 1, fb):
                ball[f] = pa + (pb - pa) * (f - fa) / (fb - fa)
    for f, p in path:
        ball[f] = np.asarray(p, float)
    return ball


def fuse_video_cache(ball, cache_dir):
    """fill NaN ball frames from the per-gap cache written by video_fill (derived data: never commit it)."""
    out, n = ball.copy(), 0
    for fn in sorted(glob.glob(os.path.join(cache_dir, "gap_*.npz"))):
        z = np.load(fn)
        fr, xy = z["frames"], z["xy"]
        m = np.isnan(out[fr, 0]) & ~np.isnan(xy[:, 0])
        out[fr[m]] = xy[m]
        n += int(m.sum())
    return out, n
