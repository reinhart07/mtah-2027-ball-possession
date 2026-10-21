"""Use the BROADCAST tracking to fill the tactical ball gaps.

The broadcast ATD lives on its own timeline (cuts, replays) but both trackings are in the same normalised
pitch coordinates. So we align them by the BALL TRAJECTORY itself: for each 1,500-frame chunk of the tactical
timeline, find the broadcast start index whose ball positions agree best (masked mean squared distance via
FFT cross-correlation). A chunk is accepted only if its best match is tight AND clearly better than the
runner-up. Accepted chunks lend their broadcast ball to tactical frames where the tactical ball is missing.

    python -m mtah.sync_broadcast --data data        # prints aggregate alignment / coverage statistics only
"""
import argparse
import os

import numpy as np
from scipy.signal import fftconvolve

from .config import Config

NO = np.iinfo(np.int32).min


def load_ball(meta, raw, cfg: Config, n=None):
    from kloppy import metrica
    df = metrica.load_tracking_epts(meta_data=meta, raw_data=raw).to_df()
    vid = df["frame_id"].to_numpy().astype(int) - 1
    n = n or int(vid.max()) + 1
    ball = np.full((n, 2), np.nan)
    ok = (vid >= 0) & (vid < n)
    ball[vid[ok], 0] = df["ball_x"].to_numpy()[ok] * cfg.pitch_l
    ball[vid[ok], 1] = df["ball_y"].to_numpy()[ok] * cfg.pitch_w
    return ball


def _corr(x, k):
    """out[s] = sum_i k[i] * x[s + i]"""
    return fftconvolve(x, k[::-1], mode="valid")


def masked_cost(a, b, min_pairs=300):
    """mean squared distance between chunk a (T,2) and every window of b (N,2); NaN = missing."""
    ma, mb = ~np.isnan(a[:, 0]), ~np.isnan(b[:, 0])
    A, B = np.where(ma[:, None], a, 0.0), np.where(mb[:, None], b, 0.0)
    cnt = _corr(mb.astype(float), ma.astype(float))
    ssd = (_corr(mb.astype(float), (A ** 2).sum(1)) + _corr((B ** 2).sum(1), ma.astype(float))
           - 2 * (_corr(B[:, 0], A[:, 0]) + _corr(B[:, 1], A[:, 1])))
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = np.clip(ssd, 0, None) / cnt
    cost[cnt < min_pairs] = np.inf
    return cost


def align(ball_t, ball_b, chunk=1500, accept_cost=4.0, ratio=0.6, tail=0.35, min_valid=300):
    """offset[t]: broadcast index = t + offset[t] (NO if the chunk was not accepted)."""
    N = len(ball_t)
    off = np.full(N, NO, dtype=np.int64)
    n_try = n_ok = 0
    for t0 in range(0, N - chunk + 1, chunk):
        a = ball_t[t0:t0 + chunk]
        if (~np.isnan(a[:, 0])).sum() < min_valid or len(ball_b) <= chunk:
            continue
        cost = masked_cost(a, ball_b)
        if not np.isfinite(cost).any():
            continue
        n_try += 1
        s = int(np.argmin(cost)); best = cost[s]
        c2 = cost.copy(); c2[max(0, s - 100):s + 101] = np.inf
        p1 = np.percentile(cost[np.isfinite(cost)], 1)      # a real match is a far outlier of the whole cost curve
        if best < accept_cost and best < ratio * c2.min() and best < tail * p1:
            off[t0:t0 + chunk] = s - t0
            n_ok += 1
    return off, n_try, n_ok


def _variants(ball_b, cfg):
    fx, fy = lambda b: np.c_[cfg.pitch_l - b[:, 0], b[:, 1]], lambda b: np.c_[b[:, 0], cfg.pitch_w - b[:, 1]]
    return {"as-is": ball_b, "flip-x": fx(ball_b), "flip-y": fy(ball_b), "flip-xy": fx(fy(ball_b))}


def pick_variant(ball_t, ball_b, cfg, n_chunks=8, chunk=1500):
    starts = np.linspace(cfg.blocks[0][0], cfg.blocks[1][1] - chunk, n_chunks).astype(int)
    res = {}
    for name, bb in _variants(ball_b, cfg).items():
        costs = []
        for t0 in starts:
            c = masked_cost(ball_t[t0:t0 + chunk], bb)
            if np.isfinite(c).any():
                costs.append(c.min())
        res[name] = float(np.median(costs)) if costs else np.inf
    best = min(res, key=res.get)
    print("[sync] orientation test (median best chunk cost, m^2):",
          {k: round(v, 2) for k, v in res.items()}, "-> using", best)
    return best, _variants(ball_b, cfg)[best]


def fuse_ball(ball_t, ball_b, off):
    t = np.arange(len(ball_t))
    bi = np.where(off != NO, t + off, -1)
    inb = (bi >= 0) & (bi < len(ball_b))
    cand = np.full_like(ball_t, np.nan)
    cand[inb] = ball_b[bi[inb]]
    use = np.isnan(ball_t[:, 0]) & ~np.isnan(cand[:, 0])
    fused = ball_t.copy()
    fused[use] = cand[use]
    return fused, use, cand


def report(ball_t, fused, use, cand, off, n_try, n_ok, cfg):
    g = np.concatenate([np.arange(a, b + 1) for a, b in cfg.blocks])
    vis0 = float((~np.isnan(ball_t[g, 0])).mean()); vis1 = float((~np.isnan(fused[g, 0])).mean())
    both = ~np.isnan(ball_t[:, 0]) & ~np.isnan(cand[:, 0])
    err = np.hypot(ball_t[both, 0] - cand[both, 0], ball_t[both, 1] - cand[both, 1])
    print(f"[sync] chunks tried {n_try}, accepted {n_ok} ({n_ok / max(1, n_try):.0%}); "
          f"graded frames covered by an accepted chunk: {float((off[g] != NO).mean()):.0%}")
    if both.sum():
        print(f"[sync] where both balls are seen ({int(both.sum())} frames): median error {np.median(err):.2f} m, "
              f"p90 {np.percentile(err, 90):.2f} m")
    print(f"[sync] graded ball visibility: tactical only {vis0:.1%} -> with broadcast {vis1:.1%} "
          f"(+{int(use[g].sum())} frames)")


def fuse_from_files(data_dir, ball_t, cfg: Config, verbose=True):
    b = os.path.join(data_dir, "broadcast")
    ball_b = load_ball(f"{b}/BAR-ATH_broadcast_atd.xml", f"{b}/BAR-ATH_broadcast_atd.txt", cfg)
    _, ball_b = pick_variant(ball_t, ball_b, cfg)
    off, n_try, n_ok = align(ball_t, ball_b)
    fused, use, cand = fuse_ball(ball_t, ball_b, off)
    if verbose:
        report(ball_t, fused, use, cand, off, n_try, n_ok, cfg)
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    a = ap.parse_args()
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball_t = load_ball(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg, cfg.n_frames)
    fuse_from_files(a.data, ball_t, cfg)


if __name__ == "__main__":
    main()
