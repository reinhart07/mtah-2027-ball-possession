import numpy as np
import pandas as pd

from .config import Config


def clean_ball(ball, cfg: Config):
    """Drop tracker jumps, interpolate short gaps. Returns (ball, speed)."""
    b = ball.copy()
    idx = np.where(~np.isnan(b[:, 0]))[0]
    if len(idx) > 1:
        d = np.hypot(np.diff(b[idx, 0]), np.diff(b[idx, 1]))
        v = d / (np.diff(idx) / cfg.fps)
        bad = idx[1:][v > cfg.max_ball_speed]
        b[bad] = np.nan
    for k in (0, 1):
        s = pd.Series(b[:, k]).interpolate(limit=cfg.max_interp_gap, limit_area="inside")
        b[:, k] = s.to_numpy()
    k = 2
    sp = np.full(len(b), np.nan)
    sp[k:-k] = np.hypot(b[2 * k:, 0] - b[:-2 * k, 0], b[2 * k:, 1] - b[:-2 * k, 1]) / (2 * k / cfg.fps)
    return b, sp


def runs(mask):
    """(start, end_exclusive) of True-runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
