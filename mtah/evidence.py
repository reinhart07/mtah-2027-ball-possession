import warnings

import numpy as np
import pandas as pd

from .ball import runs
from .config import Config

CAP = 30.0


def control_evidence(ball, P, team, cfg: Config):
    """phi > 0 -> home controls the ball, < 0 -> away, ~0 -> nobody close / ball unseen."""
    valid = ~np.isnan(ball[:, 0])
    d = np.hypot(P[:, :, 0] - ball[:, None, 0], P[:, :, 1] - ball[:, None, 1])
    d = np.minimum(np.nan_to_num(d, nan=CAP), CAP)
    dh = d[:, team == 0].min(1)
    da = d[:, team == 1].min(1)
    llr = np.clip((da - dh) / cfg.ctrl_sigma, -cfg.llr_clip, cfg.llr_clip)
    gate = np.exp(-(np.minimum(dh, da) / cfg.ctrl_radius) ** 2)
    phi = gate * llr
    phi[~valid] = 0.0
    return phi


def _out_of_bounds(b, cfg):
    m = cfg.out_margin
    return (b[:, 0] < -m) | (b[:, 0] > cfg.pitch_l + m) | (b[:, 1] < -m) | (b[:, 1] > cfg.pitch_w + m)


def _boundary_dist(xy, cfg):
    x, y = xy
    return min(x, cfg.pitch_l - x, y, cfg.pitch_w - y)


def player_activity_smooth(P, cfg: Config):
    """median over tracked players of speed (1 s displacement), then smoothed."""
    win = cfg.fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.hypot(P[win:, :, 0] - P[:-win, :, 0], P[win:, :, 1] - P[:-win, :, 1])
        ms = np.nanmedian(d, axis=1)
    out = np.full(len(P), np.nan)
    out[win // 2:win // 2 + len(ms)] = ms
    return pd.Series(out).rolling(cfg.calm_win, center=True, min_periods=10).median().to_numpy()


def dead_evidence(ball, speed, cfg: Config, act=None):
    """delta: score for the dead-ball state. Returns (delta, static_mask)."""
    valid = ~np.isnan(ball[:, 0])
    delta = np.zeros(len(ball))
    delta[valid] = -cfg.live_dead_penalty
    delta[valid & _out_of_bounds(np.nan_to_num(ball, nan=0.0), cfg)] = cfg.out_bonus

    static = np.zeros(len(ball), bool)
    slow = np.nan_to_num(speed, nan=1e9) < cfg.static_speed
    for s, e in runs(slow):
        if e - s >= cfg.static_min_len:
            static[s:e] = True
    delta[static] = cfg.static_bonus

    if cfg.gap_dead_rule:
        for s, e in runs(~valid):
            if e - s < cfg.gap_min_len or s == 0 or e >= len(ball):
                continue
            if (_boundary_dist(ball[s - 1], cfg) <= cfg.gap_before_m and
                    _boundary_dist(ball[e], cfg) <= cfg.gap_after_m):
                delta[s:e] = cfg.out_bonus
    if act is not None and cfg.calm_bonus:
        sp = pd.Series(speed).rolling(cfg.calm_win, center=True, min_periods=10).median().to_numpy()
        slow_or_unseen = np.isnan(sp) | (sp < cfg.calm_ball_speed)
        delta[(np.nan_to_num(act, nan=9.0) < cfg.calm_thr) & slow_or_unseen] += cfg.calm_bonus
    return delta, static
