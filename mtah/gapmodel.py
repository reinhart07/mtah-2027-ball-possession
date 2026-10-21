"""Player-only possession evidence, learned WITHOUT ground truth.

Where the ball is visible and the ball-based decoder is confident, we know (approximately) which team has
possession. We learn to predict that from the 22 players alone (team shape, movement, spread), with
time-blocked cross-fitting so a frame is never scored by a model that trained on its neighbourhood.
The prediction is then used as weak evidence on frames where the ball is NOT seen.
"""
import warnings

import numpy as np
import pandas as pd

from .config import Config


def _roll(a, w):
    return pd.DataFrame(a).rolling(w, center=True, min_periods=max(3, w // 5)).mean().to_numpy()


def build_features(P, team, cfg: Config, extra=None):
    N, w = len(P), cfg.fps
    d = np.hypot(P[w:, :, 0] - P[:-w, :, 0], P[w:, :, 1] - P[:-w, :, 1])
    spd = np.full(P.shape[:2], np.nan)
    spd[w // 2:w // 2 + len(d)] = d
    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in (0, 1):
            m = team == k
            x, y = P[:, m, 0], P[:, m, 1]
            cols += [np.nanmean(x, 1), np.nanmean(y, 1), np.nanstd(x, 1), np.nanstd(y, 1), np.nanmean(spd[:, m], 1)]
    F = _roll(np.column_stack(cols), w)                 # 0-4 home cx,cy,sx,sy,speed ; 5-9 away
    vel = np.full((N, 4), np.nan)
    for j, c in enumerate([0, 1, 5, 6]):                 # centroid velocity over +-1 s, m/s
        vel[w:-w, j] = (F[2 * w:, c] - F[:-2 * w, c]) / 2.0
    half = np.zeros(N)
    half[cfg.blocks[1][0]:] = 1.0
    cols = [F, vel, F[:, 5] - F[:, 0], F[:, 6] - F[:, 1], half]
    if extra is not None:
        cols.append(extra)
    return np.column_stack(cols)


def crossfit_logit(F, ref, ball, cfg: Config, chunk=12_000, k=4, seed=0, verbose=True):
    """log-odds that HOME has possession, from players only. Clipped to [-3, 3]."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    N = len(F)
    valid = ~np.isnan(ball[:, 0])
    dense = pd.Series(valid.astype(float)).rolling(51, center=True, min_periods=1).min().to_numpy() == 1
    okF = ~np.isnan(F).any(1)
    use = dense & np.isin(ref, ["H", "A"]) & okF
    y = (ref == "H").astype(int)
    fold = (np.arange(N) // chunk) % k
    logit = np.full(N, np.nan)
    for f in range(k):
        tr, te = use & (fold != f), fold == f
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(F[tr], y[tr])
        ok = te & okF
        p = np.clip(m.predict_proba(F[ok])[:, 1], 1e-3, 1 - 1e-3)
        logit[ok] = np.log(p / (1 - p))
        ev = te & use
        if verbose and ev.sum() > 500 and len(set(y[ev])) == 2:
            print(f"[gap] fold {f}: held-out AUC {roc_auc_score(y[ev], logit[ev]):.3f}  "
                  f"acc {((logit[ev] > 0) == y[ev]).mean():.3f}  "
                  f"(always-majority {max(y[ev].mean(), 1 - y[ev].mean()):.3f}, n={ev.sum()})")
    return np.clip(logit, -3, 3)
