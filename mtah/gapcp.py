"""Place possession hand-overs inside long ball-less gaps.

Pass 1 (ball-based decoder) knows the team BEFORE a gap and the team AFTER it (the ball reappears).
If they differ, a hand-over happened somewhere inside, but the plain decoder puts it at the gap's end.
If the true time is unknown, the midpoint halves the expected timing error (the median minimises expected
absolute error). `logit` (optional) adds evidence about WHERE: the split point k maximises
    normalised_agreement_with_logit(k) + beta * midpoint_prior(k).
Gaps whose two sides agree are left alone.
"""
import numpy as np

from .ball import runs
from .config import Config


def relocate_flips(lab, valid, logit, cfg: Config, snap=25):
    out = lab.copy()
    l_all = None if logit is None else np.nan_to_num(logit)
    for a, b in cfg.blocks:
        for r0, r1 in runs(~valid[a:b + 1]):
            s, e = a + r0, a + r1
            if e - s < cfg.gap_apply_min or s == a or e > b:
                continue
            L, R = lab[s - 1], lab[e]
            if L not in ("H", "A") or R not in ("H", "A") or L == R:
                continue
            idx = s + np.where(lab[s:e] != "D")[0]
            n = len(idx)
            if n < 2:
                continue
            if l_all is None:
                k = n // 2
            else:
                sgn = 1.0 if L == "H" else -1.0           # logit > 0 means HOME
                l = l_all[idx]
                cum = np.concatenate([[0.0], np.cumsum(sgn * l)])
                z = (cum - (cum[-1] - cum)) / (np.abs(l).sum() + 1e-9)
                prior = -np.abs(np.arange(n + 1) - n / 2) / (n / 2)
                k = int(np.argmax(z + cfg.gap_beta * prior))
            if k < snap:
                k = 0
            elif n - k < snap:
                k = n
            out[idx[:k]] = L
            out[idx[k:]] = R
    return out
