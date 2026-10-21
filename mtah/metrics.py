"""Internal approximation of spellF1 (the official scorer in the organisers' repo is the authority)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def possessions(labels, block):
    a, b = block
    out, cur, start = [], None, a
    for i in range(a, b + 2):
        l = labels[i] if i <= b else None
        if l != cur:
            if cur in ("H", "A"):
                out.append((start, i, cur))
            cur, start = l, i
    return out


def spell_f1(pred, truth, blocks, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    P, T = [], []
    for blk in blocks:
        P += possessions(pred, blk)
        T += possessions(truth, blk)
    if not P or not T:
        return 0.0
    iou = np.zeros((len(P), len(T)))
    for i, (s1, e1, l1) in enumerate(P):
        for j, (s2, e2, l2) in enumerate(T):
            if l1 != l2:
                continue
            inter = max(0, min(e1, e2) - max(s1, s2))
            if inter:
                iou[i, j] = inter / (max(e1, e2) - min(s1, s2))
    f1s = []
    for th in thresholds:
        m = np.where(iou >= th, iou, 0.0)
        r, c = linear_sum_assignment(-m)
        tp = int((m[r, c] > 0).sum())
        p, rc = tp / len(P), tp / len(T)
        f1s.append(0.0 if tp == 0 else 2 * p * rc / (p + rc))
    return float(np.mean(f1s))


def summary(labels, blocks, fps=25):
    ps = [p for b in blocks for p in possessions(labels, b)]
    d = np.array([(e - s) / fps for s, e, _ in ps]) if ps else np.array([0.0])
    graded = np.concatenate([labels[a:b + 1] for a, b in blocks])
    frac = {k: round(float((graded == k).mean()), 3) for k in "HAD"}
    return {"possessions": len(ps), "mean_dur_s": round(float(d.mean()), 1), "label_share": frac}
