import numpy as np

from .config import Config

H, A, D = 0, 1, 2
LETTER = np.array(["H", "A", "D"])


def viterbi(scores, cost):
    n, k = scores.shape
    back = np.zeros((n, k), np.int8)
    dp = scores[0].copy()
    for t in range(1, n):
        cand = dp[:, None] - cost
        back[t] = cand.argmax(0)
        dp = cand.max(0) + scores[t]
    path = np.empty(n, np.int8)
    path[-1] = dp.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _rle(lab):
    ch = np.where(np.diff(lab) != 0)[0] + 1
    st = np.concatenate([[0], ch])
    en = np.concatenate([ch, [len(lab)]])
    return [[int(s), int(e), int(lab[s])] for s, e in zip(st, en)]


def merge_short(lab, cfg: Config):
    """Absorb too-short runs into their longer neighbour (kills flicker)."""
    runs = _rle(lab)
    while True:
        bad = [(r[1] - r[0], i) for i, r in enumerate(runs)
               if (r[1] - r[0]) < (cfg.min_dead if r[2] == D else cfg.min_possession)]
        if not bad or len(runs) == 1:
            break
        _, i = min(bad)
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i < len(runs) - 1 else None
        if left is None or (right is not None and (right[1] - right[0]) > (left[1] - left[0])):
            right[0] = runs[i][0]
        else:
            left[1] = runs[i][1]
        del runs[i]
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged
    out = np.empty(len(lab), np.int8)
    for s, e, v in runs:
        out[s:e] = v
    return out


def _drop_short_true(mask, min_len):
    out = mask.copy()
    ch = np.where(np.diff(mask.astype(np.int8)) != 0)[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(mask)]])
    for a, b in zip(st, en):
        if mask[a] and b - a < min_len:
            out[a:b] = False
    return out


def decode(phi, delta, cfg: Config):
    """Two stages, so a stoppage can never be used as a cheap shortcut between H and A.

    1) alive / dead  (Viterbi, cost = dead_switch_cost, evidence = delta)
    2) H / A on the alive frames only, dead stretches cut out (cost = switch_cost)
    Full-match labels, 'X' outside the graded blocks."""
    out = np.full(len(phi), "X", dtype="<U1")
    c_dead = np.array([[0.0, cfg.dead_switch_cost], [cfg.dead_switch_cost, 0.0]])
    c_team = np.array([[0.0, cfg.switch_cost], [cfg.switch_cost, 0.0]])
    for a, b in cfg.blocks:
        n = b - a + 1
        dead = viterbi(np.stack([np.zeros(n), delta[a:b + 1]], 1), c_dead) == 1
        dead = _drop_short_true(dead, cfg.min_dead)
        lab = np.full(n, D, np.int8)
        alive = ~dead
        if alive.any():
            p = phi[a:b + 1][alive]
            lab[alive] = viterbi(np.stack([p, -p], 1), c_team)
        out[a:b + 1] = LETTER[merge_short(lab, cfg)]
    for fa, fb in cfg.forced_dead:
        out[fa:fb + 1] = "D"
    return out
