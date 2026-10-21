"""Sanity test on a simulated match (no organiser data needed): proves the code runs end to end
and that the decoder recovers possessions when 40% of ball frames are missing."""
import numpy as np

from mtah.config import Config
from mtah.metrics import spell_f1, summary
from mtah.pipeline import predict


def simulate(cfg, seed=0, miss=0.40):
    rng = np.random.default_rng(seed)
    n = cfg.n_frames
    S = 22
    team = np.array([0] * 11 + [1] * 11)
    pos = rng.uniform([5, 5], [100, 63], (S, 2))
    P = np.empty((n, S, 2))
    for t in range(n):
        pos = np.clip(pos + rng.normal(0, 0.05, pos.shape), 0, [cfg.pitch_l, cfg.pitch_w])
        P[t] = pos
    truth = np.full(n, "X", dtype="<U1")
    ball = np.full((n, 2), np.nan)
    for a, b in cfg.blocks:
        t, side = a, 0
        while t <= b:
            dur = int(rng.uniform(8, 20) * cfg.fps)
            end = min(b + 1, t + dur)
            truth[t:end] = "HA"[side]
            mates = np.where(team == side)[0]
            u = t
            while u < end:
                holder = rng.choice(mates)
                hold = int(rng.uniform(1, 3) * cfg.fps)
                v = min(end, u + hold)
                ball[u:v] = P[u:v, holder] + rng.normal(0, 0.4, (v - u, 2))
                nxt = rng.choice(mates)
                w = min(end, v + 15)
                if w > v:
                    ball[v:w] = np.linspace(P[v, holder], P[min(w, end - 1), nxt], w - v)
                u = w
            t, side = end, 1 - side
    # missing ball: random windows of 1-6 s
    for a, b in cfg.blocks:
        hidden, goal = 0, miss * (b - a)
        while hidden < goal:
            s = int(rng.integers(a, b - 150)); L = int(rng.uniform(1, 6) * cfg.fps)
            ball[s:s + L] = np.nan; hidden += L
    return ball, P, team, truth


def test_pipeline_runs_and_recovers():
    cfg = Config(calm_bonus=0.0)
    ball, P, team, truth = simulate(cfg)
    pred = predict(ball, P, team, cfg)
    assert len(pred) == cfg.n_frames
    assert set(np.unique(pred)) <= set("HADX")
    assert (pred[:cfg.blocks[0][0]] == "X").all()
    score = spell_f1(pred, truth, cfg.blocks)
    print("synthetic spellF1:", round(score, 3), summary(pred, cfg.blocks))
    assert score > 0.3


if __name__ == "__main__":
    test_pipeline_runs_and_recovers()
