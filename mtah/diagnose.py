"""Understand the data and the label definition on the PUBLIC 5 minutes (allowed: RULES.md).

    python -m mtah.diagnose --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv

1. peeks at the raw ATD files (is there a ball status / owner channel?)
2. confusion matrix of our labels vs the public GT
3. what the GT calls "dead": ball position / speed at the start and end of every D stretch
4. possession lists, GT vs ours, to see whether we flip on touches the definition ignores
Sanity checking only - never pick hyper-parameters by peaking on this window (EVALUATION.md).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from .ball import clean_ball
from .config import Config
from .evidence import _boundary_dist
from .metrics import spell_f1


def peek(meta, raw, n=3):
    print("== raw ATD peek ==")
    with open(raw, "r", errors="replace") as f:
        for _ in range(n):
            print(f.readline()[:400].rstrip())
    with open(meta, "r", errors="replace") as f:
        hits = [l.strip()[:200] for l in f if re.search(r"ball|status|owner|alive|dead", l, re.I)]
    print(f"-- {len(hits)} metadata lines mention ball/status/owner/alive/dead; first 15:")
    print("\n".join(hits[:15]))


def rle(lab, a):
    ch = np.where(lab[1:] != lab[:-1])[0] + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(lab)]])
    return [(a + int(s), a + int(e), lab[s]) for s, e in zip(st, en)]


def compare(pred, gt, ball_raw, cfg: Config, verbose=False):
    idx = np.where(gt != "X")[0]
    a, b = idx.min(), idx.max()
    g, p = gt[a:b + 1], pred[a:b + 1]
    print(f"\n== window {a}..{b} ({b - a + 1} frames) ==")
    print(pd.crosstab(pd.Series(g, name="GT"), pd.Series(p, name="ours")))
    print("frame agreement:", round(float((g == p).mean()), 3),
          "| internal spellF1:", round(spell_f1(pred, gt, [(a, b)]), 3))

    ball, speed = clean_ball(ball_raw, cfg)
    print("\n== GT dead-ball stretches ==")
    for s, e, l in rle(g, a):
        if l != "D":
            continue
        if not verbose:
            ours = float((p[s - a:e - a] == "D").mean())
            print(f"D {(e - s) / cfg.fps:.1f}s  our D coverage {ours:.2f}")
            continue
        def st(i):
            i = min(max(i, 0), len(ball) - 1)
            if np.isnan(ball[i, 0]):
                return "ball NaN"
            return f"xy=({ball[i,0]:.1f},{ball[i,1]:.1f}) edge={_boundary_dist(ball[i], cfg):.1f}m v={speed[i]:.1f}"
        print(f"D {s}-{e} ({(e - s) / cfg.fps:.1f}s) start[{st(s)}] end[{st(e - 1)}]")

    print("\n== possessions GT vs ours ==")
    for name, arr in (("GT", g), ("ours", p)):
        segs = [(s, e, l) for s, e, l in rle(arr, a) if l in "HA"]
        print(name, len(segs), "possessions:",
              " ".join(f"{l}{(e - s) / cfg.fps:.0f}s" for s, e, l in segs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--verbose", action="store_true", help="LOCAL ONLY: prints raw rows/coordinates - never paste into a hosted LLM")
    a = ap.parse_args()
    from .io import load_tactical
    from .pipeline import predict
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    meta, raw = f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt"
    if a.verbose:
        peek(meta, raw)
    ball, P, team = load_tactical(meta, raw, cfg)
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    compare(predict(ball, P, team, cfg), gt, ball, cfg, a.verbose)


if __name__ == "__main__":
    main()
