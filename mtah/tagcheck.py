"""How reliable are the Smart Tagging team tags as a possession signal, and do they cover the ball-less gaps?
Aggregates only.

    python -m mtah.tagcheck --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from .config import Config
from .io import load_tactical
from .pipeline import predict
from .tags import load_tag_features, tag_score


def acc(score, lab, mask, name):
    m = mask & (score != 0) & np.isin(lab, ["H", "A"])
    if m.sum() == 0:
        print(f"  {name:<34} no frames"); return
    ok = (score[m] > 0) == (lab[m] == "H")
    print(f"  {name:<34} frames {int(m.sum()):>6}  agree {ok.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    a = ap.parse_args()
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    ref = predict(ball, P, team, cfg)
    F = load_tag_features(f"{t}/BAR-ATH_tactical_smart_tagging.xml", cfg)
    sc = tag_score(F)
    graded = np.zeros(cfg.n_frames, bool)
    for x, y in cfg.blocks:
        graded[x:y + 1] = True
    noball = graded & np.isnan(ball[:, 0])
    seen = graded & ~np.isnan(ball[:, 0])
    print(f"graded frames with a team-bearing offensive/defensive tag: {(graded & (sc != 0)).sum() / graded.sum():.1%}")
    print(f"  among ball-less frames: {(noball & (sc != 0)).sum() / max(1, noball.sum()):.1%}; "
          f"conflicting tags (both teams): {(graded & (F[:, :4].sum(1) > 0) & (sc == 0)).sum() / graded.sum():.1%}")
    print("agreement of the tag signal with OUR ball-based labels (not ground truth):")
    acc(sc, ref, seen, "all, ball visible")
    acc(F[:, 0] - F[:, 1], ref, seen, "offensive tags only")
    acc(F[:, 3] - F[:, 2], ref, seen, "defensive tags only")
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    print("agreement with the PUBLIC ground truth (5 min, sanity only):")
    acc(sc, gt, gt != "X", "tag signal vs public GT")


if __name__ == "__main__":
    main()
