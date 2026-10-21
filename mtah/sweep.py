"""Sensitivity sweep. Pick a flat PLATEAU, never the single best cell (EVALUATION.md):
the public window has only 17 possessions, one possession moves spellF1 by ~6 points.

    python -m mtah.sweep --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv

Caution: the public window is the first 5 minutes, where the ball is missing on far fewer frames than
in the rest of the match, so it flatters every setting. Compare settings with each other, not with 1.0.
"""
import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from .ball import clean_ball
from .config import Config
from .decode import decode
from .evidence import control_evidence, dead_evidence, player_activity_smooth
from .io import load_tactical
from .metrics import possessions, spell_f1


def macro_f1(pred, gt, a, b):
    f = []
    for k in "HAD":
        tp = ((pred[a:b + 1] == k) & (gt[a:b + 1] == k)).sum()
        p = tp / max(1, (pred[a:b + 1] == k).sum()); r = tp / max(1, (gt[a:b + 1] == k).sum())
        f.append(0 if tp == 0 else 2 * p * r / (p + r))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--switch", type=float, nargs="+", default=[12, 24, 40])
    ap.add_argument("--radius", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--calm", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    a = ap.parse_args()
    cfg0 = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg0)
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    idx = np.where(gt != "X")[0]; ga, gb = int(idx.min()), int(idx.max())
    gt_poss = len(possessions(gt, (ga, gb)))
    act = player_activity_smooth(P, cfg0)
    b, speed = clean_ball(ball, cfg0)
    rows = []
    for rad in a.radius:
        phi = control_evidence(b, P, team, replace(cfg0, ctrl_radius=rad))
        for calm in a.calm:
            cfg = replace(cfg0, ctrl_radius=rad, calm_bonus=calm)
            delta, _ = dead_evidence(b, speed, cfg, act)
            for sc in a.switch:
                c = replace(cfg, switch_cost=sc)
                lab = decode(phi, delta, c)
                pp = possessions(lab, (ga, gb))
                full = [p for blk in c.blocks for p in possessions(lab, blk)]
                g, p_ = gt[ga:gb + 1], lab[ga:gb + 1]
                graded = np.concatenate([lab[x:y + 1] for x, y in c.blocks])
                rows.append(dict(radius=rad, calm=calm, switch=sc,
                                 pubSpell=round(spell_f1(lab, gt, [(ga, gb)]), 3),
                                 pubMacro=round(macro_f1(lab, gt, ga, gb), 3),
                                 pub_poss=len(pp),
                                 D_rec=round(float(((p_ == "D") & (g == "D")).sum() / max(1, (g == "D").sum())), 2),
                                 D_prec=round(float(((p_ == "D") & (g == "D")).sum() / max(1, (p_ == "D").sum())), 2),
                                 full_poss=len(full),
                                 full_Dshare=round(float((graded == "D").mean()), 2)))
    df = pd.DataFrame(rows)
    print(f"public GT: {gt_poss} possessions in 5 min (~{gt_poss * 106 / 5:.0f} per full match at that rate); "
          f"D share in the public window {(gt[ga:gb + 1] == 'D').mean():.2f}")
    print(df.to_string(index=False))
    print("\nread it as a surface: choose a region where spellF1 is flat, not its peak.")


if __name__ == "__main__":
    main()
