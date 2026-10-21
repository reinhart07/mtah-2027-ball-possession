"""Dead-ball detector sweep: we want high D precision (false stoppages split possessions and wreck
spellF1) with decent recall, and a plausible number of stoppages / D share over the full match.

    python -m mtah.sweep_dead --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
"""
import argparse
import itertools
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
from .sweep import macro_f1


def n_runs(lab, blocks, k="D"):
    return sum(int(((lab[a:b + 1] == k) & ~np.r_[False, lab[a:b] == k]).sum()) for a, b in blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    a = ap.parse_args()
    cfg0 = Config(calm_bonus=1.5)
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg0)
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    idx = np.where(gt != "X")[0]; ga, gb = int(idx.min()), int(idx.max())
    act = player_activity_smooth(P, cfg0)
    b, speed = clean_ball(ball, cfg0)
    phi = control_evidence(b, P, team, cfg0)
    rows = []
    for thr, dcost, mind, bsp in itertools.product([1.0, 1.3], [8, 24, 48], [12, 100], [99.0, 5.0]):
        cfg = replace(cfg0, calm_thr=thr, dead_switch_cost=dcost, min_dead=mind, calm_ball_speed=bsp)
        delta, _ = dead_evidence(b, speed, cfg, act)
        lab = decode(phi, delta, cfg)
        g, p = gt[ga:gb + 1], lab[ga:gb + 1]
        graded = np.concatenate([lab[x:y + 1] for x, y in cfg.blocks])
        rows.append(dict(calm_thr=thr, dead_cost=dcost, min_dead=mind, ball_max=bsp,
                         D_rec=round(float(((p == "D") & (g == "D")).sum() / max(1, (g == "D").sum())), 2),
                         D_prec=round(float(((p == "D") & (g == "D")).sum() / max(1, (p == "D").sum())), 2),
                         pub_Druns=n_runs(lab, [(ga, gb)]),
                         pubSpell=round(spell_f1(lab, gt, [(ga, gb)]), 3),
                         pubMacro=round(macro_f1(lab, gt, ga, gb), 3),
                         pub_poss=len(possessions(lab, (ga, gb))),
                         full_poss=sum(len(possessions(lab, blk)) for blk in cfg.blocks),
                         full_Druns=n_runs(lab, cfg.blocks),
                         full_Dshare=round(float((graded == "D").mean()), 2)))
    print(f"GT in public window: 4 D stretches, D share {(gt[ga:gb + 1] == 'D').mean():.2f}, 17 possessions")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
