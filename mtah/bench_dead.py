"""Dead-ball decisions under occlusion (no ground truth needed).

About 40% of graded frames have no ball, and there the dead-ball call rests on the players alone. We hide
the ball in windows with the REAL gap-length distribution and ask: on the hidden frames, how often does our
own un-masked decision flip between alive and dead?   (reference = the same setting's un-masked output)

    python -m mtah.bench_dead --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
Aggregates only. Read the hidden-frame error rates TOGETHER with the public-window D precision/recall:
a setting that never says "dead" has zero self-inconsistency and is useless.
"""
import argparse
import itertools
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from .benchmark import mask_ball_mix
from .benchmark_gap import real_gap_stats
from .config import Config
from .io import load_tactical
from .pipeline import predict


def hidden_errors(ref, lab, ball, masked, cfg):
    graded = np.zeros(len(ref), bool)
    for a, b in cfg.blocks:
        graded[a:b + 1] = True
    hid = graded & np.isnan(masked[:, 0]) & ~np.isnan(ball[:, 0])
    r, p = ref[hid] == "D", lab[hid] == "D"
    fp = float((p & ~r).sum() / max(1, (~r).sum()))      # alive frames that become dead
    fn = float((~p & r).sum() / max(1, r.sum()))         # dead frames that become alive
    return fp, fn, float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--thr", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--cost", type=float, nargs="+", default=[24, 48])
    a = ap.parse_args()
    cfg0 = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg0)
    lens = real_gap_stats(ball, cfg0)
    long_lens = lens[lens >= cfg0.gap_apply_min]
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    idx = np.where(gt != "X")[0]; ga, gb = int(idx.min()), int(idx.max())
    masks = {sd: mask_ball_mix(ball, cfg0, long_lens, 0.3, sd) for sd in a.seeds}
    rows = []
    for thr, cost in itertools.product(a.thr, a.cost):
        cfg = replace(cfg0, calm_thr=thr, dead_switch_cost=cost)
        ref = predict(ball, P, team, cfg)
        g, p = gt[ga:gb + 1], ref[ga:gb + 1]
        tp = ((p == "D") & (g == "D")).sum()
        graded = np.concatenate([ref[x:y + 1] for x, y in cfg.blocks])
        fps, fns = [], []
        for sd, m in masks.items():
            fp, fn, _ = hidden_errors(ref, predict(m, P, team, cfg), ball, m, cfg)
            fps.append(fp); fns.append(fn)
        rows.append(dict(calm_thr=thr, dead_cost=cost,
                         pub_D_prec=round(float(tp / max(1, (p == "D").sum())), 2),
                         pub_D_rec=round(float(tp / max(1, (g == "D").sum())), 2),
                         full_Dshare=round(float((graded == "D").mean()), 2),
                         hidden_alive_to_D=round(float(np.mean(fps)), 3),
                         hidden_D_to_alive=round(float(np.mean(fns)), 3)))
    print("\nsetting                      public window (unmasked)     whole match        hidden frames (self-consistency)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()