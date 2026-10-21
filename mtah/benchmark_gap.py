"""Does the player-only model help when the ball is hidden?  (no ground truth involved)

Hides 30% of the visible ball frames in windows of N seconds, then compares, against our own un-masked
output, the plain decoder (hold the last state, w=0) with decoder + player-only evidence.
Also prints how long the REAL ball-less gaps are (aggregate numbers only).

    python -m mtah.benchmark_gap --data data
"""
import argparse
import os
from dataclasses import replace

import numpy as np

from .ball import clean_ball, runs
from .benchmark import mask_ball, mask_ball_mix
from .config import Config
from .gapmodel import build_features, crossfit_logit
from .io import load_tactical
from .metrics import spell_f1
from .pipeline import predict


def real_gap_stats(ball, cfg):
    b, _ = clean_ball(ball, cfg)
    lens = []
    for a, e in cfg.blocks:
        lens += [r1 - r0 for r0, r1 in runs(np.isnan(b[a:e + 1, 0]))]
    lens = np.array(lens)
    tot = max(1, lens.sum())
    print(f"real ball-less gaps in graded frames: {len(lens)} gaps, {tot} frames")
    for s in (3, 10, 30):
        print(f"  share of ball-less frames inside gaps >= {s:>2}s: {lens[lens >= s * cfg.fps].sum() / tot:.2f}")
    print(f"  median gap {np.median(lens) / cfg.fps:.1f}s, p90 {np.percentile(lens, 90) / cfg.fps:.1f}s")
    return lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--weights", type=float, nargs="+", default=[0.03, 0.1])
    ap.add_argument("--conf", type=float, nargs="+", default=[1.0])
    ap.add_argument("--gaps", type=int, nargs="+", default=[4, 15, 30])
    ap.add_argument("--mix", action="store_true", help="realistic gap lengths, several seeds, paired differences")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--tags", action="store_true", help="add Smart Tagging team features to the gap model")
    a = ap.parse_args()
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    lens = real_gap_stats(ball, cfg)
    ref = predict(ball, P, team, cfg)
    extra = None
    if a.tags:
        from .tags import load_tag_features
        extra = load_tag_features(f"{t}/BAR-ATH_tactical_smart_tagging.xml", cfg)
    logit = crossfit_logit(build_features(P, team, cfg, extra), ref, ball, cfg)
    g = np.concatenate([np.arange(x, y + 1) for x, y in cfg.blocks])
    if a.mix:
        long_lens = lens[lens >= cfg.gap_apply_min]
        variants = {"viterbi": (replace(cfg, gap_mode="viterbi", gap_w=0.1, gap_conf=1.0), logit),
                    "cp": (replace(cfg, gap_mode="cp"), logit), "mid": (replace(cfg, gap_mode="mid"), None)}
        d = {k: ([], []) for k in variants}
        print("\nseed  method    spellF1  frame_agree   (vs our un-masked output; gaps ~ real length distribution)")
        for sd in a.seeds:
            masked = mask_ball_mix(ball, cfg, long_lens, 0.3, sd)
            base = predict(masked, P, team, cfg)
            bs, bf = spell_f1(base, ref, cfg.blocks), (base[g] == ref[g]).mean()
            print(f"{sd:>4}  hold      {bs:.3f}    {bf:.3f}")
            for name, (c2, lg) in variants.items():
                lab = predict(masked, P, team, c2, gap_logit=lg)
                ss, ff = spell_f1(lab, ref, cfg.blocks), (lab[g] == ref[g]).mean()
                d[name][0].append(ss - bs); d[name][1].append(ff - bf)
                print(f"{sd:>4}  {name:<8}  {ss:.3f}    {ff:.3f}")
        print("\npaired difference vs hold, mean +- sd over seeds:")
        for name, (ds, df_) in d.items():
            print(f"  {name:<8} spellF1 {np.mean(ds):+.3f} +- {np.std(ds, ddof=1):.3f}   frame agreement {np.mean(df_):+.3f} +- {np.std(df_, ddof=1):.3f}")
        return
    print("\ngap  weight  conf  spellF1_vs_unmasked  frame_agreement")
    for secs in a.gaps:
        masked = mask_ball(ball, cfg, secs * cfg.fps, 0.3, 0)
        base = predict(masked, P, team, cfg)
        print(f"{secs:>2}s   hold   -    {spell_f1(base, ref, cfg.blocks):.3f}                {(base[g] == ref[g]).mean():.3f}")
        for c in a.conf:
            for w in a.weights:
                lab = predict(masked, P, team, replace(cfg, gap_mode="viterbi", gap_w=w, gap_conf=c), gap_logit=logit)
                print(f"{secs:>2}s  {w:5.2f}  {c:4.1f}  {spell_f1(lab, ref, cfg.blocks):.3f}                {(lab[g] == ref[g]).mean():.3f}")
        rows = [("mid", replace(cfg, gap_mode="mid"), None),
                ("cp", replace(cfg, gap_mode="cp"), logit)]
        if a.tags:
            from .tags import tag_score
            rows.append(("tagcp", replace(cfg, gap_mode="cp"), np.clip(1.5 * tag_score(extra), -3, 3)))
        for name, c2, lg in rows:
            lab = predict(masked, P, team, c2, gap_logit=lg)
            print(f"{secs:>2}s  {name:>5}   -    {spell_f1(lab, ref, cfg.blocks):.3f}                {(lab[g] == ref[g]).mean():.3f}")


if __name__ == "__main__":
    main()
