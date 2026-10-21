"""python -m mtah.run --data data --out evaluation/results/Rei_JR.csv [--gap-w 0.2]"""
import argparse
import os
from dataclasses import replace

import pandas as pd

from .config import Config
from .io import load_tactical
from .metrics import summary
from .pipeline import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuse-broadcast", action="store_true", help="fill tactical ball gaps from the aligned broadcast ball")
    ap.add_argument("--gap-mode", default="off", choices=["off", "viterbi", "cp", "mid"])
    ap.add_argument("--gap-w", type=float, default=0.03, help="weight for --gap-mode viterbi")
    a = ap.parse_args()
    cfg = Config()
    t = os.path.join(a.data, "tactical")
    ball, P, team = load_tactical(f"{t}/BAR-ATH_tactical_atd.xml", f"{t}/BAR-ATH_tactical_atd.txt", cfg)
    print(f"[run] ball visible on {(~pd.isna(ball[:, 0])).mean():.1%} of all frames (docs: 56.9%)")
    if a.fuse_broadcast:
        from .sync_broadcast import fuse_from_files
        ball = fuse_from_files(a.data, ball, cfg)
    labels = predict(ball, P, team, cfg)
    if a.gap_mode == "mid":
        labels = predict(ball, P, team, replace(cfg, gap_mode="mid"))
    elif a.gap_mode != "off":
        from .gapmodel import build_features, crossfit_logit
        logit = crossfit_logit(build_features(P, team, cfg), labels, ball, cfg)
        labels = predict(ball, P, team, replace(cfg, gap_mode=a.gap_mode, gap_w=a.gap_w, gap_conf=1.0), gap_logit=logit)
    pd.DataFrame({"frame": range(cfg.n_frames), "label": labels}).to_csv(a.out, index=False, lineterminator="\n")
    print("[run] wrote", a.out, summary(labels, cfg.blocks))
    print("[run] now run: python evaluation/validate_submission.py", a.out)


if __name__ == "__main__":
    main()
