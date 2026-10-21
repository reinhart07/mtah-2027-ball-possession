"""Do Smart Tagging set pieces line up with the end of the GT dead-ball stretches?

    python -m mtah.tag_probe --data data --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from .diagnose import rle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"))
    ap.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    ap.add_argument("--verbose", action="store_true", help="LOCAL ONLY: lists tag rows - never paste into a hosted LLM")
    a = ap.parse_args()
    from kloppy import sportscode
    df = sportscode.load(os.path.join(a.data, "tactical", "BAR-ATH_tactical_smart_tagging.xml")).to_df()
    print("columns:", list(df.columns))
    df["f0"] = (df["timestamp"].dt.total_seconds() * 25).round().astype(int)
    df["f1"] = (df["end_timestamp"].dt.total_seconds() * 25).round().astype(int)
    sp = df[df["code"].str.upper() == "SET PIECES"].copy()
    tcol = next((c for c in sp.columns if c.lower() == "type"), None)
    print(f"\nSET PIECES: {len(sp)} in the match;", sp[tcol].value_counts().to_dict() if tcol else "no Type column")
    gt = pd.read_csv(a.gt)["label"].astype(str).str.strip().str.upper().to_numpy()
    idx = np.where(gt != "X")[0]; ga, gb = int(idx.min()), int(idx.max())
    print(f"\nGT dead-ball stretches in {ga}..{gb} (start, end, seconds):")
    for s, e, l in rle(gt[ga:gb + 1], ga):
        if l == "D":
            print(f"  D {s}-{e}  ({(e - s) / 25:.1f}s)")
    cols = [c for c in ("f0", "f1", tcol, "Team", "Half") if c and c in sp.columns]
    inside = sp[(sp.f1 >= ga - 500) & (sp.f0 <= gb + 500)]
    print(f"\nset pieces overlapping the public window: {len(inside)}")
    if a.verbose:
        print(inside[cols].to_string(index=False))
        print(sp[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
