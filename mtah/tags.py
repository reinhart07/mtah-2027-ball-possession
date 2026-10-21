"""Smart Tagging as per-frame evidence (automatic phases of play, each carrying a team).

Offensive phases (Build Up, Progression, Creating Chances, ...) name the team WITH the ball; defensive phases
(Defending in X third, Defensive Transition) name the team WITHOUT it. Unlike the ball track, the tags exist
across the long ball-less gaps. They are inferred, not ground truth, so we measure how reliable they are
(tagcheck) before trusting them, and let the gap model learn their weight.
Columns: 0 off_home, 1 off_away, 2 def_home, 3 def_away, 4 setpiece_home, 5 setpiece_away.
"""
import numpy as np

from .config import Config

OFF = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "BALL IN FINAL THIRD",
       "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "GOALS"}
DEF = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
       "DEFENSIVE TRANSITION"}


def tag_features_from_df(df, cfg: Config, n=None):
    n = n or cfg.n_frames
    out = np.zeros((n, 6))
    f0 = (df["timestamp"].dt.total_seconds() * cfg.fps).round().astype(int).to_numpy()
    f1 = (df["end_timestamp"].dt.total_seconds() * cfg.fps).round().astype(int).to_numpy()
    code = df["code"].astype(str).str.upper().str.strip().to_numpy()
    team = df["Team"].astype(str).str.strip().str.lower().to_numpy()
    for a, b, c, t in zip(f0, f1, code, team):
        side = 0 if t.startswith("barcelona") else (1 if t.startswith(("atletic", "athletic")) else None)
        grp = 0 if c in OFF else (1 if c in DEF else (2 if c == "SET PIECES" else None))
        if side is None or grp is None:
            continue
        out[max(a, 0):min(b, n), grp * 2 + side] += 1
    return out


def load_tag_features(path, cfg: Config):
    from kloppy import sportscode
    return tag_features_from_df(sportscode.load(path).to_df(), cfg)


def tag_score(F):
    """> 0: tags say HOME has the ball (home attacking or away defending)."""
    return (F[:, 0] + F[:, 3]) - (F[:, 1] + F[:, 2])
