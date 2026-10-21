"""Load the tactical ATD (tracking) into dense arrays indexed by VIDEO frame."""
import numpy as np
import pandas as pd

from .config import Config


def team_lookup(dataset):
    """slot id (str) -> 0 home / 1 away, from the kloppy metadata."""
    look = {}
    for ti, team in enumerate(dataset.metadata.teams):
        for p in team.players:
            look.setdefault(str(p.player_id), ti)
    for ti, team in enumerate(dataset.metadata.teams):
        for p in team.players:
            if p.jersey_no is not None:
                look.setdefault(str(p.jersey_no), ti)
    return look


def from_dataframe(df: pd.DataFrame, look: dict, cfg: Config):
    """df: kloppy to_df() layout (frame_id, ball_x, ball_y, <slot>_x, <slot>_y, <slot>_s)."""
    slots = [c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"]
    kept = [s for s in slots if s in look]
    dropped = [s for s in slots if s not in look]
    print(f"[io] {len(slots)} track slots, {len(kept)} with a team, dropped: {dropped}")
    if len(kept) < 20:
        raise RuntimeError("could not map slots to teams - inspect dataset.metadata.teams")
    vid = df["frame_id"].to_numpy().astype(int) - 1          # DATA.md: video_frame = frame_id - 1
    n = cfg.n_frames
    ok = (vid >= 0) & (vid < n)
    ball = np.full((n, 2), np.nan)
    ball[vid[ok], 0] = df["ball_x"].to_numpy()[ok] * cfg.pitch_l
    ball[vid[ok], 1] = df["ball_y"].to_numpy()[ok] * cfg.pitch_w
    P = np.full((n, len(kept), 2), np.nan)
    for j, s in enumerate(kept):
        P[vid[ok], j, 0] = df[f"{s}_x"].to_numpy()[ok] * cfg.pitch_l
        P[vid[ok], j, 1] = df[f"{s}_y"].to_numpy()[ok] * cfg.pitch_w
    team = np.array([look[s] for s in kept])
    return ball, P, team


def load_tactical(atd_meta, atd_raw, cfg: Config):
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=atd_meta, raw_data=atd_raw)
    return from_dataframe(ds.to_df(), team_lookup(ds), cfg)
