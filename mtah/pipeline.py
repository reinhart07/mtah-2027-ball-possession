import numpy as np

from .ball import clean_ball, runs
from .config import Config
from .decode import decode
from .evidence import control_evidence, dead_evidence, player_activity_smooth
from .gapcp import relocate_flips


def gap_evidence(valid, gap_logit, cfg: Config):
    """Additive player-only evidence inside long gaps, with a dead-zone for weak opinions."""
    l = np.nan_to_num(gap_logit)
    l = np.sign(l) * np.maximum(np.abs(l) - cfg.gap_conf, 0.0)
    ev = np.zeros(len(valid))
    for s, e in runs(~valid):
        if e - s >= cfg.gap_apply_min:
            ev[s:e] = cfg.gap_w * l[s:e]
    return ev


def predict(ball, P, team, cfg: Config = None, gap_logit=None):
    cfg = cfg or Config()
    b, speed = clean_ball(ball, cfg)
    valid = ~np.isnan(b[:, 0])
    phi = control_evidence(b, P, team, cfg)
    if gap_logit is not None and cfg.gap_mode == "viterbi" and cfg.gap_w:
        phi = np.where(valid, phi, gap_evidence(valid, gap_logit, cfg))
    act = player_activity_smooth(P, cfg) if cfg.calm_bonus else None
    delta, _ = dead_evidence(b, speed, cfg, act)
    lab = decode(phi, delta, cfg)
    if cfg.gap_mode == "mid":
        lab = relocate_flips(lab, valid, None, cfg)
    elif cfg.gap_mode == "cp" and gap_logit is not None:
        lab = relocate_flips(lab, valid, gap_logit, cfg)
    return lab
