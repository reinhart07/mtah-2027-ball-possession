from dataclasses import dataclass


@dataclass
class Config:
    # geometry / time (DATA.md: 25 fps, pitch 105.3 x 68.0 measured by organisers)
    fps: int = 25
    pitch_l: float = 105.3
    pitch_w: float = 68.0
    n_frames: int = 199_319
    # graded blocks (DATA.md): (first_frame, last_frame) inclusive
    blocks: tuple = ((11_185, 90_651), (110_808, 190_197))

    # FAQ.md: 2nd half is graded from 110,808 but kicks off at 113,742; those 2,934 frames are all D
    forced_dead: tuple = ((110_808, 113_741),)

    # ball cleaning
    max_ball_speed: float = 40.0     # m/s, faster = tracker jump -> drop
    max_interp_gap: int = 12         # frames (0.5 s) linearly interpolated

    # control evidence (who is closest to the ball, trusted only when close)
    ctrl_sigma: float = 1.5          # metres: distance gap -> log-likelihood ratio
    ctrl_radius: float = 2.0         # metres: gate, evidence fades beyond this
    llr_clip: float = 3.0

    # decoding (Viterbi over H / A / D)
    switch_cost: float = 24.0        # H<->A
    dead_switch_cost: float = 24.0   # H/A<->D

    # dead-ball evidence
    out_margin: float = 0.5          # m beyond the lines = ball out
    out_bonus: float = 4.0
    static_speed: float = 0.4        # m/s
    static_min_len: int = 50         # 2 s of a motionless ball = stoppage
    static_bonus: float = 2.5
    live_dead_penalty: float = 0.5   # D's score on an ordinary in-play frame
    calm_bonus: float = 1.5          # >0: players walking (low activity) is evidence of a stoppage
    calm_thr: float = 1.0            # m/s, median player speed below this = calm
    calm_win: int = 75               # frames of smoothing for the activity signal
    calm_ball_speed: float = 5.0     # calm only counts if the smoothed ball speed is below this (or ball unseen)
    gap_dead_rule: bool = True       # ball vanishes at the line, reappears near it
    gap_min_len: int = 25
    gap_before_m: float = 2.0
    gap_after_m: float = 6.0

    # post-processing
    min_possession: int = 25         # frames
    min_dead: int = 12

    # player-only possession evidence for frames where the ball is unseen (0 = off)
    gap_mode: str = "off"           # off | viterbi | cp (model-placed hand-overs) | mid (hand-over at the gap midpoint)
    gap_beta: float = 1.0            # weight of the midpoint prior inside cp
    gap_w: float = 0.0
    gap_apply_min: int = 75          # only gaps at least this long (3 s) get the model; shorter ones just hold
    gap_conf: float = 1.0            # dead-zone on |log-odds|: weak opinions are ignored
