# More Than A Hack 2027 - Phase 1: ball possession (Rei JR)

Repository: https://github.com/reinhart07/mtah-2027-ball-possession

Frame-level possession labels (`H` / `A` / `D`) for the BAR-ATH match, produced **algorithmically** from the
provided tactical tracking (ATD). No hand labelling, no private ground truth. Technical write-up: `TECH_DOC.md`.

## Reproduce the submission
```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027      # scorer + public ground truth (organisers)
pip install -r requirements.txt                                          # Python 3.13 (>= 3.11)
# place the organisers' data bundle in ./data (tactical/ and broadcast/ keep their original file names)
python -m mtah.run --data data --out Rei_JR.csv                    # the submission file, ~3 min, deterministic
python evaluation/validate_submission.py Rei_JR.csv                # run from the organisers' repo
```
The default path has no randomness: the same data and package versions give the same file byte for byte.

## Method (default path = what produced the submission)
1. **Ball cleaning**: drop jumps above 40 m/s, interpolate gaps up to 0.5 s.
2. **Control evidence**: per frame, a log-likelihood ratio from the nearest home vs. nearest away player,
   gated by distance (Gaussian, 2 m). Zero when nobody is near the ball or the ball is unseen.
3. **Dead-ball evidence**: ball out of bounds, ball vanishing at a line, and "calm" play (median player speed
   below 1.0 m/s while the ball is slow or unseen).
4. **Two-stage Viterbi**: alive/dead first, then H/A on alive frames only; zero evidence means "keep the last
   state" (last team in control keeps possession, as the rules require). Runs under 1 s are merged.
5. Frames 110,808-113,741 (graded, documented as all dead ball) are set to `D`.

All parameters live in `mtah/config.py`.

## Experiments (not part of the default path, all off by default)
`gapmodel.py`, `gapcp.py`, `tags.py`, `sync_broadcast.py`, `video_probe.py`, `video_ball.py`, `video_pilot.py`, `video_fill.py`, `video_gt_check.py`, `ft_dataset.py`, `ft_train.py` and the `sweep*`, `benchmark*`,
`bench_dead`, `tagcheck`, `diagnose`, `dead_probe`, `tag_probe` scripts document what we tried and measured. See `TECH_DOC.md` for the results.

## Data handling
The match data is licensed for this competition only: it is not in this repository (`data/` is gitignored) and must
not be committed, redistributed or pasted into hosted LLMs. The `--verbose` flags print raw excerpts for local
inspection only; default outputs are aggregate metrics.