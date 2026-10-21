# Technical document - Rei_JR
More Than A Hack 2027, Phase 1 (frame-level ball possession, BAR-ATH)

## 1. Summary
Labels come from a small, fully inspectable state model on the provided tactical tracking (ATD): per-frame
control evidence from player-ball distances, dead-ball evidence from ball and player behaviour, and a two-stage
Viterbi decoder that makes possessions sticky. No hand labelling and no private ground truth were used. The public
5 minutes were used to check correctness and to understand the label definition, never to choose between two close
variants. The default path has no randomness and regenerates the submitted file exactly (section 7).

## 2. Approach and why
The ranking metric (spellF1) counts whole possessions, so a system must be sticky by construction. The reference
baselines show it: the same nearest-player rule gives spellF1 0.04 (raw) to 0.42 (125-frame vote) on the public
window. Instead of tuning a smoothing window, we model possession as a state (H, A, D) that only changes when
evidence accumulates, which also matches the rule that a loose ball stays with the last team in control.

1. **Ball cleaning.** Jumps above 40 m/s are dropped; gaps up to 0.5 s are interpolated.
2. **Control evidence.** phi = gate * clip((d_away - d_home) / 1.5 m, +-3), with gate = exp(-(d_min / 2.0 m)^2),
   where d_home / d_away are the distances from the ball to the nearest home / away tracked player. phi is zero
   when nobody is close (a pass in flight) or the ball is unseen, so the state is simply held.
3. **Dead-ball evidence.** Ball out of bounds; ball unseen for >= 1 s between two points near a line; a ball
   motionless for >= 2 s; and "calm" play: median player speed (1 s displacement, smoothed over 3 s) below
   1.0 m/s while the smoothed ball speed is below 5 m/s or the ball is unseen. In play, dead gets a small penalty.
4. **Two-stage decoding.** Viterbi over alive/dead (switch cost 24, dead runs under 0.5 s dropped), then Viterbi
   over H/A on the alive frames only (switch cost 24), so a stoppage can never be used as a cheap shortcut between
   H and A. H/A runs shorter than 1 s are merged into their longer neighbour.
5. **Documented boundary.** Frames 110,808-113,741 are graded and are all dead ball (FAQ); they are set to `D`.

## 3. Assets used and dropped
- **Used:** tactical ATD (players and ball), frame alignment (frame_id - 1), the graded-window boundaries.
- **Tried and dropped (all measured, see section 5):** broadcast tracking (alignment to the tactical timeline),
  Smart Tagging team tags, a learned player-only possession model. Homography and the videos were not needed.

## 4. Possession rules implemented
- Possession belongs to the last team in control; a loose or contested ball keeps its previous owner.
- A change needs sustained, close control by the other team (about 0.3 s of unambiguous evidence at a 2 m gate,
  plus the 1 s minimum run); a touch, deflection or lost duel does not flip the state.
- Dead ball is a stoppage: it is never a possession, and a stoppage separates two possessions of the same team.
- Restart ownership is not modelled; the team in control when play resumes is read from the same evidence.

## 5. Evidence and measurements (aggregate numbers only)
**Public 5 minutes** (17 possessions, 7,500 graded frames):

| system | macroF1 | spellF1 | possessions | mean duration |
|---|---|---|---|---|
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 | 10.8 s |
| baseline_smooth125 | 0.581 | 0.424 | 16 | 18.5 s |
| this submission | 0.792 | 0.518 | 17 | 15.5 s |

**How much to trust these numbers.** With 17 public possessions the uncertainty is large: the organisers' bootstrap (1,000 draws) gives a 5-95% interval of 0.34-0.68 for spellF1 and 0.59-0.85 for macroF1, and differences below about 0.34 spellF1 are within noise. The public window verifies that the pipeline is correct; it does not rank it.

Dead ball: precision 0.93, recall 0.57 (F1 0.71). Team accuracy on alive frames 0.87. The public window is
optimistic: the ball is seen on 94% of its frames versus 59% over the graded match, and on the baselines the full-match
spellF1 was 0.5-0.75 of the public one. We expect the full-match score to be clearly lower than 0.52; we have no way
to measure it.

**Parameter choice.** Control gate 2 m versus 3 and 4 m: the public possession count went 19 / 25 / 31 (truth 17)
with monotonically better spellF1 at 2 m. Switch cost 24: the possession count (16-17) matches the truth, while
larger costs over-merge; spellF1 kept rising with the cost, so it was not used as the criterion. Dead-ball settings
were taken from a plateau of 24 combinations (precision 0.8-0.93, recall 0.5-0.65, full-match dead share 35-41%).

**Ball-less gaps.** 143 gaps, 63,037 graded frames (about 40%); 86% of those frames lie in gaps of 10 s or more
(median 10.5 s, 90th percentile 36 s). The decoder holds the last state through them.

**Experiments, not adopted** (all scored against our own un-masked output, since no truth exists inside gaps):
- *Player-only possession model* (gradient boosting on team shape and movement, time-blocked cross-fitting,
  self-supervised from ball-visible frames): held-out AUC 0.74-0.88, but as extra evidence in gaps it did not beat
  holding the state. Realistic gap lengths, 4 seeds: spellF1 difference vs hold -0.009 +- 0.017.
- *Change-point / midpoint placement of hand-overs inside gaps:* -0.002 +- 0.008 and -0.006 +- 0.008.
- *Smart Tagging team tags* as a possession signal: covers 73.8% of graded frames (69.7% of ball-less frames) but agrees
  with ball-based labels only 73% (70% with the public truth); adding them to the model changed AUC by < 0.01.
- *Broadcast tracking:* aligned by ball trajectory; only 13% of chunks aligned, graded ball visibility 59.3% -> 59.9%.

## 6. Limitations
- Gaps are handled by holding the state; hand-overs inside long gaps are placed at the gap end.
- Dead-ball recall is 0.57: starts of stoppages (ball still in flight) and calm-but-not-dead phases are missed or
  confused. Dead-ball thresholds were set from 4 public stretches and a plateau argument, not from many examples.
- "Controlled possession" is read from the geometry and 17 public possessions; short possessions (1-3 s) are the
  hardest to match under IoU.
- No restart-team modelling; the full-match score is unknown.

## 7. Reproduction
Python 3.13, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, kloppy 3.19.1 (pinned in requirements.txt; scikit-learn 1.6.1 only for the optional experiments, requirements-experiments.txt).
```
git clone <this repository> && cd <repo>
pip install -r requirements.txt
# unpack the organisers' data into ./data (original file names)
python -m mtah.run --data data --out Rei_JR.csv
python evaluation/validate_submission.py Rei_JR.csv    # organisers' repository
```
Runtime about 3 minutes. The default path is deterministic (no random numbers, no learned components); runs on Linux and on Windows produced byte-identical files (MD5 e7d50429dfd068b7a57730dd626a12fd).
