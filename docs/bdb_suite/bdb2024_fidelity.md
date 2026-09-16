# BDB 2024 winner-fidelity contract

The fidelity branch is pinned to
[`mpchang/uncovering-missed-tackle-opportunities@8b3de97`](https://github.com/mpchang/uncovering-missed-tackle-opportunities/commit/8b3de97f1e42351d14e5b69d8cb03f51f244806a).
It is deliberately separate from the five-role stability comparison. The
published XGBoost model is a reference implementation; LightGBM remains the
tree member of the common five-model grid.

## Cohort and event state

- Positive examples are charted solo tackles. Negative examples are PFF
  charted missed tackles. Assisted-only rows are excluded; matching the
  executed notebook, a PFF-miss row remains negative even when its `assist`
  flag is also set.
- Plays are removed when `foulName1` is present or
  `playNullifiedByPenalty == "Y"`.
- Made tackles use the unique `tackle` or `out_of_bounds` event frame. A
  charted miss uses the first frame at the minimum defender–ballcarrier
  Euclidean distance.
- The model state is exactly ten 10-Hz frames before that event. Tracking is
  first restricted from five frames after the snap through the earlier of the
  tackle and out-of-bounds frames, as in the pinned source.
- Direction and orientation are rotated to the mathematical unit-circle
  convention. Leftward plays reflect `x` and the x vector component; the
  winner leaves `y` unchanged. Before annotation and feature construction,
  position/speed/acceleration are rounded to hundredths and angles to tenths,
  matching the executed memory-optimization step.

Those rules govern the exact nine-feature fidelity path. RelNet, AttnRelNet,
and the Global Set Transformer in the protocol-v2 five-role study use their
shared suite-wide token contract instead: a 180-degree x/y rotation for
leftward plays, centered on the football. The relational pair additionally
receives typed edges, while the Global Set Transformer does not. This
intentional distinction is recorded in every prepared artifact; it does not
change the winner reference features.

Complete preprocessing must reproduce these executed notebook counts:

| Split | Made | Missed | Total |
|---|---:|---:|---:|
| Weeks 1–8 | 8,000 | 1,583 | 9,583 |
| Week 9 | 836 | 180 | 1,016 |

`prepare_bdb2024(..., require_fidelity_counts=True)` fails rather than
continuing when any count differs. Count verification is available only for
the complete, unfiltered release.

## Exact feature order

The executed classifier receives these nine values, in order:

1. Ballcarrier speed.
2. Tackler x velocity minus ballcarrier x velocity.
3. Absolute tackler–ballcarrier y velocity difference.
4. Defender–ballcarrier Euclidean distance.
5. Cosine of the angle between the defender's heading and the ballcarrier's
   one-second projected position.
6. Ballcarrier bounded Voronoi area, including the virtual player ten yards
   behind the carrier.
7. Net offense-minus-defense influence at the ballcarrier.
8. Offensive blocker influence at the candidate defender.
9. Run indicator.

The Voronoi reflection, three-yard anisotropic Gaussian influence field, and
18-yard-per-second speed scale are transcribed in
`bdb_study/fidelity/bdb2024.py`. The machine-readable receipt exposes this
order and the pinned source commit.

## Executed model versus appendix

The executed `train_model.ipynb` cell fits an `XGBClassifier` with 150 trees,
depth 7, learning rate 0.1, L2 penalty 150, row subsampling 0.75, and early
stopping patience 20. The submission appendix instead reports **250 trees**.
The fidelity branch follows the executable notebook—150 trees—and records the
250-tree appendix value as a known discrepancy. Importing or testing the
fidelity utilities does not require XGBoost; a runtime XGBoost dependency is
needed only to fit the optional reference model. The executable entry points
are `fit_executed_xgboost` and `predict_fidelity_score`; the former also
materializes the notebook's unstratified 90/10 split (seed 1234) and its exact
data-dependent class weight.

The executable fidelity command now persists the fitted booster itself, the
two exact split-index arrays, a model/training-history receipt, and one
week-nine score for every charted candidate:

```bash
caffeinate -i "$BDB_PY" -m bdb_study fidelity-bdb2024 \
  --prepared-dir data/processed/bdb_suite/bdb2024_tackle \
  --raw-dir data/bdb2024/raw \
  --output "data/bdb_suite_runs/$BDB_CAMPAIGN/bdb2024_fidelity/result.json" \
  --artifact-dir "data/bdb_suite_runs/$BDB_CAMPAIGN/bdb2024_fidelity/artifacts"
```

The definitive artifact format is Parquet. It requires the suite's pinned
`xgboost` and `pyarrow` environment; neither dependency is imported merely by
loading the fidelity utilities. `--artifact-format csv` exists for small,
dependency-light smoke fixtures and is not recommended for the full release.
The current shared development environment does not contain those two optional
packages, so the real fit and whole-release inference must wait for the
dedicated suite environment rather than modifying the active 2020 environment.

The reference artifact tree contains:

- `executed_xgboost_150.ubj`, the actually fitted model (150 is the configured
  tree budget; the receipt also records the realized early-stopping round);
- deterministic training and validation index arrays;
- `week9_candidate_scores.csv`, with candidate identity, charted label, and raw
  case-control score for all 1,016 week-nine candidates; and
- `reference_receipt.json`, binding every byte hash, the prepared-task hash,
  the exact feature order, parameters, evaluation history, and winner commit.

### Whole-release framewise extrapolation

After persisting the reference, the same command scores every tracked defender
at every retained frame of every eligible play in weeks 1--9. It reads and
releases one weekly Parquet tracking shard at a time; it never concatenates the
nine raw tracking files. Within a week, features and scores are emitted in
bounded row groups, so the full feature/score table is not held in memory.

For each week the command atomically creates:

- `frame_scores/week_XX.parquet`, containing game, play, frame, and defender
  identity; all nine unscaled winner features in locked order; and the raw
  case-control score;
- `opportunity_summaries/week_XX.parquet`, containing one play/defender row,
  frame bounds, score count, opportunity and missed-opportunity counts, and
  the exact event frame IDs; and
- `weeks/week_XX.json`, binding the raw tracking hash, fitted-model hash,
  schemas, row counts, preprocessing audit, and both output checksums.

A valid weekly receipt is checksum-validated and skipped on rerun, so a failed
whole-release job resumes at the first missing week. Existing output made from
a different model, raw shard, format, or bytes fails rather than being
overwritten. `framewise_receipt.json` is written only after all nine weeks
complete.

The raw Parquet import deliberately preserves NFL literal `"NA"` values.
Fidelity preprocessing maps `passResult="NA"` to missing, matching the
winner's `pandas.read_csv` behavior and therefore its run indicator. The same
normalization is used by the charted-candidate adapter.

## Opportunity automaton and interpretation

The threshold comparison is strict: a score must be `> 0.75` for five
consecutive frames to create one opportunity. Once in that opportunity, five
consecutive frames at or below 0.75 create a missed opportunity and reset the
state. A renewed high frame during the down counter returns to the existing
opportunity and does not create another one.

Because training deliberately contrasts confirmed tackles with charted
misses, this output is a **case-control made-versus-missed score**. It is not
an unconditional probability that an arbitrary defender will tackle the ball
carrier in the next second. Reports and artifact schemas retain that wording;
calibration results, if any, are stored separately.

The framewise extrapolation retains this warning in every receipt and column
name. Its output is a heuristic case-control score over arbitrary defenders,
not a calibrated probability of an unconditional tackle event. Missing frame
IDs reset the consecutive-frame automaton; they cannot be bridged to fabricate
a five-frame opportunity.
