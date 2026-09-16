# BDB2026 development-scale tail amendment (2026-08-21)

This amendment was recorded before any BDB2021--2026 TaskSpec was frozen and
before any smoke or pilot prediction was produced. Attempt 4 stopped when the
BDB2026 GLM development reducer found that the sequestered development games
had no target-mask support at horizon 34. No checkpoint score, validation
loss, prediction, or later-year model comparison was inspected.

The mask-only audit showed a structural support mismatch. The 36 development
games contain 6,280 requested-player paths and positive counts through horizon
33, followed by zero counts at horizons 34--94. The 236 confirmatory games
contain paths through horizon 94; every pilot test split needs horizon 34, and
repeats 4, 6, and 9 need horizon 94. Moving the rare long-horizon games into
development would change the frozen registry and remove important long paths
from confirmation, so the registry remains unchanged.

The development scale learner is amended narrowly. Positive pooled counts must
form one nonempty prefix. Within that prefix, the existing count-weighted fold
median, 0.25-yard floor, and weighted nondecreasing PAVA are unchanged. Any
unobserved trailing horizons receive exactly the final fitted PAVA level. A
leading or internal support gap, a count increase, or a finite median paired
with a zero count fails closed. The selected configuration records the observed
cutoff and the exact tail-policy identifier, and the SHA-256 continues to bind
all 94 scale values.

The carried tail is not described as interpolation, extrapolated growth, or a
conservative long-horizon error estimate. It is the minimal nondecreasing
completion when development outcomes contain no tail information. The separate
game-level split-conformal maximum-score quantile remains the mechanism for
marginal 90% whole-path coverage. No horizon-conditional guarantee, learned
tail efficiency, or simultaneous guarantee for every player is claimed.

Attempt 4 was cancelled fail-closed and recoverably archived at
`data/bdb_suite_runs/archive/pilot10_attempt4_20260821`. The replacement is a
fresh `pilot10_attempt5` campaign; no attempt-4 development checkpoint or
receipt is reusable after the code and scientific-contract change.
