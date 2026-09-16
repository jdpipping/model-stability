# Outcome-blind protocol amendment: fifth primary Global Set Transformer

Date: 2026-08-21 (America/New_York)

Status: prospective amendment made before any BDB2021--2026 pilot prediction,
smoke result, frozen development selection, task aggregate, or suite aggregate.

## Decision and rationale

The BDB2021--2026 suite adds `set_transformer` as a fifth primary model role.
The role is displayed as **Global Set Transformer**. This decision was prompted
by a design discussion about continuity with the completed BDB2020 winner and
the distinction between unrestricted global set attention and task-typed graph
attention. It was not prompted by any BDB2021--2026 development or pilot score.

The five primary roles are now, in frozen order:

1. `linear_structure`
2. `boosted_structure`
3. `relnet`
4. `attn_relnet`
5. `set_transformer`

The Global Set Transformer is a protocol-v2 temporal adaptation, not a claim
that the literal BDB2020 snapshot architecture can be reused unchanged. It
receives the same player tokens, stable slots, masks, ordered frame history,
time-to-event signal, context, task-specific output head, loss, development
folds, tuning grid, seeds, epoch-selection rule, and refit protocol as the two
relational neural roles. It intentionally omits typed task-graph edges and
instead uses masked global player-set attention. RelNet versus AttnRelNet
remains the sole controlled fixed-versus-adaptive edge-aggregation contrast.

The fifth role is committed to both `pilot10` and `full100` regardless of its
pilot performance. The pilot cannot be used to decide whether to retain it.

## Evidence boundary at amendment

The superseded attempt-2 campaign was bound by campaign hash
`96c39edb8dc84282293c70fc456a77f3e18f2956a486c84df6f1b79e20de7f03`
and terminal dispatcher job `7725228`. Its checksummed journal and campaign
receipt remain preserved under their original `pilot10_attempt2.json` paths.

At the decision time:

- no pilot primary cell had been submitted by the dispatcher;
- no pilot prediction, smoke, preflight, benchmark, runtime plan, frozen task
  selection, task aggregate, or suite aggregate existed;
- no development reducer had produced a selected development receipt;
- individual unreduced fold/candidate checkpoints existed, but their metric
  payloads were not inspected by the investigators or implementation agents.

All 710 attempt-2 job IDs were resolved from the immutable attempt-2 journal.
Only its 651 still-active IDs were cancellation targets; no BDB job existed
outside that set and the unrelated rushing campaign was untouched. After the
queue reached zero active BDB jobs, 65 unreduced development/runtime files were
moved without opening score payloads to:

- `data/bdb_suite_runs/archive/pilot10_attempt2_20260821/development`
- `data/bdb_suite_runs/archive/pilot10_attempt2_20260821/development_runtime`

The first scope contains 57 files and 1,908,689 bytes with canonical inventory
SHA-256 `97b937e2318872a5c112d4a95001ed9e382a489cd5d269088d7971c0492e6e43`.
The second contains 8 files and 5,350 bytes with canonical inventory SHA-256
`c06f100c3bcbc6c1628cd40fcde58c4b4c7d7b66482eab1a1fc65ccd6aeb9bca`.
Each inventory hash is computed from sorted relative path, byte count, and file
SHA-256 records; file bytes were hashed but metric payloads were not interpreted.

All development selection will restart under the amended code and task
contracts. No attempt-2 checkpoint or runtime probe is eligible for reuse.
Checksum-valid prepared data remain reusable because their observation cohort and
tensor semantics are unchanged and independently validated.

## Amended design and inference

- `pilot10`: 10 repeats x 6 anchors x 5 roles = 300 primary cells per task and
  1,800 across six tasks; no pilot ablation or sensitivity cells.
- `full100`: 100 repeats x 6 anchors x 5 roles = 3,000 primary cells per task
  and 18,000 across six tasks.
- Targeted structural ablations remain restricted to RelNet and AttnRelNet:
  80 fits per task and 480 across six tasks.
- The BDB2024 and BDB2025 `sensitivity20` profiles include all five primary
  roles and contain 200 fits each.
- The within-task primary max-t family expands from 36 to 60 contrasts:
  `choose(5, 2) x 6 anchors`. A supported-best model must beat all four
  comparators under that simultaneous family.
- For BDB2026, residual-versus-absolute decoder selection is shared across all
  three neural roles using a prespecified equal-family grouped-development
  criterion; the exact tie rule remains residual decoding.
- Set Transformer parameter count and representative FLOPs are checksummed and
  capped prospectively. The 5% parameter and 15% FLOP matching claim continues
  to apply specifically to RelNet versus AttnRelNet.

The next admissible cluster campaign must use a fresh attempt-3 journal,
receipt, manifests, development receipts, preflights, runtime plans, run
directories, and aggregate identities. Attempt 1 and attempt 2 remain archived
evidence and must never be resumed or overwritten.
