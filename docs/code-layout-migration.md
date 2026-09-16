# Code layout migration

The September 2026 cleanup separates reusable research implementation from
standalone commands and retired campaign tooling.

| Previous path | Current location |
| --- | --- |
| `neural_models.py` | `rushing_study/neural_models.py` |
| Probability/interval helpers in `training_size_sweep.py` | `rushing_study/intervals.py` |
| `prepare_data.py` | `scripts/prepare_data.py` |
| `create_venv.sh` | `scripts/create_venv.sh` |
| `paper/plot_*.py` | `scripts/plots/` |
| Maintained presentation build/render commands | `scripts/presentations/cassis2026/` |
| Reusable `hpc/betty/` wrappers and environment tools | `scripts/betty/` |

The legacy training-size sweep and its worker, versioned pilot launchers,
failed attempts, one-off recovery controllers, and historical export/refit
helpers are retired. Their source and dedicated tests were moved to the
ignored `archive/`, preserving the local records needed for future diagnosis.
No archive or internal content is included in the GitHub repository.

## Completed research records

Completed result payloads, manifests, checksum sidecars, and scientific
protocol amendments retain their original bytes and recorded identities.
Historical paths inside those records describe the source at execution time;
they are not current command examples.

Moving executable source changes its identity even when its calculations are
unchanged. Current imports and provenance declarations use the new paths.
The frozen rushing configuration has a new exact source-layout digest; its
scientific settings, split design, models, seeds, and tuning choices are
unchanged. The old configuration and manifests are not silently admitted
against changed source. Historical resource/probe checks also continue to
verify source bytes and reject drift.

The local `archive/source-layout-v1-20260915/` snapshot preserves the original
source layout. It is not present in a clone. Reproducing an old campaign
requires that original source release plus its data, environment, and hardware
records; request those separately. A source snapshot alone does not contain
raw datasets or grant permission to rerun a frozen campaign.

## Maintained scope

The maintained product includes the study packages, preprocessing, current
plots and presentation renderer inputs, and reusable Betty wrappers. Retired
controllers are not advertised as current launch tools. Cluster-specific
work can be adapted from the wrappers; any new fitting requires fresh
admission evidence rather than reusing a historical source attestation.

Tests cover the current command/import layout, scientific contracts, result
analysis, and strict rejection of source drift. A path migration does not
claim a fresh hardware determinism or runtime admission check.
