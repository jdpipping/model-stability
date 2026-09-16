# CASSIS 2026 presentation

[`index.qmd`](index.qmd) is the maintained source for *Efficient Coverage
Playbook: Comparing Predictive Uncertainty Across Model Classes for NFL Big
Data Bowl Problems*. The talk presents the completed BDB2020–2025 studies.

## Render the presentation

The retained figures, fonts, theme, and bibliography are sufficient to render
the deck. Install Quarto, then run from the repository root:

```sh
quarto render presentations/cassis2026/index.qmd
```

Open the generated `index.html`. To create the print edition with separate
reveal states, render the deck first, then run:

```sh
.venv/bin/python scripts/presentations/cassis2026/prepare_reveal_print.py
```

Open `PRINT-reveals.html` and use the browser's print dialog to save a PDF.
The print helper also supports module execution:

```sh
.venv/bin/python -m scripts.presentations.cassis2026.prepare_reveal_print --help
```

Rendered HTML, its `index_files/` support directory, and Quarto caches are
ignored. Rendering retained assets requires no historical source archive,
original tracking dataset, model refit, or inference recomputation.

## Maintained tools and inputs

The maintained presentation tools are
[`prepare_reveal_print.py`](../../scripts/presentations/cassis2026/prepare_reveal_print.py)
and [`build_figures.R`](../../scripts/presentations/cassis2026/build_figures.R).
The publication plotters live separately in [`scripts/plots/`](../../scripts/plots/).

`figures/` and compact `data/` exports are maintained renderer inputs. The
figure manifest and data receipts retain their original hashes and provenance.
Source-code paths recorded in old receipts describe the historical export
process; those old programs are not supported current commands.

The R figure builder can refresh figures from those retained tables and
examples when the additional local study inputs are available. It requires the
R packages imported at the top of the script, plus:

- `data/bdb2025/raw/players.csv`;
- `data/bdb_suite_runs/definitive/bdb2020_rushing_harmonized/full100/`;
- the retained completed-study results under `results/`.

From the repository root:

```sh
Rscript scripts/presentations/cassis2026/build_figures.R --help
Rscript scripts/presentations/cassis2026/build_figures.R --only=rushing_accuracy,manzone_results
```

The builder verifies the renderer's example-data files against the existing
receipt. It does not execute or attest to the historical extraction code. The
local data inputs above are not included in a fresh clone; reuse the retained
figures until those inputs are available.

## Historical preparation

Completed extraction, example refit, sensitivity, and intermediate-inference
helpers are preserved locally in `archive/retired-analysis/`. The original
source layout is preserved at `archive/source-layout-v1-20260915/`. Both are
ignored, absent from a clone, and available separately when a historical replay
is needed. Full export regeneration also requires the original datasets and a
matching scientific environment.

Do not rewrite the retained data receipts to make newer source code match old
results. Ordinary slide editing uses the existing assets.

The current written methods summary is `paper/methods-brief.tex` at the
repository root. `abstract/cassis2026.tex` preserves the earlier submitted
abstract's evidence snapshot.
