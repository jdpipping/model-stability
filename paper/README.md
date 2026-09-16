# Paper and methods brief

The current written overview is [`methods-brief.tex`](methods-brief.tex), the
September 2026 summary of the completed BDB2020–2025 analyses. Its figures use
retained result aggregates and the conference presentation's compact exports.
This folder was previously named `working/`.

## Contents

- `methods-brief.tex`: current source; its PDF is a local build output.
- `figures/`: retained publication figures, including those required to compile
  the methods brief. PDFs in this subfolder are inputs, not ignored builds.
- `references.bib`: research bibliography.

## Build the brief

Install a TeX distribution with `latexmk` and the packages listed in the source.
From the repository root:

```sh
mkdir -p output/pdf
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/pdf methods-brief.tex
```

Build products belong in the ignored `output/` folder. Compiling beside the
source also works; this folder's `.gitignore` excludes document PDFs and the
root ignore rules exclude LaTeX auxiliary files.

## Regenerate publication figures

Plotting programs live in [`scripts/plots/`](../scripts/plots/). They cover the
completed later-year studies and the original BDB2020 confirmatory study,
including checks on their source aggregates. Use the repository's Python
environment. For example, from the repository root:

```sh
.venv/bin/python scripts/plots/plot_bdb2021_completion.py --input presentations/cassis2026/data/raw_terminal/bdb2021/primary_metrics.csv
.venv/bin/python scripts/plots/plot_bdb2022_punt_returns.py --input presentations/cassis2026/data/raw_terminal/bdb2022/primary_metrics.csv
```

These commands validate checksums and refresh the retained figures in `paper/figures/`.
Each plotter also provides `--help`. Direct script execution and module execution
both work, for example `python -m scripts.plots.plot_bdb2021_completion --help`.
The later-year plotters provide `--output-dir`. Plots stored
inside checksummed study releases should remain preserved; direct exploratory
renders to `output/` instead.

## Historical material

The ignored `archive/` folder preserves earlier work locally:

- `archive/legacy-pilots/working/`: the early pilot writeup, plotter, and figures.
  Its sibling `results/` folder preserves the old relative data paths.
- `archive/paper-drafts/`: the August 2026 prospective manuscript and a copy of
  its bibliography. It predates the later-year results and is not the current
  findings summary.

The conference abstract in `abstract/` is also an earlier evidence snapshot;
the current talk source is `presentations/cassis2026/index.qmd`. The publication
plotters and retained figures do not depend on these local archives. Historical
presentation extraction and analysis preparation tools are also archived;
current deck rendering uses the retained assets, as explained in its README.
