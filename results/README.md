# Research results

This directory retains current result summaries at their original paths so
paper scripts, presentation preparation, and checksum records continue to agree.

- `rushing/full100-confirmatory-20260820/`: completed legacy four-model rushing
  campaign, including summary metrics and its recorded design/provenance.
- `bdb/`: cross-year task aggregates and paired receipt/checksum files.
- `rushing/exploratory-token-cnn/`: explicitly exploratory ablation output;
  it is separate from confirmatory evidence.

Keep each retained payload with its `.sha256` and receipt files. Names such as
`full100_serial_*_attempt1` are evidence identifiers; do not rename or combine
them for presentation. Make a derived table or figure instead.

The folder-local `.gitignore` excludes large per-cell artifacts, execution
logs, and row-level prediction/reliability exports. Their absence in a clone
means it is not a complete raw-run archive. Some regeneration requires the
original local data/run directories.

Earlier single runs, five-repeat pilot sweeps, partial exports, and smoke
outputs now live locally in `archive/legacy-pilots/results/`.
