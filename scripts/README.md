# Commands and tools

Core modeling and analysis are Python packages at the repository root.
This folder contains commands that use those packages or prepare research
artifacts.

- `prepare_data.py`: prepare the original rushing input tensors. Run from the
  repository root; use `python -m scripts.prepare_data --help` first.
- `create_venv.sh`: original workstation setup using `requirements.txt`. It
  recreates `.venv`, so use it only when that environment is disposable;
  frozen study locks and independent environments are described in the root README.
- [`plots/`](plots/): publication plotters; see the [paper guide](../paper/README.md).
- [`presentations/cassis2026/`](presentations/cassis2026/): maintained slide
  figure/render tools; see the [presentation guide](../presentations/cassis2026/README.md).
- [`betty/`](betty/README.md): reusable scheduler wrappers and environment tools.

Python commands support direct execution from the repository root; package
entrypoints can also be invoked with `python -m scripts.<module>`.
Data, logs, local notes, and generated builds follow the root and folder-local
ignore rules. Reusable implementation belongs in the study packages.

Superseded pilot runners, campaign-specific recovery code, failed attempts,
and historical extraction helpers are in the local ignored archive. They
are not part of this repository's supported command surface.
