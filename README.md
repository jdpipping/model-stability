# Model Stability

How stable are predictions, uncertainty estimates, and model comparisons when
we train on different samples of games? This project studies that question
across NFL Big Data Bowl tasks using repeated, paired game-level splits.

The cross-year suite compares structural linear and boosted models, relational
neural networks, attention-based relational networks, and set transformers.
The completed reports currently cover BDB2020–2025; the BDB2026 task is defined
in the suite but is not part of the completed evidence presented in the talk.
The original four-model rushing study and the later harmonized BDB2020 task
retain separate protocols and provenance.

## Start here

- **Research summary:** [current methods brief](paper/methods-brief.tex) and
  [paper sources and figures](paper/README.md).
- **Conference talk:** [CASSIS 2026 presentation](presentations/cassis2026/README.md).
- **Study protocol and execution:** [cross-year BDB runbook](docs/bdb_suite/README.md)
  and [task definitions](docs/bdb_suite/task_matrix.md).
- **Completed rushing study:** [design and workflow](docs/rushing-study.md).
- **Repository maintenance:** [layout and ignore-rule guide](docs/repository-guide.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `bdb_study/` | Cross-year task adapters, models, experiments, and analysis |
| `rushing_study/` | Frozen four-model rushing study and interval analysis |
| `configs/` | Task specifications, protocol amendments, and dependency locks |
| `paper/` | Current manuscript source, bibliography, and publication figures |
| `abstract/` | Conference submission source, retained as submitted |
| `presentations/` | Talk source, retained figure assets, and compact supporting data |
| `results/` | Current result summaries and their checksum/provenance records |
| `experiments/` | Clearly separated exploratory work |
| `scripts/` | Data preparation, plotting, presentation tools, and reusable Betty wrappers |
| `tests/` | Study, execution, and analysis checks |
| `docs/` | Scientific runbooks and repository maintenance |

`internal/` holds active private notes, planning, and maintenance records.
`archive/` holds retired work. Both stay local and are ignored by Git, along
with raw `data/`, local environments, and generated output. The archive contains old pilots, drafts, reference
notebooks, exports, and the unrelated fantasy-schedule project. It is local
storage, not a backup or a published research release.

Core implementation lives in `bdb_study/` and `rushing_study/`; standalone
commands live in [`scripts/`](scripts/README.md). The Betty kit contains reusable
scheduler wrappers and environment tools. Superseded pilot launchers, failed
attempts, and one-off recovery controllers are preserved only in the ignored
archive, with their dedicated tests and historical records.

Completed result records retain their original hashes. The reorganized code
has a new source identity; see [the migration notes](docs/code-layout-migration.md)
before attempting to resume an old campaign.

## Environment and commands

Use Python 3.12. The cross-year suite and the completed rushing campaign have
separate recorded environments. For a new cross-year checkout, create a
separate environment using the suite lock:

```bash
python3.12 -m venv .venv-bdb
source .venv-bdb/bin/activate
python -m pip install -r configs/bdb_suite/requirements-lock.txt
python -m bdb_study --help
```

The lock reflects the recorded study environment; GPU/container setup is
covered in the [runbook](docs/bdb_suite/README.md) and [cluster guide](scripts/betty/README.md).
Raw competition data and full per-cell run artifacts must be supplied
separately. Planning and resuming a campaign require matching data, code,
environment, and hardware receipts. Follow its runbook before executing jobs.

The legacy rushing environment uses `requirements-lock.txt`; its CLI is
`python -m rushing_study --help`. `requirements.txt` and `scripts/create_venv.sh`
remain the original workstation setup, not a replacement for either frozen lock.

Both suites provide unittest checks. Run the cross-year checks from the root
in the suite environment:

```bash
python -m unittest discover -s tests/bdb_study -p 'test_*.py'
```

Plotting checks use the local scientific Python dependencies and do not fit
models. Building the talk from its checked-in figures is separate from
regenerating every figure from raw data; see the presentation guide.
