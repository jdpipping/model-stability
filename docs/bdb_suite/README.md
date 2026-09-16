# Cross-year BDB study

The maintained implementation is `bdb_study/`. It separates task definitions,
data preparation, model fitting, uncertainty estimation, and aggregation.
See [task_matrix.md](task_matrix.md) for the task definitions and
[bdb2024_fidelity.md](bdb2024_fidelity.md) for the separate winner diagnostic.

## Scientific scope

The suite defines BDB2021–2026 and the separately amended harmonized BDB2020
rushing task. The published talk currently reports completed BDB2020–2025
evidence; the existence of a task configuration is not a claim of completion.
Linear-Structure, Boosted-Structure, RelNet, AttnRelNet, and Global Set
Transformer share paired game-level experimental designs. Model selection,
calibration, tuning, and uncertainty procedures are described in the task
specifications and [current methods brief](../../paper/methods-brief.tex).

The original four-model BDB2020 study remains distinct; see
[its workflow](../rushing-study.md). Protocol amendments under
`configs/bdb_suite/protocol_amendments/` and this directory record scientific
choices. They remain part of the research provenance.

## Environment and inspection

Use Python 3.12 and the dedicated suite lock. Do not overwrite an environment
used by a completed campaign.

```bash
python3.12 -m venv .venv-bdb
source .venv-bdb/bin/activate
python -m pip install -r configs/bdb_suite/requirements-lock.txt
python -m bdb_study --help
python -m bdb_study inventory
```

Raw competition data and machine-local source receipts are separate inputs.
The retained summaries in `results/` do not constitute a full run archive.
Use `python -m bdb_study status --run-dir <RUN_DIR>` to inspect an available
run; each command's `--help` lists its exact options.

The workflow is `import-data` → `prepare` → `dev-cv` → `freeze` → `plan` →
`run` → `aggregate-task`, with explicit determinism and runtime admission
checks where required. Follow the locked TaskSpec and its receipts; do not
infer permission to start a new campaign from an old command or job log.

For an isolated synthetic check, choose a new output directory:

```bash
python -m bdb_study synthetic-smoke --output-dir /tmp/model-stability-smoke
```

## Execution and provenance

[Reusable Betty wrappers](../../scripts/betty/README.md) can run explicit
commands under the recorded scheduler/container setup. Campaign-specific
pilots, incident recovery controllers, and failed attempts are retired in the
local ignored archive. They are not supported launch entrypoints in this
repository. Other clusters can adapt the wrappers to their own environment.

The source-layout migration changes executable identity. Completed manifests,
receipts, result payloads, and recorded hashes have not been rewritten.
Admission checks deliberately reject old manifests when current source bytes
no longer match. Replaying a completed campaign requires its original source,
data, environment, and hardware; new work needs fresh admission evidence.
Historical capacity/probe records remain available for validation, but their
presence does not authorize using changed code under an old admission.
See [the migration notes](../code-layout-migration.md).

## Tests

From the repository root in an environment with the scientific dependencies:

```bash
python -m unittest discover -s tests/bdb_study -p 'test_*.py'
```

The complete test collection, including function-style tests, can be run with
`python -m pytest tests` when pytest is available in the development environment.
