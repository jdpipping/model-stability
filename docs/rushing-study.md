# Confirmatory rushing study

This guide records the four-model BDB2020 study. Its scientific design and
completed results remain separate from the cross-year suite.
See [the project overview](../README.md) for current entrypoints.

## Source layout and recorded runs

Shared neural models now live in `rushing_study/neural_models.py`; probability
normalization and conformal interval helpers live in `rushing_study/intervals.py`.
The maintained data preparation and environment setup commands are under
`scripts/`. Superseded pilot sweep commands and their worker remain only in the
local ignored archive; the current study does not import or bind them.

The September 2026 source migration changes the code and configuration identity.
It preserves the model definitions, interval calculations, seeds, splits,
hyperparameters, and analysis choices. Existing completed-run manifests and
results retain their original bytes and hashes. They require the matching
archived source snapshot for reproduction; the current source tree deliberately
does not admit them as if they were planned against the new layout. New runs need
new preflight evidence and manifests for the current source identity.

The local pre-migration source snapshot is
`archive/source-layout-v1-20260915/`. Historical source and pilot commands are
ignored and are not distributed with the maintained repository. For historical
reproduction, use the separately retained matching source, recorded environment,
and data; do not rewrite completed manifests to match the current checkout.

## Data preparation

Run these from the repository root. Both forms work without setting `PYTHONPATH`:

```bash
python scripts/prepare_data.py --help
python -m scripts.prepare_data --help
```

Without `--help`, preparation reads `data/raw/train.csv` and writes processed
arrays under `data/processed/`. The macOS environment setup command is
`./scripts/create_venv.sh`; it recreates `.venv/` and installs `requirements.txt`.

## Design and execution

The confirmatory rushing-yards study is separate from the earlier five-repeat pilot. The default profile uses 50 new, season-stratified whole-study reruns; `plan --full` selects the locked 100-repeat profile for a denser empirical distribution. Both profiles use frozen L2 one-vs-rest logistic, LightGBM, Zoo CNN, and Set Transformer models with paired training sizes of 20, 40, 80, 160, 240, and 360 games. The complete rerun is the inferential unit.

Install the recorded environment for the study with `python -m pip install -r requirements-lock.txt`. Reproducibility is enforced by exact hashes of every declared study-code and input-data file; unrelated workspace state is not an admission input. The recommended hybrid profile runs the two tabular models in 12 one-thread CPU workers while one GPU worker runs the two neural models at the same time. It requires a TensorFlow-visible GPU when the manifest is planned and run. Start with the required one-epoch smoke preflight in a distinct directory outside the repository:

```bash
python -m rushing_study plan --smoke --hybrid --smoke-neural-epochs 1 \
  --run-dir /tmp/rushing-confirmatory-smoke
python -m rushing_study run --run-dir /tmp/rushing-confirmatory-smoke \
  --branch main --repeats 1 --anchors 20 --resume
python -m rushing_study verify-determinism \
  --run-dir /tmp/rushing-confirmatory-smoke --model zoo_cnn --n-train 20 \
  --rtol 0 --atol 0
python -m rushing_study verify-determinism \
  --run-dir /tmp/rushing-confirmatory-smoke --model set_transformer --n-train 20 \
  --rtol 0 --atol 0
```

Freeze the definitive manifest only after that preflight succeeds. The planner verifies identical declared code-file hashes, data hashes, environment, and hardware, then embeds the four-cell smoke evidence and exact determinism receipts for both neural models. Later edits to unrelated files are permitted; edits to any hash-bound study file still fail loudly:

```bash
python -m rushing_study plan --hybrid \
  --config configs/rushing_confirmatory_v1.json \
  --preflight-run-dir /tmp/rushing-confirmatory-smoke
```

For the 100-repeat full profile, add `--full` to both the smoke planner and the definitive planner. A smoke run from one profile cannot attest the other:

```bash
python -m rushing_study plan --smoke --full --hybrid --smoke-neural-epochs 1 \
  --run-dir /tmp/rushing-confirmatory-smoke-full
# Run the same four-cell smoke and both determinism commands against that directory.
python -m rushing_study plan --full --hybrid \
  --config configs/rushing_confirmatory_v1.json \
  --preflight-run-dir /tmp/rushing-confirmatory-smoke-full
```

`plan` prints the content-addressed run directory. Use that exact directory for every later command:

```bash
python -m rushing_study run --run-dir <RUN_DIR> --branch main --resume
python -m rushing_study status --run-dir <RUN_DIR>
python -m rushing_study run --run-dir <RUN_DIR> --branch sensitivity --resume
python -m rushing_study aggregate --run-dir <RUN_DIR> --require-complete
```

If the prespecified stage-one sensitivity decision triggers an extension, run the full sensitivity branch and aggregate again:

```bash
python -m rushing_study run --run-dir <RUN_DIR> --branch sensitivity \
  --extend-sensitivity --resume
python -m rushing_study aggregate --run-dir <RUN_DIR> --require-complete
```

After the full100 run is finalized, the optional game-clustered 90% interval
companion is generated without fitting or refitting anything. It opens only
the saved, marker-validated calibration/test probabilities and histories,
uses one frozen semantic seed per repeat to select one calibration play per
game, and writes a separate immutable/checksummed artifact. The locked local
conformal interval above remains the primary rushing interval.

```bash
python -m rushing_study cluster-interval-sensitivity \
  --run-dir <FINALIZED_FULL100_RUN_DIR> \
  --output-dir <NEW_CLUSTER_INTERVAL_OUTPUT_DIR>
python -m rushing_study verify-cluster-interval-sensitivity \
  --run-dir <FINALIZED_FULL100_RUN_DIR> \
  --output-dir <NEW_CLUSTER_INTERVAL_OUTPUT_DIR>
```

The destination must not exist and must be disjoint from the source run. The
receipt binds all 2,400 source cell markers, per-cell selected calibration
registries and padding values, example-equal and game-equal coverage/width,
the summary, and the source run-level marker.

For a hybrid manifest, `run` without `--queue` starts both locked queues concurrently: `cpu_tabular` owns L2 one-vs-rest logistic and LightGBM with 12 workers, while `gpu_neural` owns Zoo CNN and Set Transformer with one worker. Every queue reads the same split manifests, so comparisons remain paired. The parent verifies the complete TensorFlow/GPU inventory once; CPU children re-verify the remaining frozen runtime contract and avoid importing TensorFlow, since their two locked implementations are CPU-only. Resume is cell-safe across both queues, and `status` reports queue-specific counts and approximate wall-clock ETAs. To operate or debug one queue separately, use exactly `--queue cpu_tabular --workers 12` or `--queue gpu_neural --workers 1`; ordinary runs should omit both options. The sensitivity branch uses the same routing. Omit `--hybrid` from both planning commands to retain the locked single-worker sequential profile.

Each cell is finalized atomically with SHA-256 checksums. Manifests contain exact game partitions, nested subsets, all derived seeds, data/code/dependency provenance, and deterministic runtime settings. Cell artifacts retain calibration/test probabilities, prediction-level interval and CRPS fields, model history, and sensitivity candidate scores. Final analysis validates the exact 1,200-cell default grid or 2,400-cell full grid, reports both play-weighted and game-equal metrics, and performs the prespecified 10,000-draw repeat-block bootstrap. Large cell artifacts stay outside the source tree; archive the run directory with the study release.

The 50 or 100 reruns estimate stability of the complete modeling procedure conditional on the observed 688-game dataset. The empirical rerun distribution is the primary stability evidence. The repeat-block bootstrap is a secondary inferential wrapper over those completed reruns; it does not resample plays or refit models. Independent future BDB seasons are still needed for external replication.
