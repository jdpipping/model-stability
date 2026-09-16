# Betty execution kit

This directory contains reusable Slurm wrappers for running the study CLI on
Betty, plus environment and benchmark tools. The maintained study commands are
available through `python -m bdb_study --help` and the
[study runbook](../../docs/bdb_suite/README.md).

## Tools

| File | Purpose |
| --- | --- |
| `cpu.sbatch` | Run a command in the NVIDIA container on 12 CPU cores. |
| `mig45.sbatch`, `mig90.sbatch`, `b200.sbatch` | Run a command on the named GPU allocation. |
| `mig45_6cpu.sbatch`, `mig90_14cpu.sbatch` | GPU wrappers with explicit CPU allocations. |
| `common.sh` | Shared container, project-path, and deterministic-runtime settings. |
| `inventory_environment.py` | Report installed versions for a requirements file. |
| `benchmark_models.py` | Measure the retained rushing-study model implementations. |
| `requirements-overlay.txt` | Additional packages for the recorded NVIDIA image. |

Each wrapper forwards its arguments as the command to run. With no arguments,
it runs the model benchmark. For example, after configuring the project paths
on Betty:

```bash
mkdir -p output/betty-logs
sbatch --time=00:10:00 \
  --output="$PWD/output/betty-logs/%x-%j.out" \
  --error="$PWD/output/betty-logs/%x-%j.err" \
  scripts/betty/cpu.sbatch python3 -m bdb_study --help
```

Set `ZOO_BETTY_PROJECT_HOST` to the checkout on the host and
`ZOO_BETTY_PROJECT` to that checkout's path inside the container. Set
`ZOO_BETTY_STORAGE_ROOT` and `ZOO_BETTY_MOUNT` to the corresponding mount roots.
`ZOO_BETTY_OVERLAY` and `ZOO_BETTY_IMAGE` select the Python overlay and container.
The existing `ZOO_BETTY_*` variable names remain supported. Wrapper account,
partition, and log defaults describe the original Betty allocation; override
Slurm options when submitting from another allocation. Create the chosen log
directory before submission: Slurm creates log files but does not create their
parent directories. The example overrides the historical log locations.

## Historical evidence

`capacity/` and `probes/attempt9_mig45_current_pair/` preserve small immutable
records referenced by historical runtime validation. Their original paths and
checksums remain unchanged inside the records. `paths.py` explicitly resolves
relocated record paths; content validation still requires the recorded bytes.
Changed executable source is never admitted using an archived file's checksum.

Superseded pilots, campaign controllers, recovery scripts, and their dedicated
tests are kept locally under the ignored `archive/retired-betty/` folder. They
are not maintained launch interfaces or dependencies of this execution kit.
Historical source-bound campaigns require their original source checkout.

Local benchmark output and Slurm logs are excluded by [`.gitignore`](.gitignore).
