# Betty fastest-lane decision, 2026-08-22

This is a read-only operational capacity record for the pilot10 attempt-7
execution design. It changes no scientific input, configuration, seed,
selection rule, or analysis rule, and no predictive score was inspected.

Betty's `wharton-dgx-b200` policy exposes two different limits. `MaxTRESPA`
allows this account at most four GPUs concurrently. The QoS-wide `GrpTRES`
value is 16 GPUs shared by all accounts; it is not a 16-GPU allowance for
`ajw-wharton`. The `wharton-genoa` `MaxTRESPA` CPU ceiling is 128.

All three probes ran the same 4,096-example, one-selector-epoch,
one-refit-epoch workload from checksum-identical
`hpc/betty/benchmark_models.py`:

| Job | Resource | CPUs / memory | Set Transformer | Zoo CNN | Total | Four-lane throughput |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 7776667 | MIG45 | 6 / 56G | 17.045494149 s | 12.175264271 s | 29.220758420 s | 0.136888986 workloads/s |
| 7776669 | MIG90 | 14 / 112G | 15.092121154 s | 10.766169019 s | 25.858290173 s | 0.154689269 workloads/s |
| 7776670 | full B200 | 28 / 224G | 14.875810969 s | 11.766304546 s | 26.642115515 s | 0.150138227 workloads/s |

At the account's equal four-GPU ceiling, MIG90 is 13.0034% faster than MIG45
and 3.0312% faster than a full B200 for this workload. The selected layout is
therefore four MIG90 lanes at 14 CPUs each plus six CPU-only lanes at 12 CPUs
each: `4 * 14 + 6 * 12 = 128` CPUs and exactly four GPUs.
The launch wrappers are separately checksum-bound as
`hpc/betty/mig90_14cpu.sbatch` and `hpc/betty/cpu.sbatch` in the machine
record; the recorded probe wrapper remains the exact wrapper that produced
job 7776669.

There was no reason to wait for a reset. The account had zero active jobs,
at least seven MIG90 slices were free, partition capacity was available,
monthly usage was approximately 51.9% with headroom, and no MaxJobs,
MaxSubmitJobs, or GrpTRESRunMins limit applied. Waiting cannot increase the
account's four-GPU `MaxTRESPA` ceiling.

The complete machine record, including exact timings, allocations, job IDs,
and source hashes, is `hpc/betty/capacity/2026-08-22_fastest_lane.json`.
