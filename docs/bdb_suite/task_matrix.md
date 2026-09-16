# Locked task matrix

| Task | Development/main games | Cutoff and target | Primary score | Main anchors |
|---|---:|---|---|---|
| BDB2021 completion | 30 / 223 | First `pass_forward`/`pass_shovel`; C vs I/IN | Brier | 10,20,40,60,100,130 |
| BDB2022 punt return | 90 / 622 | Earliest `punt_received`; return yards on −20…110 | CRPS | 10,20,40,60,160,360 |
| BDB2023 sack | 20 / 102 | Snap; official S vs C/I/IN/R | Brier | 10,20,30,40,50,60 |
| BDB2024 tackle candidate | 30 / 106 | Ten frames before inferred candidate event; solo tackle vs PFF miss | Brier | 10,20,30,40,50,60 |
| BDB2025 coverage | 30 / 106 | Final 20 pre-snap frames; Man vs Zone | Brier | 10,20,30,40,50,60 |
| BDB2026 trajectory | 36 / 236 | Official input endpoint; masked future x/y with a jointly development-selected residual-or-absolute neural target | RMSE | 10,20,40,60,100,140 |

## Separate retrospective harmonization

| Task | Development/main games | Cutoff and target | Primary score | Main anchors |
|---|---:|---|---|---|
| BDB2020 rushing harmonized | 40 / 648 | Supplied handoff snapshot; rushing yards on −28…51 | CRPS | 20,40,80,160,240,360 |

The [September 6, 2026 amendment](../../configs/bdb_suite/protocol_amendments/20260906_bdb2020_harmonized_five_role.json)
authorizes `bdb2020_rushing_harmonized` as a separate retrospective five-role
`full100` task that skips `pilot10`. It does not overwrite or relabel the
completed, read-only legacy `bdb2020_rushing` evidence and does not silently
enter or replace the existing seven-task synthesis. Its successor automatic
start is false; there is no BDB2027 task.

BDB2024 and BDB2025 share the same development and confirmatory game
registries.  Their outputs remain separate tasks and are not independent
seasonal evidence.

For BDB2026, grouped development masks have positive future-horizon support
through H33. The frozen scale learner fits its weighted PAVA curve on that
observed prefix and carries the final fitted level through H94; game-level
split-conformal calibration supplies the marginal whole-path coverage step.
