# State Forecaster — Project State Summary (v7)

## 1. Purpose of this document
Self-contained reference for the **State Forecaster** application: purpose, architecture, design decisions, current status, features, assumptions, limitations. Audiences: (a) the project lead / original NN developer; (b) future development sessions. **v7 supersedes v6.** The v6→v7 delta is the **productionization pass**: the single prototype script was refactored into a clean 6-file module structure, Black-formatted and fully docstring/ented; a per-horizon-step MAE feature was added; and — most significantly — the inter-process **shutdown was redesigned** around a single Event, eliminating a class of intermittent shutdown hangs. A concise "what changed since v6" digest for colleagues is at the end (§15).

## 2. What the app is and where it sits
Forecasts near-future distribution-system state (per-phase-node **Voltage magnitude (per-unit)** and **Angle (radians)**) from a stream of state estimates. Production pipeline:

```
GridLAB-D / OpenDSS (simulation measurements)
  → [optional: sensor-simulator service — Gaussian noise; NOT run currently]
  → State Estimator (existing C++ app; sparse-matrix EKF; publishes to GridAPPS-D ActiveMQ bus;
    also writes its published estimates directly to a .jsonl file)
     → State Forecaster (THIS app; Python; subscribes, forecasts, publishes forecasts to the bus)
        → other ADMS apps consume forecasts
```State Estimator outputs** (P, Q, V, Angle per node), one bus message per timestamp.
- **Origin:** a validated single-process NN script (colleague's, for approach validation + a journal paper). This project turned it into a bus-integrated, streaming, multi-process ADMS service.
- **Query-free** — no CIM calls; node set is derived from the estimate stream (§9.1).

**Test grids:** IEEE 13 Node → **41 phase-nodes**; IEEE 123 Node → **274 phase-nodes**. 9500-node **out of scope** (§10).

## 3. Current status (headline)
- **Productionized multi-process streaming app**, GridAPPS-D bus-integrated (input + output), validated end-to-end on live data (including a full 14-day, 274-node run) and now organized as a clean 6-module codebase.
- **Runs on a Windows Hyper-V Ubuntu VM** (the earlier VirtualBox instability that caused "heartbeat timeout" hangs is gone).
- **Shutdown is now simple and race-free** (single `sim_done` Event; see §7) — the intermittent shutdown-hang saga is resolved.
- **At a clean milestone.** Remaining work is future-facing (§13), not blocking.

## 4. Why the single-process → multi-process transformation was a major effort
Same forecasting math as the original script, under fundamentally different constraints: the original had **global, upfront knowledge** (whole dataset in memory); streaming has **none**. Removing global knowledge forced re-engineering of nearly every subsystem:

| Concern | Original (whole file) | Now (streaming, 3 processes, bus) |
|---|---|---|
| Data availability | Entire dataset in memory | One timestamp at a time; **rolling buffer** + retention/eviction |
| Normalization | Fit once on full dataset | **Incremental/causal scalers** per block |
| Node discovery | Scan whole file | Lazy, from **first streamed record**; no CIM |
| Control flow | Sequential loop | **Concurrent processes** + queues + Event + clean shutdown |
| Train vs. forecast | Interleaved | **Decoupled processes** in parallel |
| Model sharing | Same in-memory object | Weights **+ scaler state** serialized across process boundary |
| Forecast unit | One batched forecast per block | **One forecast per incoming estimate** (incremental normalization) |
| History for a forecast | Trivially available | Forecaster keeps its **own gapless recent-history buffer** |
| Missing timestamps | N/A | **Imputation** to preserve position-based indexing |
| Data source | File | **File OR live GridAPPS-D bus** (selected by `gappsd_simid`) |
| Accuracy measurement | MAE vs. known file actuals | **Deferred scoring** vs. later-arriving estimates |
| Shutdown | Loop ends | **Event-driven** (`sim_done`), abandon queues via cancel_join_thread |

Concurrency also added IPC semantics, process lifecycle/shutdown, cross-process model transport (a real PyTorch shared-memory pitfall — §7), and re-validation against byte-identical baselines after each change.

## 5. Development history (condensed)
1. Forecast after every block; functions.
2. Removed MC-dropout + Excel export → fast single forward pass.
3. **Step A** — CSV+pandas → line-delimited JSON generator; removed `deg2rad`. *Byte-identical to CSV.*
4. **Step B** — streaming ingestion: RollingBuffer, incremental scalers, forecast-then-train, warm-up. *Matched Step A.*
5. Per-node forecast JSON.
6. **Change 3** — three-process split; block-1 losses byte-identical to single-process; shared `assemble_input_vector`.
7. `TS_INCREMENT_SEC` flexibility (300/900 s validated).
8. Gappy-data imputation (linear interpolation).
9. Real SE format + units (`SvEstVoltages`, `angleRad`, `vpu`); output splits `ConnectivityNode`+`phase`.
10. Guards: insufficient-data graceful exit; loud cadence-mismatch banner.
11. GridAPPS-D integration: bus subscribe + 3-level unwrap; per-record imputer (`make_imputer`); feeder file/bus driver-split; forecaster publish + `Forecast`-wrapped message.
12. Deferred live-MAE scoring; first live runs; Hyper-V migration; platform bridge-delay fix + throttling model.
13. **Productionization (v7):** 6-file module split; Black (line-length 80); full docstring/comment pass; per-step MAE (`COMPUTE_MAE_STEPS`); **shutdown redesign** (single Event).

## 6. Runtime cadence & configurable increment
- **`TS_INCREMENT_SEC`** (in `forecast_common.py`) = estimate spacing. Default **60 s**; validated at **300** and **900** s with no code changes (forecast horizon, lags, block boundaries, memory all derive). Forecast horizon = `FUT × TS_INCREMENT_SEC`.
- Must match the true data cadence and divide 86,400 / 604,800 — **enforced at runtime by the cadence guard** (loud one-time banner on mismatch; §8b).
- **Throttling model:** with the platform's GOSS-HELICS bridge-delay fix, data can arrive far faster than realistic. Unthrottled runs may finish before much forecasting occurs (data outruns training) — they are feeder/shutdown tests. **Throttled runs are the real forecast validation.** File driver paces via `FEED_RATE_HZ`; the live State Estimator is paced by a `usleep` on publish (interim, pending a platform `interval` fix).
- **Block size (`BLOCK_DAYS`=2) vs. training window:** the 2-day block is the *retrain cadence*, not the training window — each retrain trains on the whole retained buffer (~`RETENTION_DAYS`=10). So block size trades startup latency / first-model adequacy / retrain cost; the daily & weekly trends are captured by the *retention window*, not the block size.

## 7. Architecture & non-trivial design decisions
**Three worker processes** launched by `state_forecaster.py`:
- **data_feeder** (`forecast_feed.py`): reads estimates (bus or file), imputes g consumers. Torch-free.
- **trainer** (`forecast_train.py`): accumulates 2-day blocks, trains the DNN, publishes model snapshots.
- **forecaster** (`forecast_predict.py`): ingests estimates, forecasts the latest real timestamp with the newest model, publishes/records forecasts, scores deferred MAE.

**Queues + Event:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot (`model_q`) | Trainer | Forecaster | latest-only (drain, keep newest) |
| Data → Trainer (`train_data_q`) | Feeder | Trainer | keep-all FIFO |
| Data → Forecaster (`fc_data_q`) | Feeder | Forecaster | keep-all FIFO; forecasts only latest real ts |
| **`sim_done` (mp.Event)** | Feeder (set) | Trainer + Forecaster (check) | **single shutdown signal** |

- **Latest-only forecasting lives inside the forecaster:** ingest every record (gapless history), forecast only the newest real one.
- **Model handoff = serialized bytes blob** (`torch.save`→`BytesIO`), NOT `share_memory()` (which risks half-updated reads + training-loop locking). *Real bug fixed earlier:* live tensors on a queue use sender-owned shared-memory FDs → receiver fails once the sender exits. Bytes have no such lifecycle.
- **Scaler state travels WITH the model** (`extract_scaler_state`/`apply_scaler_state`); the forecaster never fits scalers.
- **`spawn` start method** (CUDA-safe); each child makes its own CUDA + GridAPPS-D connection. Ctrl-C in main terminates children.
- **Lazy, stream-self-describing init** (`sorted(node the first record); no CIM.
- **Forecaster efficiency:** incremental normalization (normalize one row per estimate); full re-normalize only on snapshot arrival (rare).

**SHUTDOWN REDESIGN (v7) — the key architectural change, and its hard-won lesson.**
The prior design used a queued `DONE` sentinel on every queue plus sticky per-consumer flags. This suffered an **intermittent shutdown hang**: the trainer put `DONE` on `model_q` then called `model_q.cancel_join_thread()` and exited — but `cancel_join_thread` tells the queue's feeder thread *not to flush pending puts on exit*, so the `DONE` could be **discarded before delivery**, leaving the forecaster spinning forever waiting for a sentinel that never arrived. (Diagnosed via an infinite-loop trace showing `model_done=False` with an empty `model_q`.)

**The fix was to simplify, not patch:** a single shared **`sim_done` Event** is now the *only* shutdown signal for both trainer and forecaster. No queue carries a `DONE` sentinel.
- **Feeder:** sets `sim_done` **after** enqueuing all real records (ordering matters — see §8b tail note), then `cancel_join_thread()` on its data queues.
- **Trainer:** uses a **timeout `get()`** (so a blocked trainer wakes periodically to re-check `sim_done` — critical, since there's no DONE to unblock it). On `sim_done`: trains no further blocks, and if the sim ended *during* an in-flight block, **suppresses that snapshot push** (no consumer for it), then exits.
- **Forecaster:** exits when **`sim_done` is set AND its data queue drains empty** (and no pending snapshot). No sticky flags.

**Why this is robustly better:** an Event cannot be lost in transit (unlike a queued item that a premature `cancel_join_thread` can discard), so the entire lost-sentinel race class is *eliminated, not reduced*. And a clean invariant emerged:

> **`cancel_join_thread()` is only safe on queues you *consume and abandon*, never on a queue you *produce a still-needed message to*.** Removing the queued DONE removed the only such must-deliver message, so `cancel_join_thread()` is now unconditionally safe on every queue at exit.

Validated with ~5 clean runs each of file and bus drivers (the bug had reproduced ~1-in-3, so clean repetition matters; and the fix is a structural elimination, not a probability reduction).

## 8. Data format, units, MAE, output

**Live bus message (from SE) — 3-level nesting:** `message` (top; may carry `processStatus:"COMPLETE"`) → `message` (inner) → `Estimate` (has `timeStamp` + `SvEstVoltages: [entries]`). Each entry: `ConnectivityNode` (CIM mRID UUID), `phase`, `P`, `Q`, `v`, `vpu`, `angle`, `angleRad` (+ variances, ignored). The SE writes the `Estimate` structure directly to the `.jsonl` file, so file and bus paths share entry parsing.

**→ internal record `{"timestamp", "nodes": {key: {P,Q,V,Angle}}}`:** key = `ConnectivityNode + "." + phase` (Approach 3 — combined key is the NN identity, split back at output). Phase as-is, no mapping. `vpu`→V (per-unit), `angleRad`→Angle (radians); `P`/`Q`=="NA"→0.0; V/angle not NA-coerced. Shared `parse_sv_entry`/`sv_entries_to_nodes` (in `forecast_feed.py`) are the single source of truth for entry interpretation.

**Why per-unit V:** a single global MinMax scaler across all nodes; per-unit keeps every node ~0.95–1.05 (physical volts span multiple bases and would compress load nodes into noise). **Why radians:** pipeline is radians-native; SE publishes `angleRad`.

**Published forecast message:** top-level `timestamp` + `simulation_id`; forecast payload nested under **`Forecast`** (`step_sec`, `horizon`, `forecast_times`, `nodes`). Published via `gapps.send(service_output_topic('state-forecaster', simid), ...)`. On exit the forecaster publishes its own `processStatus:"COMPLETE"` (symmetry with the SE; verified consumable by a subscriber utility).

**§8b. Guards & the tail note.**
- **Insufficient-data:** stream ends before a full block → skip training, exit cleanly.
- **Cadence guard:** min spacing of the first ~20 real records = true cadence; loud banner on mismatch with `TS_INCREMENT_SEC`. Retired a footgun that twice produced plausible-but-wrong results.
- **Shutdown tail note:** `sim_done` is an Event (no FIFO ordering with queued records), set *after* the feeder's last data put. In principle a just-enqueued tail record could still be in transit when the forecaster sees the Event and exits, dropping the final forecast or two — **benign** (throttled: records arrive seconds apart; fast: forecasting mostly skipped; and end-of-sim tail forecasts have no consumer). Hardening (two consecutive empty drains) is noted but deliberately **not** implemented.

**§8c. MAE (both deferred; forecaster only).**
- **Aggregate deferred MAE** (`COMPUTE_LIVE_MAE`): a forecast can't be scored when made (its future hasn't arrived), so one forecast per model version is held and scored against the actual estimates that later arrive. Logs `[MAE] vN ... Voltage MAE / Angle MAE (node-steps)`.
- **Per-step MAE** (`COMPUTE_MAE_STEPS`, e.g. `[1, 8, 15]`, 1-based, `None` disables): additionally reports MAE at specific horizon steps on a `[MAE-STEPS]` line, to see accuracy degrade further out. Startup-asserted to be within 1..FUT.
- **Caveat:** single-sample-per-version → noisy version-to-version; stability across versions is the meaningful read, and it measures forecast-vs-*estimate* agreement (not vs. physical truth).
- **`forecast_output.jsonl`:** every published forecast appended (one JSON/line) for offline validation/plotting; open/append/close per write (durable, no handle held); large on long runs; `None` disables.

## 9. Assumptions & limitations
1. **Fixed, complete node set every timestamp** — lazy init assumes it. True for GridLAB-D estimates. (Node-dropout → would require CIM queries; §13.)
2. **Grid-aligned timestamps** — gaps handled by imputation; misaligned passed through with a warning.
3. **No imputed/real flag reaches the model** — interpolated history treated as real.
4. **Large-gap fidelity** — linear interpolation degrades over long gaps; better handled by matching `TS_INCREMENT_SEC` to true cadence.
5. **Config must match data cadence** — enforced by the cadence guard.
6. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training set (memory + train time); confirmed engaging from block 4 at 274 nodes.
7. **Deferred MAE = single-sample-per-version, forecast-vs-estimate** (§8c caveat).
8. **Recurring "hard window" anomaly** — one window periodically shows higher error (esp. angle); data-driven (load profile), not code.
9. **Per-unit forecast output** — physical-unit output (if ever needed) via nominal-voltage dict or stream-derived `v/vpu` multipliers. Deferred.
10. **Exotic-phase feature encoding** — `encode_phase` handles single-char `1/2/3` and `A/B/C`; multi-char (`s1`, `ABC`) would mis-encode. Out of scope.
11. **Throttling required for meaningful forecast validation** — unthrottled runs are feeder/shutdown tests.
12. **`usleep` throttle in the SE is interim** — pending a platform `interval` fix (currently broken below `interval=60`).
13. **No sensor-simulator noise in current live runs** — makes live MAE somewhat optimistic vs. file baselines.
14. **>2-week simulations need a load profile longer than 2 weeks** — else data repeats and lag features lose meaning.
15. **Long-run platform memory leak** — very long sims (e.g., a 1-year run) bog down measurement delivery over time (a *platform* issue, not the app's); the app itself sustained a year-long run (snapshot v168+).
16. **`gapps.send(done_json)` at forecaster shutdown is bus-only** — never confirmed problematic, but if a bus-only shutdown hang ever recurs after the Event redesign, this send is the prime suspect (the file driver can't hit it, `gapps` being None).

## 10. Scope boundary (deliberate)
Centralized Forecaster suits ~123-node/274-phase-node (likely low thousands). **Not** 9500-node: the SE scales centrally via grid **sparsity**; the Forecaster is a **dense** whole-network model with no sparsity to exploit → training time binds. 9500-node needs **regional decomposition** — out of scope. Memory target 32–48 GB (comfortable at 274 nodes; retention bounds memory).

## 11. Validation reference numbers
- **13-node file baseline Voltage MAE (pu):** `0.001248, 0.001183, 0.001566, 0.001179, 0.001220, 0.001233`; 5-min baseline `0.002133, 0.001913, 0.002281, 0.002009, 0.002046, 0.002146`. All format/unit/refactor changes confirmed byte-identical to baselines.
- **123-node single-process** V MAE ~0.0007–0.0019 pu.
- **Live 274-node, throttled 5-min, full 14-day sim:** trainer kept up (6 blocks, cap engaging from block 4); forecaster memory plateaued at the retention horizon (~789k rows); deferred MAE in-band across versions; clean shutdown.
- **Per-step MAE** (validated on matched cadence): clean angle degradation with horizon (v1: A 0.01358→0.01732 over steps 1→15); voltage near its noise floor so noisier — as expected.
- **Shutdown redesign:** ~5 clean runs each, file + bus.

## 12. Code layout (`state-forecaster/` directory; prototype preserved separately)
- **`state_forecaster.py`** — entry point / launcher: creates queues + `sim_done`, spawns the 3 processes, manages lifecycle. (Invoked script; uses `torch.multiprocessing` for spawn only — no NN work.)
- **`forecast_common.py`** — torch-free shared utilities: `utc_str`, `setup_logger`, `LOG_DIR`, `TS_INCREMENT_SEC`.
- **`forecast_dnn.py`** — shared DNN core (torch): model/feature config, `time_features`, `encode_phase`, `assemble_input_vector` (+ `INPUT_DIM`/`OUTPUT_DIM`), incremental scalers + extract/apply, `RollingBuffer`, `DNN`, `build_model`.
- **`forecast_feed.py`** — data_feeder process (torch-free): parse helpers, `read_json_records`, cadence warning, `make_imputer`, `feeder_proc` (bus + file drivers, shared `emit`).
- **`forecast_train.py`** — trainer process: `ReplayDataset`, `make_loader`, `train_block`, `snapshot_to_bytes`, `trainer_train_block`, `trainer_proc`.
- **`forecast_predict.py`** — forecaster process: `build_forecast_json`, `drain_latest`/`drain_all`, `normalize_row`, `NormStore`, `forecast_latest`, `forecaster_proc` (+ deferred/per-step MAE).
- **`util/`** — `drop_timestamps.py`, `check_imputation.py`. (CSV/simplified converters retired — SE writes `.jsonl` directly.)
- Black-formatted (line-length 80); full docstrings + why-comments; dev-history archaeology removed.
- **Environment:** Windows host + Hyper-V Ubuntu 22.04 VM (4 vCPUs, dynamic 24–48 GB).

## 13. Roadmap / future considerations
**Near-term:** run the sensor-simulator (noise) for a "full scenario"; longer-than-2-week load profiles for multi-week runs; optionally a forecast-topic subscriber as a standing consumer.
**Larger / deferred (deliberately not designed for now):**
- **Missing-node imputation** — needs an authoritative node list → **CIM queries** (would also enable mRID→human-readable-node-name mapping for output). Main thing that trades away the query-free elegance.
- **Train-only / persist model to a DB** to warm-start forecasting (skip the initial block wait). Localized change at the trainer's shutdown/snapshot path.
- **Higher-fidelity MAE** (average many forecasts/version) if a robust per-version number is needed.
- **Imputed/real validity feature** + retrain.
- **Suppress-flag** for `forecast_output.jsonl` for production speed.
- **GridAPPS-D app packaging** (registration, config file, containerization per platform conventions).

**Design conventions to preserve:** validate against byte-identical baselines where possible; keep query-free until node-dropout forces CIM; feeder = single source seam (file vs. bus on `gappsd_simid`); **single `sim_done` Event as the only shutdown signal**; `cancel_join_thread()` only on consume-and-abandon queues; defer structural elegance to productionization but not functional correctness; simple over clever.

## 14. Key configuration parameters (now distributed across modules)
| Param | Module | Default | Note |
|---|---|---|---|
| `TS_INCREMENT_SEC` | common | 60 | estimate spacing; validated 300/900; must match data cadence & divide 86,400/604,800 |
| `LOG_DIR` | common | "./logs" | per-process logs + forecast_output.jsonl live here |
| `HIST` / `FUT` | dnn | 15 / 15 | input history / forecast horizon (samples) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | dnn | 86,400 / 604,800 | lag features |
| `RETENTION_DAYS` | dnn | 10 | rolling buffer horizon (≥ week-lag + block span; whole-week + block-multiple) |
| `MAX_WINDOW_SAMPLES` | dnn | 500,000 | per-block train-set cap (memory + train time) |
| `DROPOUT_P` | dnn | 0.03 | training regularization |
| `SEED` | dnn | 42 | reproducibility (re-seeded per spawned child) |
| `USE_GPU` / `DEVICE` | dnn | — | CUDA if available |
| `BLOCK_DAYS` | train | 2 | retrain cadence (NOT the training window; §6) |
| `EPOCHS_PER_BLOCK` | train | 8 | + early stopping (patience 2) |
| `BATCH_SIZE` / `VAL_FRACTION` | train | 512 / 0.05 | |
| `TRAINER_POLL_SEC` | train | 0.1 | blocked-get timeout to re-check sim_done |
| `JSON_PATH` | feed | — | file-driver input (uncomment per model/increment) |
| `FEED_RATE_HZ` | feed | 50 (test) | file-driver pacing; dev convenience |
| `FEEDER_POLL_SEC` | feed | 0.05 | bus-driver idle poll |
| `CADENCE_CHECK_SAMPLES` | feed | 20 | records sampled for cadence guard |
| `COMPUTE_LIVE_MAE` | predict | True | deferred per-version MAE |
| `COMPUTE_MAE_STEPS` | predict | [1,8,15] | per-horizon-step MAE (1-based; None disables) |
| `FORECAST_OUTPUT_JSONL` | predict | logs/…jsonl | published-forecast record file; None disables |
| `FORECAST_LOG_EVERY` | predict | 200 | forecast log heartbeat (publish/file not throttled) |
| `FORECASTER_POLL_SEC` | predict | 0.05 | idle poll |
| `gappsd_simid` | (CLI arg) | — | GridAPPS-D sim id → bus mode; absent → file mode |

## 15. What changed since v6 (digest for colleagues)
> **State Forecaster — v6 → v7 changes**
>
> The prototype was **productionized** with no change to forecasting behavior:
>
> 1. **Refactored the single ~1,400-line script into 6 modules** by responsibility — `state_forecaster.py` (launcher), `forecast_common.py` (shared utils), `forecast_dnn.py` (shared model/buffer/scaler core), `forecast_feed.py` (data feeder), `forecast_train.py` (trainer), `forecast_predict.py` (forecaster). The feeder and common modules are deliberately torch-free; the model core is shared by trainer and forecaster. New `state-forecaster/` directory; the old two-file prototype is preserved unchanged.
> 2. **Black-formatted** (line-length 80) and **fully documented** (docstrings on all functions/classes; explanatory comments where intent isn't obvious; removed stale development-history comments).
> 3. **New feature — per-horizon-step MAE** (`COMPUTE_MAE_STEPS`): reports forecast accuracy at chosen steps (e.g., 1st/8th/15th) to show how accuracy degrades further into the forecast horizon. Useful for evaluating longer horizons.
> 4. **Redesigned inter-process shutdown** to fix an intermittent hang-on-exit. Shutdown is now driven by a single shared "simulation done" Event instead of per-queue sentinel messages — simpler, and it eliminates the race that occasionally left the forecaster hanging. Verified across repeated file- and bus-driver runs.
> 5. Minor: removed now-unused code/comments and the CSV conversion utilities (the State Estimator writes the `.jsonl` input directly).
>
> **Behaviorally identical** to v6 for forecasting; the changes are structure, documentation, one new evaluation metric, and shutdown robustness. No config/interface changes for running it.
