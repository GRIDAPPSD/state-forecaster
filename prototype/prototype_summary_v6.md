
# State Forecaster — Project State Summary (v6)

## 1. Purpose of this document
Self-contained reference for the **State Forecaster** application: purpose, architecture, design decisions, current status, features, assumptions, limitations. Audiences: (a) the project lead / original NN developer; (b) future development sessions. **v6 supersedes v5.** Delta over v5: resolved the "heartbeat timeout" (VirtualBox instability → migrated to Hyper-V); a GridAPPS-D platform fix removed a per-timestamp bridielding large speedups (+ a throttling model); redesigned trainer shutdown via a `sim_done` Event; fixed a multiprocessing queue-cleanup deadlock; forecaster now emits `processStatus COMPLETE` on exit; State Estimator now writes the `.jsonl` input file directly (conversion utilities retired); `.json`→`.jsonl` naming; a new forecast-output `.jsonl` file; and a **full 14-day live validation at 274 phase-nodes**.

## 2. What the app is and where it sits
Forecasts near-future distribution-system state (per-phase-node **Voltage magnitude (per-unit)** and **Angle (radians)**) from a stream of state estimates. Production pipeline:

```
GridLAB-D / OpenDSS (simulation measurements)
  → [optional: sensor-simulator service — Gaussian noise; NOT run currently]
  → State Estimator (existing C++ app; sparse-matrix EKF; publishes to GridAPPS-D ActiveMQ bus;
    also writes its published estimates directly to a .jsonl file)
     → State Forecaster (THIS app; Python; subscribes, forecasts, publishes forecasts to the bus)
        → other ADMS apps consume forecasts
```
- Ingests **State Estimator outputs** (P, Q, V, Angle per node), one bus message per timestamp.
- **Origin:** a validated single-process NN script (colleague's, for approach validation + a journal paper). This project turned it into a bus-integrated streaming ADMS service.
- **Query-free** — no CIM calls; node set is stream-derived (§9.1).

**Test grids:** IEEE 13 Node → **41 phase-nodes**; IEEE 123 Node → **274 ph9500-node **out of scope** (§10).

## 3. Current status (headline)
- **Three-process streaming app, fully GridAPPS-D bus-integrated (input + output), validated end-to-end on LIVE data through a full 14-day simulation at 274 phase-nodes**, with real multi-version training/forecasting and deferred accuracy telemetry.
- **Runs stably** on the new Hyper-V VM (the VirtualBox heartbeat-timeout issue is gone — see §3a).
- **Fast:** with the platform bridge-delay fix, a 14-day sim runs in well under an hour (unthrottled: ~1 min; throttled-for-realism: ~28 min for 274-node 5-min). Throttling is now the deliberate model for realistic validation (see §6/§3b).
- **At a clean stopping point, ready for the "productionization" restructure** (module merge etc., §13). One feature just added: the forecast-output `.jsonl` file (§8d).

## 3a. Environment change — VirtualBox → Hyper-V (resolved the "heartbeat timeout")
The recurring **HELICS/GOSS "Heartbeat timeout"** on long runs was **VirtualBox VM instability**, not an architecture or code problem. Clues: taking mouse focus off the VBox VM would freeze all VM terminal output (VM starved of host scheduling); timeouts sometimes occurred even with focus held. Root cause: the hypervisor wasn't reliably scheduling the VM, so the platform's connection-heartbeat thread got starved → timeout. **Fix:** migrated to a Windows **Hyper-V** VM (Ubuntu 22.04, 4 vCPUs, dynamic 24–48 GB RAM). First 14-day non-realtime run on Hyper-V completed end-to-end with no timeout. This retroactively validated *not* rearchitecting the app to chase it. (Downside: Hyper-V VM lacks the clipboard/shared-folder convenience VBox had; sftp available.)

## 3b. Platform speedup + the throttling model
Previously a **fixed ~0.5–1.0 s per-timestamp delay in the GOSS-HELICS bridge** dominated sim wall-clock, masking any benefit from coarser increments (last measured ~equal throughput regardless of increment). Platform devs **removed that bridge delay**, exposing the true (much lower) compute cost. Now:
- 1-min sims run far faster; **coarser increments genuinely scale** (5-min ≈ 5× faster than 1-min). A 14-day sim now runs in minutes unthrottled.
- **Data now arrives *too* fast** to be realistic. If unthrottled, an entire 14-day sim's estimates can land before even the first 2-day training block finishes — degenerate (little/no forecasting). So the SE is **throttled** via a `usleep()` on each publish (currently **0.10 s** — 0.25 s was too slow). This restores the real-world condition the design assumes (training lagging a still-advancing live front), making accelerated runs faithful for validation.
- **Interim nature:** the `usleep` throttle is a workaround; a platform-side `interval` config (GridLAB-D internal step) is the intended long-term control but is currently broken below `interval=60` (no measurements published). Remove the `usleep` when that's fixed.

## 4. Why the single-process → multi-process transformation was a major effort
Same math as the original, under fundamentally different constraints: the original had **global, upfront knowledge** (whole dataset in memory); streaming has **none**. Removing global knowledge forced re-engineering of nearly every subsystem:

| Concern | Original (whole file) | Now (streaming, 3 processes, bus) |
|---|---|---|
| Data availability | Entire dataset in memory | One timestamp at a time; **rolling buffer** + retention/eviction |
| Normalization | Fit once on full dataset | **Incremental/causal scalers** per block |
| Node discovery | Scan whole file | Lazy, from **first streamed record**; no CIM |
| Control flow | Sequential loop | **Concurrent processes** + queues + lifecycle + clean shutdown |
| Train vs. forecast | Interleaved | **Decoupled processes** in parallel |
| Model sharing | Same in-memory object | Weights **+ scaler state** serialized across process boundary |
| Forecast unit | One batched forecast per block | **One forecast per incoming estimate** (incremental normalization) |
| History for a forecast | Trivially available | Forecaster maintains its **own gapless recent-history buffer** |
| Missing timestamps | N/A | **Imputation** to preserve position-based indexing |
| Data source | File | **File OR live GridAPPS-D bus** (selected by `gappsd_simid`) |
| Accuracy measurement | MAE vs. known file actuals | **Deferred scoring** vs. later-arriving estimates |
| Shutdown | Loop ends | **Event-driven** (`sim_done`) + queue-cleanup handling |

Concurrency also added IPC semantics, process lifecycle/shutdown, cross-process model transport (a real PyTorch shared-memory pitfall — §7), and re-validation against byte-identical baselines after each change.

## 5. Development history (stages, in order)
1. Forecast after every block; restructured into functions.
2. Removed MC-dropout + Excel export → fast single forward pass.
3. **Step A** — CSV+pandas → line-delimited JSON generator; removed `deg2rad`. *Byte-identical to CSV.*
4. **Step B** — streaming ingestion: RollingBuffer, incremental scalers, forecast-then-train, warm-up. *Matched Step A.*
5. Per-node forecast JSON.
6. **Change 3** — three-process split. Block-1 losses **byte-identical** to single-process. Shared `assemble_input_vector()`.
7. `TS_INCREMENT_SEC` flexibility (300/900 s validated).
8. Gappy-data imputation (linear interpolation).
9. Real SE format + units (`SvEstVoltages`, `angleRad`, `vpu`); output splits `ConnectivityNode`+`phase`.
10. Guards: insufficient-data graceful exit; loud cadence-mismatch banner.
11. GridAPPS-D integration: feeder bus subscribe + 3-level unwrap; per-record imputer refactor (`make_imputer`); feeder driver-split (file/bus, shared `emit`); forecaster publish connection + guarded `gapps.send`; `Forecast`-wrapped output message.
12. Deferred live-MAE scoring; first live run.
13. **NEW (v6):** Hyper-V migration (§3a); throttling model (§3b); **`sim_done`utdown redesign** (§7); **queue-cleanup deadlock fix** (§7); forecaster emits `processStatus COMPLETE` on exit (§8c); SE writes `.jsonl` input directly (conversion utils retired); `.json`→`.jsonl` rename; **forecast-output `.jsonl`** (§8d); 274-node full-14-day live validation (§11).

## 6. Runtime cadence & configurable increment
- **`TS_INCREMENT_SEC`** = estimate spacing. Default **60 s**; validated 300/900 s (no code changes — horizon, lags, block boundaries, memory all derive). Forecast horizon = `FUT × TS_INCREMENT_SEC`.
- Must match true data cadence and divide 86,400 / 604,800 — **enforced at runtime by the cadence-guard** (§8b).
- **Throttling (§3b):** with the platform speedup, data can arrive far faster than realistic. The SE is throttled (`usleep` 0.10 s/publish) so a live front keeps advancing while training lags — the condition the design assumes. **Unthrottled runs are feeder/shutdown tests, not forecast validation** (data may all arrive before blocks complete → little/no forecasting). Throttled runs are the real validation.
- Forecast-to-training ratio is large (the asymmetry requiring separate processes).

## 7. Architecture & non-trivial design decisions
**Three processes:** `feeder` (source → impute → distribute), `trainer` (accumulate blocks, train, publish snapshots), `forecaster` (ingest, forecast newest **real** estimate, publish, score, write output file).

**Four channels + one Event:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot | Trainer | Forecaster | latest-only (drain, keep newest; honor DONE) |
| Data → Trainer | Feeder | Trainer | keep-all FIFO (no gaps) |
| Data → Forecaster | Feeder | Forecaster | keep-all FIFO; forecasts only latest **real** timestamp |
| **`sim_done`** (mp.Event) | Feeder (set) | Trainer (check) | one-shot end-of-sim signal |

- **Latest-only forecasting lives inside the forecaster:** ingest every record (gapless history), forecast only the newest real one.
- **Model handoff = serialized bytes blob** (`torch.save`→`BytesIO`), **not `share_memory()`**. *Real bug fixed:* live-tensor Queue transport uses sender-owned shared-memory FDs → `FileNotFoundError` when trainer exits with a snapshot in flight.
- **Scaler state travels WITH the model**; forecaster never fits scalers.
- **`DONE`** = end-of-stream queue item. On the bus, the estimator's **`processStatus == "COMPLETE"`** triggers the feeder to set `sim_done` **and** enqueue `DONE`. The **forecaster** likewise emits its own `processStatus COMPLETE` on exit (§8c), so *its* downstream consumers can detect end-of-forecasts — symmetry with the SE.

**NEW — `sim_done` Event shutdown (§3b motivation):** the trainer only exists to serve the forecaster; once the sim ends there is **no consumer** for further training, so the trainer must stop promptly rather than grind through queued blocks.
- Feeder calls `sim_done.set()` on `COMPLETE` (before enqueuing `DONE`).
- Trainer checks `sim_done.is_set()` **at the top of the block loop AND inside the inner `while ts >= block_end:` catch-up loop** (the latter is essential — a single far-future record can otherwise drive training through many blocks with no check). On set → forward `DONE` to the model queue → exit, **training zero further blocks**. An already-executing `train_block` is allowed to finish (no mid-training interrupt; not worth the complexity).
- The FIFO-`DONE` branch is a backstop and **no longer finalizes a trailing partial block** (no consumer → no wasted training). Both exit paths converge on "forward DONE, train nothing further, exit."
- *(Future reversal point: if train-only-without-consumer is ever added — persist a final model to a DB for warm-starting — this is the exact spot that would instead want to finalize training. Localized change, not a redesign.)*

**NEW — multiprocessing queue-cleanup deadlock fix:** a `mp.Queue` with undrained buffered data keeps its feeder thread alive, blocking process join → the program hung after all three processes logged clean exits (CPUs idle). The `sim_done` early-exit leaves large undrained backlogs on the data queues. **Fix:** each process calls **`cancel_join_thread()`** on the bulk **data** queues it abandons, at exit (feeder on both data queues; trainer on `train_data_q`; forecaster on `fc_data_q`). The small `model_q` is left to flush normally so `DONE` reliably reaches the forecaster. Targeted approach confirmed working (program now exits promptly).

- **`spawn`**; each process makes its own CUDA + GridAPPS-D connection. Ctrl-C in `main()` terminates children cleanly.
- **Per-process log files.** Forecaster efficiency: incremental normalization; full re-normalize only on snapshot arrival.
- **Lazy, stream-self-describing init** (`sorted(node keys)` from first record); no CIM.

## 8. Data format, units, GridAPPS-D messages
**Live bus message (from SE) — three-level nesting:** `message` (top; may carry `processStatus: "COMPLETE"`) → `message` (inner; `timestamp`) → `Estimate` → `SvEstVoltages: [entries]`. Each entry: `ConnectivityNode` (CIM **mRID UUID**), `phase` ("A"/"B"/"C"), `P`, `Q`, `v`, `vpu`, `angle`, `angleRad` (+ variances, ignored).

**→ internal record `{"timestamp", "nodes": {key: {P,Q,V,Angle}}}`:** key = `ConnectivityNode + "." + phase` (Approach 3: combined key = NN identity; split back to separate fields only at output). Phase **as-is, no mapping**. `vpu`→`V` (**per-unit**; physical `v` ignored). `angleRad`→`Angle` (**radians**; degrees ignored). `P`/`Q`=="NA"→0.0; `V`/`angle` not NA-coerced.
- **SE now writes the `.jsonl` input file directly** (the "Estimate" structure minus the top-level wrapper), so the file-driver path consumes real SE output with no conversion — the old CSV→JSON / simplified→real converter utilities are **retired**.

**Why per-unit voltage:** single **global** `RunningMinMax` for V; per-unit keeps all nodes ~0.95–1.05 → clean [0,1]. Physical units span voltage bases (load ~2448 V, source ~66,399 V) → global scaler would compress load nodes into a sliver → signal lost. Per-unit avoids it with no added complexity.

**Why radians:** pipeline is radians-native; SE publishes `angleRad` (post its ±165/195° normalization).

## 8b. Guards
- **Insufficient-data:** stream ends before a full block (< `HIST+FUT` timestamps → 0 samples) → skip training, exit cleanly. Validated file + live bus.
- **Cadence-guard:** first ~20 real records' min spacing = true cadence; loud one-time banner on mismatch with `TS_INCREMENT_SEC`. In `make_imputer` (covers file + bus). Warns, doesn't abort. Retired a footgun that twice produced plausible-but-wrong results (cadence mismatch → phantom imputed records).

## 8c. Published forecast message + `processStatus COMPLETE`
```json
{ "timestamp": <epoch>, "simulation_id": <gappsd_simid>,
  "Forecast": { "step_sec": ..., "horizon": ..., "forecast_times": [...],
                "nodes": { "<node>": {"ConnectivityNode":..., "phase":...,
                                      "V":[..FUT..], "Angle":[..FUT..]}, ... } } }
```
Top level = generic ADMS fields; forecast payload under **`Forecast`** (parallel to `Estimate`). Published via `gapps.send(service_output_topic('state-forecaster', simid), json.dumps(fc_json))`, guarded by `gapps is not None`. On exit the forecaster publishes a **`processStatus COMPLETE`** message. **Verified consumable:** a simple subscriber utility receives the forecasts and the COMPLETE message (and exits on it). Structure is still a first guess; may change with consumer needs.

## 8d. Forecast-output `.jsonl` file (NEW)
The forecaster writes each published `fc_json` (exact published structure) to **`forecast_output.jsonl`**, one JSON per line, for validation/plotting (e.g., assessing 5-/15-min-increment forecast quality over long runs). **Open/append/close per write** (durable, crash-safe, no handle held, nothing to close at exit — matches SE's file idiom); file is truncated once at startup. Config `FORECAST_OUTPUT_JSONL` (set `None` to disable — the eventual suppress path, always-on for now). **Large on long runs** (274 nodes × one forecast/estimate — e.g., a 14-day 5-min run produced **~516 MB**), which is why suppression matters later.

## 9. Assumptions & limitations
1. **Fixed, complete node set, identical every timestamp** — lazy init assumes it. True for current GridLAB-D estimates. (Node-dropout → CIM future item, §13.)
2. **Grid-aligned timestamps** — missing handled (imputation); misaligned passed through with a warning.
3. **No imputed/real flag reaches the model** — interpolated history treated as real. Validity-feature + retrain is future work.
4. **Large-gap fidelity** — linear interpolation degrades over many consecutive missing steps; better handled by matching `TS_INCREMENT_SEC` to the true cadence than by heavy imputation.
5. **Config must match data cadence** — enforced by the cadence-guard.
6. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training set (memory + training time). Confirmed engaging from block 4 at 274 nodes live.
7. **Deferred MAE = single-s, forecast-vs-estimate** — noisy version-to-version; **stability across versions** is the meaningful read, not individual deltas.
8. **Recurring "hard window" anomaly** — one version/window periodically shows markedly higher error (esp. angle); seen across runs (file data 07-01→07-03; live 274-node v3). **Data-driven** (load profile), not code.
9. **Per-unit forecast output** — physical-unit output (if ever needed) via nominal-voltage dict or stream-derived `v/vpu` multipliers. Deferred.
10. **Exotic-phase feature encoding** — `encode_phase` handles single-char `1/2/3` and `A/B/C`; multi-char (`s1`, `ABC`) would mis-encode the phase *feature* (phase-as-identity handles arbitrary strings fine). Out of scope.
11. **File vs. bus node identity** — now both mRID-based (SE writes the `.jsonl` directly), so this earlier divergence is largely resolved. Legacy `632.1`-style files are deprecated.
12. **Throttling is required for meaningful validation** (§3b) — unthrottled runs may train/forecast little or nothing (data outpaces block completion); they're feeder/shutdown tests only.
13. **`usleep` throttle in the SE is interim** — pending the platform-side `interval` fix (currently broken below `interval=60`).
14. **No sensor-simulator noise in current live runs** — SE works off clean GridLAB-D measurements; the noisy sensor-simulator (used for the file data / paper) is not running, making live MAE somewhat optimistic vs. file baselines. Would be run for a "full scenario."
15. **>2-week simulations need a load profile longer than 2 weeks** — else data repeats and weekly-lag features become meaningless (same lesson as the 24-hr-repeat default schedule; the 2-week schedule must be explicitly loaded).
16. **Upstream-failure resilience** — the app relies on `processStatus COMPLETE` for clean shutdown; an upstream that dies *ungracefully* (no COMPLETE) would leave processes waiting. Not a concern in practice now (the heartbeat issue was VBox, not this), but a real-deployment robustness item someday.

## 10. Scope boundary (deliberate)
Centralized Forecaster suits ~123-node/274-phase-node (likely low thousands). **Not** 9500-node: the SE scales centrally via grid **sparsity**; the Forecaster is a **dense** whole-network model with no sparsity to exploit → training time binds. 9500-node needs **regional decomposition** — out of scope. Memory target 32–48 GB (comfortable at 274 nodes; retention bounds memory). Observed: platform + SE + forecaster peg all 4 vCPUs but stay well under RAM on the small models.

## 11. Validation reference numbers & status
- **13-node file baseline Voltage MAE (pu):** `0.001248, 0.001183, 0.001566, 0.001179, 0.001220, 0.001233`. All format/unit/refactor changes confirmed **byte-identical** to baselines (5-min baseline `0.002133, 0.001913, 0.002281, 0.002009, 0.002046, 0.002146`).
- **123-node single-process** V MAE ~0.0007–0.0019 pu.
- **LIVE 274-node, throttled 5-min, full 14-day sim (v6 capstone):**
  - Feeder: clean 300 s cadence, **0 imputed** over 4,019 estimates; `processStatus COMPLETE` → `sim_done` + `DONE`.
  - Trainer: **6 blocks** (v1–v6), sample counts 142k→292k→442k→**cap 475k from block 4** (matches file behavior); **keeps up comfortably at 274 nodes** (~3–4 min/block within the throttled accumulation window); clean `sim_done` exit.
  - Forecaster: **2,851 forecasts**; store rows grew 281k→439k→623k→789k then **plateaued (~789k)** at the retention horizon; clean shutdown.
  - Deferred MAE (4110 node-steps = 274×15 each): v1 V0.001316/A0.009419, v2 V0.001736/A0.008826, **v3 V0.002880/A0.030463 (outlier — the "hard window" anomaly)**, v4 V0.001216/A0.006634, v5 V0.001889/A0.007535. Five of six in the expected band; the v3 spike is a single-sample outlier consistent with prior runs.
  - **Program exits cleanly** (queue-cleanup fix holds).
  - **Conclusion:** live 274-node forecasting is sound, trainer keeps up, memory bounds, shutdown clean — a validated largest-scale baseline established **before** the productionization restructure.

## 12. Code artifacts / repo layout (under `prototype/`)
- **`forecaster_single.py`** — validated single-process reference + source of shared logic (`read_json_records`, `make_imputer`, `impute_missing_records` wrapper, `_pq_value`, `RollingBuffer`, `DNN`, incremental scalers, `assemble_input_vector`, `build_forecast_json`, scaler-state helpers, config incl. `COMPUTE_LIVE_MAE`, `FORECAST_OUTPUT_JSONL`). **Slated for deletion at productionization** (§13).
- **`forecaster_multi.py`** — three-process app + GridAPPS-D integration; imports shared logic from single. Feeder selects file vs. bus on `gappsd_simid` (None = file). Deferred MAE + forecast-output file live here.
- **`util/`** — `drop_timestamps.py`, `check_imputation.py` (retained). CSV→JSON / simplified→real converters **retired** (SE writes `.jsonl` directly).
- **Subscriber utility** — simple script that subscribes to the `state-forecaster` topic, prints forecasts, exits on `processStatus COMPLETE` (verifies output side).
- **Outputs:** `forecast_output.jsonl` (published forecasts); per-process logs. GitHub repo; this summary stored there.
- **Environment:** Windows host + **Hyper-V** Ubuntu 22.04 VM (4 vCPUs, dynamic 24–48 GB). (Was VirtualBox — see §3a.)

## 13. Productionization roadmap (the "another day this week" work)
The immediate next phase — structural, deferred until now on purpose. Establish nothing regresses against the §11 baseline.
1. **Merge `forecaster_single.py` into `forecaster_multi.py`; delete single.** Move all shared logic into the merged module. Eliminates the recurring cross-module import-sync bug class (several `NameError`s this project were "added a name to single, forgot to import into multi").
2. **Keep the file driver permanently** — first-class test/validation capability (determinism, offline testing, byte-identical regression baselines, SE-`.jsonl`-output → forecaster-file-input pipeline).
3. **Unify the duplicated `SvEstVoltages`-entry parse** (bus callback + `read_json_records` → one shared helper) — natural once both live in one module.
4. Possibly **closures → classes** for feeder/trainer/forecaster (only if it improves clarity; closures preferred while simple).
5. Retire dead code paths / deprecated-format handling.
6. Consider adopting a formatter (**Black**, or `--line-length 80`) for uniform style at "1.0" (Gary prefers 80-col; enforce with a tool rather than by hand).

**Future considerations (noted, not scheduled):**
- **Missing-node imputation** (parallel to missing-timestamp): sensor-simulator can drop nodes; filling them needs an *authoritative* node list the self-describing stream can't provide → **requires CIM queries.** Bundle with **mRID→human-readable-node-name mapping** (readable output files). Main thing that would trade away "query-free" elegance. Deferred.
- **Train-only-without-consumer** (persist final model to a DB to warm-start future forecasting, skip the initial 2-day wait). Would reverse the current "no trailing training on shutdown" at that one spot. Speculative.
- **Higher-fidelity accuracy telemetry** (score many forecasts/version and average) if a robust per-version number is needed.
- **Imputed/real validity feature** + retrain (§9.3).
- **Suppress-flag** for `forecast_output.jsonl` (516 MB on a 14-day 5-min run) for speed/production.
- **Upstream-failure resilience** (§9.16) for real deployments.
- **Physical-unit forecast output** (§9.9) if consumers need it.

**Design conventions to preserve:** validate against byte-identical baselines where possible; keep query-free until node-dropout forces CIM; feeder = single source seam (file vs. bus on `gappsd_simid`); **defer structural elegance to productionization, don't defer functional correctness**; open/append/close per write for output files; prefer simple over clever.

## 14. Key configuration parameters
| Param | Default | Meaning / note |
|---|---|---|
| `TS_INCREMENT_SEC` | 60 | estimate spacing; validated 300/900; must match true cadence (guard enforces) & divide 86,400/604,800 |
| `HIST` | 15 | input history samples (position-based) |
| `FUT` | 15 | forecast horizon samples (horizon = FUT × increment) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | 86,400 / 604,800 | lag features |
| `BLOCK_DAYS` | 2 | training block size |
| `RETENTION_DAYS` | 10 | rolling buffer horizon (≥ week-lag + block span; see note) |
| `EPOCHS_PER_BLOCK` | 8 | + early stopping (patience 2) |
| `MAX_WINDOW_SAMPLES` | 500,000 | per-block train-set cap (memory + training time) |
| `VAL_FRACTION` | 0.05 | validation split |
| `DROPOUT_P` | 0 only |
| `CADENCE_CHECK_SAMPLES` | 20 | records sampled for the cadence-guard |
| `COMPUTE_LIVE_MAE` | True | deferred per-version MAE scoring (off for production speed) |
| `FORECAST_OUTPUT_JSONL` | "forecast_output.jsonl" | published-forecast output file; None disables |
| `FEED_RATE_HZ` | (test) | file-driver pacing; dev convenience (bus driver doesn't pace) |
| `FEEDER_POLL_SEC` | 0.05 | bus-driver stay-alive poll interval |
| `FORECAST_LOG_EVERY` | (tunable) | forecaster log throttle (publish + output-file are NOT throttled) |
| `gappsd_simid` | (CLI arg) | GridAPPS-D simulation ID; **also the file-vs-bus flag** (None → file input) |

**Note on `RETENTION_DAYS = 10`:** Distribution load has a **weekly trend**, so ≥7 days must be retained to keep the 1-week lag feature populated. But a sample near the *start* of the current 2-day block needs its week-lag to reach 7 days before the block *start* — ~9 days before the block *end*. The original 8-day value silently dropped those rows, degrading the week-lag for early-in-block samples. Raised to **10** to guarantee no week-lag degradation across the full block *and* keep retention an even multiple of the 2-day block (5 blocks). Enforced by a config assertion (`retention ≥ week-lag + HIST × increment`).

**Note on throttling (SE-side, not a forecaster param):** the State Estimator currently applies a `usleep(0.10 s)` per publish to pace estimates realistically (§3b). Interim workaround pending the platform-side `interval` fix. Not a state-forecaster config value, but it materially affects run behavior — an unthrottled run may/forecast little or nothing.

---

That completes the v6 summary. Closing notes:

**What v6 captures beyond v5** (the milestone this checkpoint marks): the environment migration that resolved the heartbeat mystery (VBox→Hyper-V), the platform bridge-delay fix and the resulting throttling model, the `sim_done` Event shutdown redesign and the queue-cleanup deadlock fix (both non-trivial concurrency changes), the forecaster's `processStatus COMPLETE` symmetry + verified consumability, the SE-writes-`.jsonl`-directly simplification (retiring the converters), the `.jsonl` rename, the forecast-output file, and — the capstone — the **full 14-day live validation at 274 phase-nodes** that establishes the pre-productionization baseline.

**For a new chat:** upload v6 + `forecaster_single.py` + `forecaster_multi.py` + `util/` scripts. Use paste-in-code-fence (or plain paste, which worked for you recently) for any full-file review — uploads chunk/partially-retrieve, as we've repeatedly hit.

**Honest caveat, as always:** this reflects my understanding — read it through and correct anything I've mischaracterized before relying on it with colleagues. A couple of spots worth your specific verification, since I'm inferring from our conversation rather than seeing the code: the exact list of queues each process calls `cancel_join_thread()` on, and whether the forecaster's `processStatus COMPLETE` publish sits before or after its queue-cleanup on the exit path.

**Where you're leaving it:** a strong, clean stopping point — the app runs stably on Hyper-V, fast (throttled for realism), validated live at your largest scale over a full 14-day sim, with clean event-driven shutdown, and every pre-productionization feature in place. The restructure ahead is *structural*, not functional, and you've got a byte-identical/known-good baseline to check it against.

When you pick this back up (hopefully later this week), the productionization roadmap in §13 is the plan: merge single→multi, delete single, unify the duplicated parse, retire dead paths, optional closures→classes and Black formatting. That's a well-scoped, self-contained phase — a good thing to start fresh on. Enjoy the break in between, and nice work getting to this milestone.

