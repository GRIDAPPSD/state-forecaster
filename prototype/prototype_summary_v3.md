
# State Forecaster — Project State Summary (v3)

## 1. Purpose of this document
Self-contained reference for the **State Forecaster** application: purpose, architecture, design decisions, current status, features, assumptions, and limitations. Audiences: (a) the project lead and the original neural-network developer; (b) future development sessions, to resume with full context. This v3 supersedes v2 and adds: the switch to the real GridAPPS-D State Estimator publish format, radian/per-unit unit alignment (`angleRad`, `vpu`), the `processStatus` end-of-stream signal, and two robustness guards (insufficient-data, cadence-mismatch).

## 2. What the app is and where it sits
The **State Forecaster** forecasts near-future dist (per-phase-node **Voltage magnitude (per-unit)** and **Angle (radians)**) from a stream of state estimates. Production pipeline:

```
GridLAB-D / OpenDSS (simulation measurements)
  → State Estimator (existing C++ app; sparse-matrix EKF; publishes state estimates
    to the GridAPPS-D ActiveMQ message bus)
     → State Forecaster (THIS app; Python; subscribes to estimates, forecasts,
       publishes forecasts to the bus)
        → other ADMS apps consume forecasts
```
- Ingested values are **State Estimator outputs** (P, Q, V, Angle per node), one bus message per timestamp.
- **Origin:** started from a validated single-process NN script (colleague's, for approach validation + a journal paper). This project turns it into a **bus-integrated, streaming ADMS service**.
- Uses only **state estimates** — not raw simulation measurements. Makes **no CIM/query calls** (deliberate; §9.1).

**Test grids:** IEEE 13 Node → **41 phase-nodes**; IEEE 123 Node → **274 phase-nodes**. 9500-node model is **out of scope** (§10).

## 3. Current status (headline)
- **Fully functional three-process streaming application**, validated end-to-end on both feeders, now aligned with the **real GridAPPS-D State Estimator publish format and units**.
- **Complete:** per-block forecasting; streaming ingestion (rolling buffer + incremental scalers); train/forecast process split with model+scaler handoff; per-node forecast JSON output; configurable timestamp increment; missing-timestamp imputation; real-format + unit alignment; insufficient-data and cadence-mismatch guards.
- **Next (planned, not started):** GridAPPS-D bus integration — first step is a single boolean flag selecting file input vs. live bus messages in the feeder, supporting both for now (§13).

## 4. Why the single-process → multi-process transformation was a major effort
The original script and the current app solve the *same forecasting math* under *fundamentally different systems constraints*. The original had **global, upfront knowledge** (whole dataset in memory); the streaming production shape has **none**. Removing global knowledge forced re-engineering of nearly every subsystem:

| Concern | Original (whole file) | Now (streaming, 3 processes) |
|---|---|---|
| Data availability | Entire dataset in memory | One timestamp at a time; **rolling buffer** + retention/eviction |
| Normalization | Scalers fit once on full dataset | **Incremental/causal scalers** updated per block |
| Node discovery | Scan whole file | Lazy, from **first streamed record**; no CIM |
| Control flow | Sequential loop | **Concurrent processes** + queues + lifecycle + clean shutdown |
| Train vs. forecast | Interleaved | **Decoupled processes** in parallel |
| Model sharing | Same in-memory object | Weights **+ scaler state** serialized across process boundary |
| Forecast unit | One batched forecast per block | **One forecast per incoming estimate** (incremental normalization) |
| History for a forecast | Trivially available | Forecaster maintains its **own gapless recent-history buffer** |
| Missing timestamps | N/A | **Imputation** to preserve position-based indexing |
| Data format/units | Simplified/whatever | **Real SE bus format**, radians + per-unit |

The concurrency itself added work absent from the original: IPC semantics, process lifecycle/shutdown, cross-process model transport (hit a real PyTorch shared-memory pitfall — §7), and re-validation against trusted baselines after each change. That verification discipline is a large part of why the result is trustworthy.

## 5. Development history (stages, in order)
1. **Change 1** — forecast after every block (was once at end); restructured into functions.
2. **Removed MC-dropout + Excel export** — forecast = fast single forward pass.
3. **Step A** — CSV+pandas → line-delimited JSON via a `read_json_records()` generator (the "streaming seam"); removed `deg2rad`. *Byte-identical to CSV baseline.*
4. **Step B** — true streaming ingestion: **RollingBuffer**, **incremental scalers**, forecast-then-train per block, emergent warm-up. *Matched Step A accuracy on both feeders.*
5. **Per-node forecast JSON** — `build_forecast_json()`, the eventual bus-publish unit.
6. **Change 3** — three-process split (feeder/trainer/forecaster). Block-1 trainer losses **byte-identical** to single-process → zero drift. Shared `assemble_input_vector()` extracted (re-validated byte-identical).
7. **`TS_INCREMENT_SEC` flexibility** — 300/900 s validated, no code changes (§6).
8. **Gappy-data imputation** — linear interpolation in the feeder (§8).
9. **Real-format + unit alignment** (this week): switched input parsing to the GridAPPS-D SE publish format (`SvEstVoltages`), `angleRad` (radians), `vpu` (per-unit voltage); output splits into `ConnectivityNode` + `phase` (§8b).
10. **Robustness guards** (this week): graceful exit on insufficient data; loud cadence-mismatch banner (§8c).

## 6. Runtime cadence & the configurable timestamp increment
- **`TS_INCREMENT_SEC`** = spacing between estimates. Default **60 s**; validated at **300** and **900** s with **no code changes** (forecast horizon, lag lookups, block boundaries, memory all derive). Forecast horizon = `FUT × TS_INCREMENT_SEC` (FUT=15 → 15/75/225 min).
- **Requirement:** config must match the true data cadence, and the increment must divide `DAY_LAG_SEC` (86,400) and `WEEK_LAG_SEC` (604,800). The **cadence-guard (§8c)** now enforces the cadence match at runtime.
- **Validated trade-off (13-node):** Voltage MAE ~0.00125 (60s) → ~0.0021 (300s) → ~0.0042 (900s) — monotonic, graceful; fewer samples/longer horizon in exchange for much longer simulations per wall-clock.
- **Real estimate rate ≈ 1/sec.** Accelerated feed rates (4–500/sec) are **dev conveniences**, not requirements. Forecast-to-training-update ratio ≈ 2,880:1 (the asymmetry requiring separate processes).

## 7. Architecture & non-trivial design decisions
**Three processes:** `feeder` (reads/paces records, imputes gaps, distributes), `trainer` (accumulates blocks, trains, publishes model snapshots), `forecaster` (ingests, forecasts each newest **real** estimate).

**Four channels:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot | Trainer | Forecaster | latest-only (drain, keep newest; honor DONE) |
| Data → Trainer | Feeder | Trainer | keep-all FIFO (no gaps) |
| Data → Forecaster | Feeder | Forecaster | keep-all FIFO; forecasts only the **latest real** timestamp |

- **Latest-only forecasting lives inside the forecaster**, not the queue: ingest every record (gapless history), fire a forecast only on the newest real one — discard stale *forecast opportunities*, never data.
- **Model handoff = serialized bytes blob** (`torch.save`→`BytesIO`), **not `share_memory()`** (which risks half-updated reads + would need training-loop locking). *Real bug hit & fixed:* live-tensor Queue transport uses sender-owned shared-memory FDs → `FileNotFoundError` when the trainer exited with a snapshot in flight. Bytes transport fixed it.
- **Scaler state travels WITH the model** (`extract_scaler_state`/`apply_scaler_state`); forecaster never fits scalers.
- **`DONE`** = distinguished queue item for clean end-of-stream. In live operation the estimator's **`processStatus`** message triggers the feeder to enqueue `DONE` (§13).
- **`spawn` start method** (CUDA + multiprocessing); each process inits CUDA independently. Ctrl-C in `main()` terminates children cleanly.
- **Per-process log files** (`feeder.log`, `trainer.log`, `forecaster.log`).
- **Forecaster efficiency (scales to ~thousands of nodes):** incremental normalization — normalize one incoming row = O(nodes); full re-normalize of the whole buffer happens **only on snapshot arrival** (~per block, rare; ~tens of seconds at 274 nodes — flagged for the thousands-node future).
- **Lazy, stream-self-describing init:** node set discovered from the **first record** (`sorted(node keys)` → identical mapping across processes). **No CIM/queries** — minimizes ADMS coupling, keeps it offline-testable and field-portable.

## 8. Data format & unit alignment (GridAPPS-D State Estimator publish format)
The app now ingests the **real** State Estimator bus format instead of the earlier "simplified" format (which was a development convenience, now deprecated).

**Input record (one JSON object per line / per bus message):**
```json
{"SvEstVoltages": [
    {"ConnectivityNode": "632", "phase": "1",
     "P": .., "Q": .., "v": <physical>, "vpu": <per-unit>,
     "angle": <degrees>, "angleRad": <radians>, ...},
    ...],
 "timeStamp": 1700000000}
```

**Field handling (in `read_json_records`, the sole format seam):**
- `timeStamp` → internal `timestamp`.
- Internal node key = `ConnectivityNode + "." + phase` (e.g. `"632"+"1"` → `"632.1"`). This single combined key is the NN's node identity (**Approach 3**: combined internally, split back to separate fields only at output). Phase used **as-is, no number↔letter mapping** (guards against errors with phases like `s1`, `ABC`; dots are only separators).
- `vpu` → internal `V` (**per-unit**; physical `v` ignored — see rationale below).
- `angleRad` → internal `Angle` (**radians**, post-normalization; degrees `angle` ignored).
- `P`/`Q` == `"NA"` → `0.0` (SOURCEBUS etc.). `V`/`angle` are NOT NA-coerced (missing voltage/angle should fail loudly, not silently zero).
- variance fields ignored.

**Output (`build_forecast_json`)** splits the internal key back into separate `ConnectivityNode` + `phase` fields (GridAPPS-D convention), phase as-is. Forecasts are **per-unit V + radian Angle** (self-consistent, dependency-free).

**Why per-unit voltage (not physical):** the code uses a **single global** `RunningMinMax` for voltage across all nodes. Per-unit keeps every node in ~0.95–1.05, so one global scaler maps them cleanly to [0,1]. Physical units span multiple voltage bases (e.g. load ~2448 V, source bus ~66,399 V), so a global scaler would be anchored by the highest base and compress load nodes into a tiny [0,1] sliver → forecast signal lost in numerical noise. Per-unit input avoids this with zero added complexity. (Per-node scalers would also fix it but add real complexity/state — deliberately not done.) The State Estimator was updated to publish `vpu` (and `angleRad`) alongside its physical/degree fields, coordinated in that repo.

**Coordinated State Estimator changes (in its repo, ready for integration):** publishes `angleRad`, `vpu`, and a **`processStatus`** message signaling end-of-estimates (so the forecaster needn't subscribe to simulation-log messages to detect completion).

**Companion converple_to_real_json.py` converts the old simplified files to this real format (splits node key → phase-less `ConnectivityNode` + `phase`; writes `vpu`/`angleRad`; no unit conversion — lossless). Used to regenerate all test files (13/123-node, 1/5/15-min).

*All format/unit changes were validated **byte-identical** to prior baselines (they're value-preserving renames/restructures).*

## 8b. Robustness guards (added this week)
- **Insufficient-data guard:** if a stream ends before any full training block (fewer than `HIST+FUT` = 30 timestamps → zero training samples), the app skips training with a clear message and exits cleanly instead of crashing in the `DataLoader`. Validated on 1-line and 50-line inputs, single- and multi-process. Maps to a real scenario: a simulation stopped early / `processStatus` arriving before a block accumulates.
- **Cadence-guard:** observes raw spacing of the first ~20 real records (min spacing = true cadence, since gaps only increase spacing) and, on mismatch with `TS_INCREMENT_SEC`, prints a **loud one-time banner** naming configured vs. detected cadence, the consequence, and the fix. Lives in `impute_missing_records` (centralized, stream-native, no file dependency). Warns (does not abort) to avoid multi-process shutdown complexity. Suppresses the per-gap "not a multiple" spam once a mismatch is known. Validated across all three branches: too-small, too-large, and matched (silent + byte-identical). **This retired a footgun that silently produced plausible-but-wrong results twice during testing** (a config/data cadence mismatch causes the imputer to fabricate phantom records).

## 9. Assumptions & limitations
1. **Fixed, complete node set, identical every timestamp.** Lazy init assumes every timestamp carries all nodes. True for GridLAB-D-derived estimates; may not hold for real field data. Escape hatch: a one-time CIM query — deliberately not built.
2. **Grid-aligned timestamps.** Missing timestamps handled (imputation); misaligned ones not expected (passed through with a warning).
3. **No imputed/real flag reaches the model.** Interpolated history is treated as real. Acceptable now; a validity-feature + retrain is future work.
4. **Large-gap fidelity.** Linear interpolation degrades over many consecutive missing steps; the high-variability regime is better handled by raising `TS_INCREMENT_SEC` than by heavy imputation.
5. **Config must match data cadence** — now enforced at runtime by the cadence-guard (§8b).
6. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training set (memory + training time). Sufficient at 123-node.
7. **Multi-process forecast accuracy not independently measured** (it forecasts the live front, no actuals yet). Trustworthy by construction. Optional future "compare-to-actual-when-it-arrives" telemetry.
8. **Recurring data anomaly (07-01→07-03 window):** elevated error across all runs; confirmed **data-driven** (shared load-profile schedule), not code. Note for evaluation/paper.
9. **Per-unit forecast output.** Forecasts are published per-unit. If physical-unit output is ever required, options are: a nominal-voltage multiplier dict (reintroduces a GridAPPS-D dependency), or deriving multipliers from the stream itself (`v/vpu` per node — dependency-free but unusual). Deferred.
10. **Exotic-phase feature encoding.** `encode_phase` handles single-char `1/2/3` and `A/B/C` (both map to the same one-hots). Multi-char phases (`s1`, `ABC`) would mis-encode the phase *feature* — out of scope; needs NN-level consideration if ever required. (Phase as *identity* handles arbitrary strings fine.)

## 10. Scope boundary (deliberate)
Centralized Forecaster is appropriate to ~123-node/274-phase-node (likely low thousands). **Not** designed for 9500-node: the SE scales centrally via grid **sparsity**; the Forecaster is a **dense** whole-network model with no sparsity to exploit, so training time binds. 9500-node needs **regional decomposition** (partition, parallel sub-forecasters, reconcile) — different architecture, out of scope. Memory target 32–48 GB (comfortable at 274 nodes; retention bounds memory regardless of run length).

## 11. Validation reference numbers (13-node, single-process)
Baseline aligned-forecast Voltage MAE (pu): `0.001248, 0.001183, 0.001566, 0.001179, 0.001220, 0.001233`. All format/unit changes this week (real format, `angleRad`, `vpu`) confirmed **byte-identical** to their respective baselines (e.g. 5-min baseline `0.002133, 0.001913, 0.002281, 0.002009, 0.002046, 0.002146`). 123-node single-process Voltage MAE ~0.0007–0.0019 pu. Multi-process (274 nodes): trainer keeps up with large margin at real rates; forecaster memory plateaus at retention horizon (~3.95M rows); imputation validated (20% dropped → forecast MAE ratios 0.96–1.10 vs. baseline).

## 12. Code artifacts / repo layout
- **`forecaster_single.py`** — validated single-process version; correctness reference + source of shared logic (`read_json_records`, `impute_missing_records`, `RollingBuffer`, `DNN`, incremental scalers, `assemble_input_vector`, `build_forecast_json`, scaler-state helpers, config).
- **`forecaster_multi.py`** — three-process app; imports shared logic from `forecaster_single.py`.
- **`util/csv_to_jsonl.py`** — CSV → simplified JSON (legacy; the simplified format is now deprecated).
- **`util/simple_to_real_json.py`** — simplified → real SE format (`SvEstVoltages`, `vpu`/`angleRad`, split node/phase).
- **`util/drop_timestamps.py`** — creates gappy test files (random grid-aligned drops).
- **`util/check_imputation.py`** — measures interpolation accuracy vs. true dropped values, NN-independent.
- **Input data:** real-format files for 13/123-node at 1/5/15-min increments (line-delimited JSON).
- Repo on GitHub; this summary stored there (renders as Markdown).

## 13. Next: GridAPPS-D integration (planned)
**First step (next):** add a single boolean config flag selecting the feeder's data source — **file input vs. live GridAPPS-D bus messages** — supporting both for now. Expected to be a modest amount of code: the feeder's record source becomes either `read_json_records(file)` (existing) or a bus-subscription callback that enqueues arriving estimates. Everything downstream (imputer, cadence-guard, trainer, forecaster) is unchanged — the feeder is the integration seam, exactly as designed.

**Integration mapping (groundwork already done):**
- Feeder subscribes to the State Estimator output topic (instead of reading a file). Records arrive already in the `SvEstVoltages` format the reader now parses.
- **`processStatus`** message → triggers the feeder to enqueue `DONE` (clean end-of-stream; no simulation-log subscription needed).
- Forecaster publishes forecasts to a bus topic (per-unit V, radian Angle, with `ConnectivityNode`+`phase` fields).
- The app remains **query-free** (no CIM) — node set still discovered from the first arriving estimate.

**Candidate later work:** physical-unit forecast output (§9.9); imputed/real validity feature + retrain (§9.3); accuracy-vs-actual telemetry (§9.7); per-snapshot re-normalize optimization if pushing well beyond 274 nodes; startup **synthetic-gap** option in the feeder to demonstrate imputation on small models (which don't naturally produce gaps).

## 14. Key configuration parameters
| Param | Default | Meaning / note |
|---|---|---|
| `TS_INCREMENT_SEC` | 60 | estimate spacing; validated 300/900; must match data cadence (cadence-guard enforces) and divide 86,400 & 604,800 |
| `HIST` | 15 | input history samples (position-based) |
| `FUT` | 15 | forecast horizon samples (horizon = FUT × increment) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | 86,400 / 604,800 | lag features |
| `BLOCK_DAYS` | 2 | training block size |
| `RETENTION_DAYS` | 10 | rolling buffer horizon (≥ week-lag + block span; see note below) |
| `EPOCHS_PER_BLOCK` | 8 | + early stopping (patience 2) |
| `MAX_WINDOW_SAMPLES` | 500,000 | per-block train-set cap (memory + training time) |
| `VAL_FRACTION` | 0.05 | validation split |
| `DROPOUT_P` | 0.03 | training regularization only |
| `CADENCE_CHECK_SAMPLES` | 20 | real records sampled for the startup cadence-guard (§8b) |
| `FEED_RATE_HZ` | (test) | feeder pacing; convenience only**, not a requirement |
| `FORECAST_LOG_EVERY` | (tunable) | forecaster log throttle (bounds log volume) |

**Note on `RETENTION_DAYS = 10`:** Distribution load has a **weekly trend**, so ≥7 days must be retained to keep the 1-week lag feature populated. But a sample near the *start* of the current 2-day block needs its week-lag to reach 7 days before the block *start* — i.e. ~9 days before the block *end*. The original 8-day value silently dropped those rows, degrading the week-lag for early-in-block samples. Raised to **10** to guarantee no week-lag degradation across the full block *and* keep retention an even multiple of the 2-day block (5 blocks). Enforced by a config assertion (`retention ≥ week-lag + HIST × increment`).

*(A new config flag — the file-vs-bus data-source boolean — will be added next week as the first GridAPPS-D integration step; §13.)*

---

That completes the v3 summary. A few closing notes for the weekend:

**Coverage check** — v3 is fully self-contained and captures everything through this week: the real-format switch, `angleRad`/`vpu` unit alignment (with the per-unit-scaling rationale), the `processStatus` integration hook, both new guards (insufficient-data, cadence-mismatch), the deprecation of the simplified format, and the next-step integration plan. This plus the source files reconstitutes the project with no dependence on v1/v2 or this conversation.

**For a new chat next week** (if this one ends or you start fresh): upload the v3 summary **plus** `forecaster_single.py`, `forecaster_multi.py`, and optionally the `util/` scripts. Use the **file-upload** path (not paste) to avoid the mangling. A fresh session starts with zero memory of our discussion, so those uploads are its entire context — which is exactly why the summary and clean source matter.

**One honest caveat** — as always, this summary reflects my understanding of our work; give it a read-through before relying on it with colleagues, and correct anything I've mischaracterized. You know the domain and the actual code state better than the summary does. Treat it as a strong draft you own.

**Where you're leaving things:** the prototype is aligned with the real data format and units, edge cases are covered, guards are in, and the integration seam (the feeder) is clearly defined with the file-vs-bus flag as the clean first step. That's a tidy place to pause before the integration push.

