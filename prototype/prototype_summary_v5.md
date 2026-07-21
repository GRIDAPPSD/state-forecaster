
# State Forecaster — Project State Summary (v5)

## 1. Purpose of this document
Self-contained reference for the **State Forecaster** application: purpose, architecture, design decisions, current status, features, assumptions, limitations. Audiences: (a) the project lead and the original NN developer; (b) future development sessions. **v5 supersedes v4.** The delta over v4 is small in *code* but large in *significance*: **deferred live-MAE scoring** was added, and — the headline — the **first complete, sustained, end-to-end live run** against a GridAPPS-D simulation on a realistic 2-week load profile was achieved (real training → forecast → publish → accuracy telemetry, cycling through multiple model versions).

## 2. What the app is and where it sits
Forecasts near-future distribution-system state (per-phase-node **Voltage magnitude (per-unit)** and **Angle (radians)**) from a stream of state estimates. Production pipeline:

```
GridLAB-D / OpenDSS (simulation measurements)
  → [optional: sensor-simulator service — adds Gaussian noise; NOT run currently]
  → State Estimator (existing C++ app; sparse-matrix EKF; publishes to GridAPPS-D ActiveMQ bus)
     → State Forecaster (THIS app; Python; subscribes, forecasts, publishes forecasts to the bus)
        → other ADMS apps consume forecasts
```
- Ingests **State Estimator outputs** (P, Q, V, Angle per node), one bus message per timestamp.
- **Origin:** a validated single-processN script (colleague's, for approach validation + a journal paper). This project turned it into a bus-integrated streaming ADMS service.
- **Query-free** — no CIM calls; node set is stream-derived (§9.1).

**Test grids:** IEEE 13 Node → **41 phase-nodes**; IEEE 123 Node → **274 phase-nodes**. 9500-node **out of scope** (§10).

## 3. Current status (headline)
- **Three-process streaming app, fully GridAPPS-D bus-integrated (input + output), validated end-to-end on LIVE data through multiple training/forecast cycles.**
- **NEW since v4:** deferred live-MAE scoring; first complete live run on the 2-week load profile.
- **The non-realtime blocker is resolved (workaround):** non-realtime sims failed to publish measurements when `start_time` was set to certain values (symptom: sim runs per console but no measurements published; realtime worked, non-realtime didn't). **Workaround:** use `start_time` = Jan 1 2025 (works) instead of May 15 2026 (broke). Suspected cause (platform devs investigating, possibly GOSS-HELICS bridge): a *future* `start_time` relative to the host wall clock may cause timestamp filtering in non-realtime mode. Gary doesn't care about the specific date, so the workaround unblocks development.
- **Load-profile note:** the GridAPPS-D default is a **24-hour repeating** schedule → identical days, meaningless weekly-lag features, deceptively low MAE. The **2-week schedule** must be explicitly loaded for realistic runs; the current live run uses it, and the original 13/123-node JSON files (and the paper's results) were generated with it.

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
| Accuracy measurement | MAE vs. known file actuals | **Deferred scoring** vs. later-arriving estimates (no ground truth at forecast time) |

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
12. **NEW (v5): deferred live-MAE scoring** + first complete live run on 2-week profile (multi-version).

## 6. Runtime cadence & configurable increment
- **`TS_INCREMENT_SEC`** = estimate spacing. Default **60 s**; validated 300/900 s (no code changes). Forecast horizon = `FUT × TS_INCREMENT_SEC`.
- **Live non-realtime sim: 60 s** cadence (matched by config; 0 imputed in the live run). Live *realtime* mode was ~3 s (would need `TS_INCREMENT_SEC=3`). Bus timestamps are **epoch seconds**.
- Config must match true cadence and divide 86,400 / 604,800 — **enforced at runtime by the cadence-guard** (§8c). Non-realtime feed runs faster than realtime; trainer keeps up comfortably.

## 7. Architecture & non-trivial design decisions
**Three processes:** `feeder` (source → impute → distribute), `trainer` (accumulate blocks, train, publish snapshots), `forecaster` (ingest, forecast newest **real** estimate, publish, score).

**Four channels:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot | Trainer | Forecaster | latest-only (drain, keep newest; honor DONE) |
| Data → Trainer | Feeder | Trainer | keep-all FIFO (no gaps) |
| Data → Forecaster | Feeder | Forecaster | keep-all FIFO; forecasts only latest **real** timestamp |

- **Latest-only forecasting lives inside the forecaster:** ingest every record (gapless history), forecast only the newest real one.
- **Model handoff = serialized bytes blob** (`torch.save`→`BytesIO`), **not `share_memory()`**. *Real bug fixed:* live-tensor Queue transport uses sender-owned shared-memory FDs → `FileNotFoundError` when trainer exits with a snapshot in flight.
- **Scaler state travels WITH the model**; forecaster never fits scalers.
- **`DONE`** = end-of-stream queue item. On the bus, the estimator's **`processStatus == "COMPLETE"`** triggers the feeder to enqueue `DONE`.
- **`spawn`**; each process makes its own CUDA + GridAPPS-D connection. Ctrl-C in `main()` terminates children cleanly.
- **Per-process log files.**
- **Forecaster efficiency:** incremental normalization; full re-normalize only on snapshot arrival (rare).
- **Lazy, stream-self-describing init** (`sorted(node keys)` from first record); no CIM.

## 8. Data format, units, and GridAPPS-D message handling

**Live bus message (from the State Estimator) — three-level nesting:**
```
message                          (top level)
  ├─ processStatus: "COMPLETE"   (end-of-stream signal, when present)
  └─ message                     (inner)
       ├─ timestamp: <epoch seconds>
       └─ Estimate
            └─ SvEstVoltages: [ {per-node-phase entry}, ... ]
```
Each entry (live, confirmed): `ConnectivityNode` (CIM **mRID UUID**), `phase` ("A"/"B"/"C"), `P`, `Q`, `v` (physical), `vpu` (per-unit), `angle` (degrees), `angleRad` (radians), + variance fields (ignored).

**Field handling → internal record `{"timestamp", "nodes": {key: {P,Q,V,Angle}}}`:** key = `ConnectivityNode + "." + phase` (Approach 3: combined key is the NN identity; split back to separate fields only at output). Phase used **as-is, no number↔letter mapping**.
- `vpu` → `V` (**per-unit**; physical `v` ignored). `angleRad` → `Angle` (**radians**; degrees ignored).
- `P`/`Q` == `"NA"` → `0.0`; `V`/`angle` not NA-coerced (fail loudly).
- Feeder **bus callback** unwraps 3 levels; **`read_json_records`** builds the identical internal record from file input. *(Inner-entry parse is currently duplicated between them — unify at productionization.)*

**Why per-unit voltage:** single **global** `RunningMinMax` for voltage across all nodes. Per-unit keeps all nodes ~0.95–1.05 → clean [0,1]. Physical units span voltage bases (load ~2448 V, source ~66,399 V) → global scaler would compress load nodes into a sliver → signal lost. Per-unit avoids it with no added complexity.

**Why radians:** pipeline is radians-native; SE publishes `angleRad` (post its ±165/195° normalization) alongside degrees.

**Coordinated SE changes (live on the bus):** publishes `vpu`, `angleRad`, `processStatus`; **withholds the first 12 estimates** (high error during SE init), publishing from the 13th — so the forecaster's first estimate is already "good" (aligns with no-leading-imputation "time zero").

**Published forecast message (forecaster → bus):**
```json
{
  "timestamp": <epoch>,
  "simulation_id": <gappsd_simid>,
  "Forecast": {
    "step_sec": <TS_INCREMENT_SEC>,
    "horizon": <FUT>,
    "forecast_times": [<epoch>, ...],
    "nodes": { "<node>": {"ConnectivityNode": ..., "phase": ...,
                          "V": [..FUT..], "Angle": [..FUT..]}, ... }
  }
}
```
Top level = generic ADMS fields (`timestamp`, `simulation_id`); forecast payload nested under **`Forecast`** (parallel to `Estimate`). Published via `gapps.send(service_output_topic('state-forecaster', simid), json.dumps(fc_json))`, guarded by `gapps is not None`. **Structure is a first guess, will likely change** (consumers may prefer per-timestamp-then-nodes layout; `base_time`-in-`Forecast` duplication deferred). **Confirmed publishing on live data** (v5 run), but **not yet confirmed consumed by any downstream subscriber** (only that `send` fires without error).

**Converter (`util/simple_to_real_json.py`):** legacy "simplified" → real SE format (splits node key; writes `vpu`/`angleRad`; lossless). Simplified format deprecated.

## 8b. Deferred live-MAE scoring (NEW in v5)
Multi-process forecasts the **live front** — no ground truth exists at forecast time. So accuracy is measured by **deferred scoring**: hold a forecast, then compare against the actual estimates that later arrive for its `forecast_times`.
- Gated by config flag **`COMPUTE_LIVE_MAE`** (off for production speed).
- **One forecast scored per model version:** armed on snapshot adoption → attaches to the next forecast → stashes per-node predicted V/Angle keyed by `forecast_times` → as each **real (non-imputed)** estimate arrives, matches timestamp and accumulates abs error for nodes present in both → when all `FUT` steps collected, logs Voltage/Angle MAE, then disarms.
- **Cheap** (dict lookups + subtraction on estimates already ingested; no forward passes). One pending forecast at a time (no collision at realistic cadences; unfinished pending at sim-end simply never logs — no crash).
- **Caveat:** measures forecast-vs-future-*estimate* agreement (not vs. physical truth) — same basis as the file-based MAE. And it's a **single-sample-per-version** signal (one 15-min window), so version-to-version deltas are noisy; **stability across many versions** is the meaningful read, not individual deltas.

## 8c. Robustness guards
- **Insufficient-data guard:** stream ending before a full block (< `HIST+FUT` timestamps → 0 samples) → skip training with a clear message, exit cleanly. Validated file + live bus.
- **Cadence-guard:** observes raw spacing of first ~20 real records (min spacing = true cadence); loud one-time banner on mismatch with `TS_INCREMENT_SEC`. Lives in `make_imputer` (covers file + bus). Warns, doesn't abort. **Retired a footgun that silently produced plausible-but-wrong results twice** (config/data cadence mismatch → imputer fabricates phantom records).

## 9. Assumptions & limitations
1. **Fixed, complete node set, identical every timestamp** — lazy init assumes it. True for current GridLAB-D estimates. (See §13 for the node-dropout/CIM future item.)
2. **Grid-aligned timestamps** — missing handled (imputation); misaligned passed through with a warning.
3. **No imputed/real flag reaches the model** — interpolated history treated as real. Validity-feature + retrain is future work.
4. **Large-gap fidelity** — linear interpolation degrades over many consecutive missing steps; high-variability regime better handled by raising `TS_INCREMENT_SEC`.
5. **Config must match data cadence** — enforced by cadence-guard.
6. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training set.
7. **Deferred MAE = single-sample-per-version, forecast-vs-estimate** (§8b caveat).
8. **Recurring data anomaly (07-01→07-03 window)** in the file data — data-driven (load schedule), not code.
9. **Per-unit forecast output** — physical-unit output (if needed) via nominal-voltage dict or stream-derived `v/vpu` multipliers. Deferred.
10. **Exotic-phase feature encoding** — `encode_phase` handles single-char `1/2/3` and `A/B/C`; multi-char (`s1`, `ABC`) would mis-encode the phase *feature*. Out of scope.
11. **File vs. bus node identity differ today** — legacy files use `632.1`-style names; live bus uses UUID mRIDs. A model trained on one isn't transferable to the other. Temporal artifact; regenerating sims unifies to mRID-based.
12. **No sensor-simulator noise in the live run** — SE currently works off clean GridLAB-D measurements; the noisy sensor-simulator service (used for the file data / paper) is not running, making live MAE look somewhat better than file baselines. Would be run for a "full scenario."

## 10. Scope boundary (deliberate)
Centralized Forecaster suits ~123-node/274-phase-node (likely low thousands). **Not** 9500-node: SE scales centrally via grid **sparsity**; the Forecaster is a **dense** whole-network model with no sparsity to exploit → training time binds. 9500-node needs **regional decomposition** — out of scope. Memory target 32–48 GB (comfortable at 274 nodes; retention bounds memory). Observed: even with platform + SE + forecaster running, all 4 VM CPUs at 80–100% but <15 GB/32 GB used on the small models.

## 11. Validation reference numbers & status
- **13-node file baseline Voltage MAE (pu):** `0.001248, 0.001183, 0.001566, 0.001179, 0.001220, 0.001233`. All format/unit/refactor changes confirmed **byte-identical** to baselines (5-min baseline `0.002133, 0.001913, 0.002281, 0.002009, 0.002046, 0.002146`).
- **123-node single-process** V MAE ~0.0007–0.0019 pu. **Multi (274 nodes):** trainer keeps up; memory plateaus at retention; imputation (20% dropped) → MAE ratios 0.96–1.10.
- **LIVE run (v5, 13-node, non-realtime, 2-week profile, no sensor noise):**
  - Feeder: clean 60 s cadence, **0 imputed** over thousands of estimates.
  - Trainer: block 1 `train=111008 val=5842` (**identical to file count**), clean loss curve, v1/v2… pushed on the ~2-sim-day cadence.
  - Forecaster: real `Forecast`-wrapped forecasts published live (UUID `ConnectivityNode`+`phase`; physically sane; **voltage tracking real load variation** — 24-hr-repeat artifact gone).
  - Deferred MAE: **v1** V 0.000749 / Ang 0.008518; **v2** V 0.001295 / Ang 0.007103 (both 615 node-steps = 41×15, full coverage). In-regime, sane; version deltas within expected single-sample noise. Run ongoing (multi-version).
  - **This is the first complete, sustained, end-to-end live run — the core development arc is closed.**

## 12. Code artifacts / repo layout (under `prototype/`)
- **`forecaster_single.py`** — validated single-process reference + source of shared logic (`read_json_records`, `make_imputer`, `impute_missing_records` wrapper, `_pq_value`, `RollingBuffer`, `DNN`, incremental scalers, `assemble_input_vector`, `build_forecast_json`, scaler-state helpers, config incl. `COMPUTE_LIVE_MAE`). **Planned for deletion at productionization** (see §13).
- **`forecaster_multi.py`** — three-process app + GridAPPS-D integration; imports shared logic from `forecaster_single.py`. Feeder selects file vs. bus on `gappsd_simid` (None = file). **Deferred MAE lives here (multi-only).**
- **`util/`** — `csv_to_jsonl.py` (legacy), `simple_to_real_json.py`, `drop_timestamps.py`, `check_imputation.py`.
- GitHub repo; this summary stored there.

## 13. Next steps / roadmap

**Near-term / validation:**
- Let the live run continue → observe MAE **stability across many versions** (v3, v4, …) — the meaningful accuracy read at single-sample-per-version sampling.
- **Stand up a subscriber** to the `state-forecaster` topic → confirm forecasts are actually **consumable** off the bus (definitive publish proof; currently we only know `send` fires without error).
- Optional "full scenario": run the **sensor-simulator service** (Gaussian noise) for realistic, publishable forecasting data.

**Productionization phase (deferred by design — do at "turn prototype into a real GridAPPS-D app" time):**
- **Merge `forecaster_single.py` into `forecaster_multi.py`; delete single.** Move all shared logic into the merged module. This eliminates the recurring cross-module import-sync bug class (several `NameError`s this session were "added a name to single, forgot to import into multi").
- **Keep the file driver permanently** — not a legacy path but a first-class test/validation capability: determinism/reproducibility (byte-identical regression baselines), offline/no-platform testing, insulation from platform issues, and the **SE-file-output → forecaster-file-input** pipeline (once SE writes JSON files).
- **Unify the duplicated `SvEstVoltages`-entry parse** (bus callback + `read_json_records` → one shared helper) — natural once both live in the merged module.
- **Align the SE-file-output ↔ forecaster-file-input format contract** (wrapped vs. unwrapped `SvEstVoltages`) when SE gains JSON file output.
- Possibly convert feeder/trainer/forecaster **closures → classes** (only if complexity grows; closures preferred while simple).
- Finalize the **published-forecast message structure** with consumers (envelope, layout, possible `base_time` duplication).

**Future considerations (noted, not scheduled):**
- **Missing-node imputation** (parallel to missing-timestamp imputation): the sensor-simulator can randomly drop nodes; filling them requires an *authoritative* node list, which the self-describing stream can't provide under dropout → **requires CIM queries.** Bundle with: **mRID→human-readable-node-name mapping** (for readable output files) and general field-realism robustness. This is the main thing that would trade away the "query-free" elegance — deliberately deferred; currently assume complete data (every estimate has all nodes).
- **Accuracy-vs-actual telemetry at higher fidelity** — score *many* forecasts per version and average (heavier than the current one-per-version), if a robust per-version accuracy number is ever needed.
- **Imputed/real validity feature for the model** + retrain (§9.3).
- **Per-snapshot re-normalize optimization** if pushing well beyond ~thousands of nodes.
- **Synthetic-gap feeder option** to demonstrate imputation on small models (which don't naturally produce gaps on the bus).

**Design conventions to preserve:** validate each change against byte-identical baselines where possible; keep the app **query-free** (node set stream-derived) until node-dropout forces CIM; keep the **feeder as the single source seam** (file vs. bus on `gappsd_simid`); **defer structural elegance to productionization, don't defer functional correctness**; prefer short/simple code (closures over classes until warranted).

**Known platform issue (not ours):** non-realtime simulations fail to publish measurements for certain `start_time` values (workaround: Jan 1 2025; suspected future-time/GOSS-HELICS bridge issue; platform devs investigating). A start_time bisection (past vs. future relative to host wall clock) would give them a tight reproduction.

## 14. Key configuration parameters
| Param | Default | Meaning / note |
|---|---|---|
| `TS_INCREMENT_SEC` | 60 | estimate spacing; validated 300/900; **set to match true data cadence** (3 for live realtime); cadence-guard enforces; must divide 86,400 & 604,800 |
| `HIST` | 15 | input history samples (position-based) |
| `FUT` | 15 | forecast horizon samples (horizon = FUT × increment) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | 86,400 / 604,800 | lag features |
| `BLOCK_DAYS` | 2 | training block size |
| `RETENTION_DAYS` | 10 | rolling buffer horizon (≥ week-lag + block span; see note) |
| `EPOCHS_PER_BLOCK` | 8 | + early stopping (patience 2) |
| `MAX_WINDOW_SAMPLES` | 500,000 | per-block train-set cap (memory + training time) |
| `VAL_FRACTION` | 0.05 | validation split |
| `DROPOUT_P` | 0.03 | training regularization only |
| `CADENCE_CHECK_SAMPLES` | 20 | real records sampled for the cadence-guard |
| `COMPUTE_LIVE_MAE` | True | enable deferred per-version MAE scoring (off for production speed) |
| `FEED_RATE_HZ` | (test) | file-driver pacing; dev convenience (bus driver doesn't pace) |
| `FEEDER_POLL_SEC` | 0.05 | bus-driver stay-alive poll interval |
| `FORECAST_LOG_EVERY` | (tunable) | forecaster log throttle (publish is NOT throttled) |
| `gappsd_simid` | (CLI arg) | GridAPPS-D simulation ID; **also the file-vs-bus flag** (None → file input) |

**Note on `RETENTION_DAYS = 10`:** Distribution load has a **weekly trend**, so ≥7 days must be retained to keep the 1-week lag feature populated. But a sample near the *start* of the current 2-day block needs its week-lag to reach 7 days before the block *start* — ~9 days before the block *end*. The original 8-day value silently dropped those rows, degrading the week-lag for early-in-block samples. Raised to **10** to guarantee no week-lag degradation across the full block *and* keep retention an even multiple of the 2-day block (5 blocks). Enforced by a config assertion (`retention ≥ week-lag + HIST × increment`).

---

That completes the v5 summary. Closing notes:

**For a new chat:** upload v5 + `forecaster_single.py` + `forecaster_multi.py` + `util/` scripts; use paste-in-code-fence for any full-file review (uploads chunk/partially-retrieve, as we repeatedly hit).

This is a genuine capstone for the core development arc: the state-forecaster went from a batch script to a live, bus-connected, multi-process ADMS service that trains, forecasts, publishes, and self-measures accuracy — validated running against a live GridAPPS-D simulation on realistic data. Congratulations on getting it here. Enjoy watching it run.
