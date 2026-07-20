
# State Forecaster — Project State Summary (v4)

## 1. Purpose of
Self-contained reference for the **State Forecaster** application: purpose, architecture, design decisions, current status, features, assumptions, and limitations. Audiences: (a) the project lead and the original neural-network developer; (b) future development sessions, to resume with full context. **v4 supersedes v3** and adds the entire **GridAPPS-D bus integration** (live estimate ingestion + forecast publishing), the per-record imputer refactor, the feeder driver-split, and the published-forecast message shape.

## 2. What the app is and where it sits
The **State Forecaster** forecasts near-future distribution-system state (per-phase-node **Voltage magnitude (per-unit)** and **Angle (radians)**) from a stream of state estimates. Production pipeline:

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
- Uses only **state estimates** (not raw measurements). Makes **no CIM/query calls** (deliberate loose coupling; §9.1).

**Test grids:** IEEE 13 Node → **41 phase-nodes**; IEEE 123 Node → **274 phase-nodes**. 9500-node model **out of scope** (§10).

## 3. Current status (headline)
--process streaming application, now integrated with the GridAPPS-D bus on both input and output**, validated end-to-end against **live** state-estimator data (for everything that can be validated without a full training block — see the blocker below).
- **Complete:** per-block forecasting; streaming ingestion (rolling buffer + incremental scalers); train/forecast process split with model+scaler handoff; per-node forecast JSON; configurable timestamp increment; missing-timestamp imputation; real SE bus format + units (`vpu`, `angleRad`); insufficient-data and cadence guards; **GridAPPS-D subscribe (feeder) and publish (forecaster) seams**; `Forecast`-wrapped output message.
- **BLOCKER (platform, not code):** GridAPPS-D **non-realtime simulations are currently broken** (broke while adding >1-min timestamp increments). Only realtime sims run, producing estimates every ~3 s; a 120 s test yields ~27 estimates — far too few for even one 2-day training block. So **actual live training/forecasting (and the first real `gapps.send` publish) cannot yet be exercised.** Gary is working with GridAPPS-D platform devs to restore non-realtime (or at least 1-min increments). All code is ready and waiting for that fix.

## 4. Why the single-process → multi-process transformation was a major effort
Same math as the original script, but under fundamentally different constraints: the original had **global, upfront knowledge** (whole dataset in memory); the streaming production shape has **none**. Removing global knowledge forced re-engineering of nearly every subsystem:

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
| Data source | File | **File OR live GridAPPS-D bus** (selectable) |

Concurrency added IPC semantics, process lifecycle/shutdown, cross-process model transport (hit a real PyTorch shared-memory pitfall — §7), and re-validation against trusted baselines after each change. That verification discipline is why the result is trustworthy.

## 5. Development history (stages, in order)
1. **Change 1** — forecast after every block; restructured into functions.
2. **Removed MC-dropout + Excel export** — forecast = fast single forward pass.
3. **Step A** — CSV+pandas → line-delimited JSON via `read_json_records()` generator; removed `deg2rad`. *Byte-identical to C
4. **Step B** — streaming ingestion: **RollingBuffer**, **incremental scalers**, forecast-then-train per block, emergent warm-up. *Matched Step A on both feeders.*
5. **Per-node forecast JSON** — `build_forecast_json()`.
6. **Change 3** — three-process split. Block-1 trainer losses **byte-identical** to single-process → zero drift. Shared `assemble_input_vector()` extracted.
7. **`TS_INCREMENT_SEC` flexibility** — 300/900 s validated, no code changes.
8. **Gappy-data imputation** — linear interpolation.
9. **Real SE format + unit alignment** — `SvEstVoltages`, `angleRad`, `vpu`; output splits `ConnectivityNode`+`phase`.
10. **Robustness guards** — insufficient-data graceful exit; loud cadence-mismatch banner.
11. **GridAPPS-D integration** (this session): feeder bus subscribe + 3-level unwrap; per-record imputer refactor (`make_imputer`); feeder driver-split (file vs. bus, shared `emit`); forecaster publish connection + guarded `gapps.send`; `Forecast`-wrapped output message. Validated end-to-end on live data (minus the training-block-dependent parts).

## 6. Runtime cadence & configurable increment
- **`TS_INCREMENT_SEC`** = estimate spacing. Default **60 s**; validated at **300/900 s** (no code changes — horizon, lags, block boundaries, memory all derive). Forecast horizon = `FUT × TS_INCREMENT_SEC`.
- **Live realtime cadence ≈ 3 s** (set `TS_INCREMENT_SEC=3` for live realtime testing so the cadence-guard stays quiet). Real bus timestamps observed as **epoch seconds** (10-digit), no ms conversion needed.
- **Requirement:** config must match true data cadence and divide `DAY_LAG_SEC` (86,400) / `WEEK_LAG_SEC` (604,800). Enforced at runtime by the **cadence-guard** (§8c).
- Real estimate rate ≈ 1/sec (realtime sims ~3 s). Accelerated feed rates are **dev conveniences**. Forecast-to-training ratio ≈ thousands:1 (the asymmetry requiring separate processes).

## 7. Architecture & non-trivial design decisions
**Three processes:** `feeder` (source → impute → distribute), `trainer` (accumulate blocks, train, publish snapshots), `forecaster` (ingest, forecast newest **real** estimate, publish).

**Four channels:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot | Trainer | Forecaster | latest-only (drain, keep newest; honor DONE) |
| Data → Trainer | Feeder | Trainer | keep-all FIFO (no gaps) |
| Data → Forecaster | Feeder | Forecaster | keep-all FIFO; forecasts only latest **real** timestamp |

- **Latest-only forecasting lives inside the forecaster**, not the queue: ingest every record (gapless history), fire a forecast only on the newest real one.
- **Model handoff = serialized bytes blob** (`torch.save`→`BytesIO`), **not `share_memory()`**. *Real bug fixed:* live-tensor Queue transport uses sender-owned shared-memory FDs → `FileNotFoundError` when trainer exits with a snapshot in flight. Bytes transport fixed it.
- **Scaler state travels WITH the model** (`extract_scaler_state`/`apply_scaler_state`); forecaster never fits scalers.
- **`DONE`** = distinguished queue item for clean end-of-stream. On the bus, the estimator's **`processStatus == "COMPLETE"`** message triggers the feeder to enqueue `DONE` (no simulation-log subscription needed).
- **`spawn` start method**; each process inits CUDA (and its own GridAPPS-D connection) independently. Ctrl-C in `main()` terminates children cleanly.
- **Per-process log files** (`feeder.log`, `trainer.log`, `forecaster.log`).
- **Forecaster efficiency (scales to ~thousands of nodes):** incremental normalization — normalize one incoming row = O(nodes); full re-normalize of the whole buffer only on snapshot arrival (~per block, rare; ~tens of seconds at 274 nodes).
- **Lazy, stream-self-describing init:** node set from the **first record** (`sorted(node keys)` → identical mapping across processes). **No CIM/queries.**

## 8. Data format, units, and GridAPPS-D message handling

**Live bus message (from the State Estimator) — three-level nesting:**
```
message                                    (top level)
  ├─ processStatus: "COMPLETE"             (end-of-stream signal, when present)
  └─ message                               (inner)
       ├─ timestamp: <epoch seconds>
       └─ Estimate
            └─ SvEstVoltages: [ {per-node-phase entry}, ... ]
```
Each `SvEstVoltages` entry (live, confirmed on the wire):
`ConnectivityNode` (CIM **mRID UUID**), `phase` ("A"/"B"/"C"), `P`, `Q`, `v` (physical), `vpu` (per-unit), `angle` (degrees), `angleRad` (radians), plus variance fields (ignored).

**Field handling → internal record `{"timestamp", "nodes": {key: {P,Q,V,Angle}}}`:**
- Internal node key = `ConnectivityNode + "." + phase` (Approach 3: combined key is the NN's identity; split back to separate fields only at output). Phase used **as-is, no number↔letter mapping**.
- `vpu` → `V` (**per-unit**; physical `v` ignored). `angleRad` → `Angle` (**radians**; degrees `angle` ignored).
- `P`/`Q` == `"NA"` → `0.0` (SOURCEBUS etc.); `V`/`angle` not NA-coerced (fail loudly).
- The feeder's **bus callback** unwraps the 3 levels and builds this internal record; **`read_json_records`** builds the identical internal record from file input. *(This inner-entry parse is currently duplicated between the two — deliberate prototype-phase choice; unify in productionization.)*

**Why per-unit voltage:** the code uses a **single global** `RunningMinMax` for voltage across all nodes. Per-unit keeps every node ~0.95–1.05 → clean [0,1] mapping. Physical units span multiple voltage bases (load ~2448 V, source ~66,399 V) → a global scaler would compress load nodes into a tiny sliver → forecast signal lost. Per-unit input avoids this with zero added complexity (per-node scalers would also fix it but add real complexity — deliberately not done).

**Why radians:** the whole pipeline is radians-native; the estimator publishes `angleRad` (post its internal ±165/195° normalization) alongside degrees.

**Coordinated State Estimator changes (in its repo, live on the bus):** publishes `vpu`, `angleRad`, and a **`processStatus`** completion message. Also **withholds the first 12 estimates** (high error during SE init) and publishes starting at the 13th — so the first estimate the forecaster sees is already "good," aligning naturally with the no-leading-imputation "time zero."

**Published forecast message (forecaster → bus), current first-guess shape:**
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
Top level carries generic ADMS fields (`timestamp`, `simulation_id`); forecast-specific payload nested under **`Forecast`** (parallel to how estimates nest under `Estimate`). Published via `gapps.send(service_output_topic('state-forecaster', simid), json.dumps(fc_json))`, guarded by `gapps is not None` (file runs skip publishing). **Structure is a first guess and will likely change** — consumers may prefer a per-timestamp-then-nodes layout over the compact per-node vector layout; `base_time`-inside-`Forecast` duplication was considered and deferred. *Not yet exercised live* (needs a real forecast → blocked on non-realtime sims).

**Converter utility (`util/simple_to_real_json.py`):** converts legacy "simplified" files to the real SE format (splits node key → phase-less `ConnectivityNode` + `phase`; writes `vpu`/`angleRad`; lossless). The "simplified" format is now deprecated.

## 8b/8c. Robustness guards
- **Insufficient-data guard:** if a stream ends before any full training block (< `HIST+FUT` timestamps → zero training samples), the app skips training with a clear message and exits cleanly instead of crashing the `DataLoader`. Validated on 1-line, 50-line, single/multi, **and live bus** (a 120 s realtime run: 27 estimates → 0 samples → clean exit).
- **Cadence-guard:** observes raw spacing of the first ~20 real records (min spacing = true cadence); on mismatch with `TS_INCREMENT_SEC`, prints a **loud one-time banner** (configured vs. detected, consequence, fix). Lives in the per-record imputer (`make_imputer`), so it covers both file and bus paths, stream-native. Warns (doesn't abort). Suppresses per-gap spam once a mismatch is known. **Retired a footgun that silently produced plausible-but-wrong results twice during testing** (config/data cadence mismatch → imputer fabricates phantom records).

## 9. Assumptions & limitations
1. **Fixed, complete node set, identical every timestamp** — lazy init assumes it. True for GridLAB-D-derived estimates. Escape hatch (a CIM query) deliberately not built.
2. **Grid-aligned timestamps** — missing handled (imputation); misaligned passed through with a warning.
3. **No imputed/real flag reaches the model** — interpolated history treated as real. Acceptable now; validity-feature + retrain is future work.
4. **Large-gap fidelity** — linear interpolation degrades over many consecutive missing steps; the high-variability regime is better handled by raising `TS_INCREMENT_SEC`.
5. **Config must match data cadence** — now enforced at runtime by the cadence-guard.
6. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training set (memory + training time). Sufficient at 123-node.
7. **Multi-process forecast accuracy not independently measured** (forecasts the live front). Trustworthy by construction. Optional future compare-to-actual telemetry.
8. **Recurring data anomaly (07-01→07-03 window)** — elevated error across all runs; **data-driven** (shared load-profile schedule), not code. Note for evaluation/paper.
9. **Per-unit forecast output** — if physical-unit output is ever required: a nominal-voltage multiplier dict (reintroduces a dependency) or stream-derived multipliers (`v/vpu` per node). Deferred.
10. **Exotic-phase feature encoding** — `encode_phase` handles single-char `1/2/3` and `A/B/C`; multi-char (`s1`, `ABC`) would mis-encode the phase *feature*. Out of scope (phase-as-identity handles arbitrary strings fine).
11. **File vs. bus node identity differ today** — legacy files use `632.1`-style names; live bus uses UUID mRIDs. A model trained on one is not transferable to the other. Temporal artifact (old files); regenerating sims once non-realtime works will unify them to mRID-based. GridAPPS-D app stays mRID-based (message precedent). Optional mRID→name mapping for human-readable output files deferred.
12. **Publish path written but not yet executed live** (§3 blocker). Latent risk retired only when a real forecast fires the `gapps.send`.

## 10. Scope boundary (deliberate)
Centralized Forecaster is appropriate to ~123-node/274-phase-node (likely low thousands). **Not** for 9500-node: the SE scales centrally via grid **sparsity**; the Forecaster is a **dense** whole-network model with no sparsity to exploit, so training time binds. 9500-node needs **regional decomposition** — different architecture, out of scope. Memory target 32–48 GB (comfortable at 274 nodes; retention bounds memory regardless of run length).

## 11. Validation reference numbers & status
- **13-node baseline Voltage MAE (pu):** `0.001248, 0.001183, 0.001566, 0.001179, 0.001220, 0.001233`. All format/unit/refactor changes confirmed **byte-identical** to their baselines (e.g. 5-min baseline `0.002133, 0.001913, 0.002281, 0.002009, 0.002046, 0.002146`) — including the per-record imputer refactor.
- **123-node single-process** Voltage MAE ~0.0007–0.0019 pu. **Multi-process (274 nodes):** trainer keeps up with large margin; forecaster memory plateaus at retention horizon; imputation (20% dropped) → forecast MAE ratios 0.96–1.10 vs. baseline.
- **Live bus (13-node, 120 s realtime):** feeder received 27 real + 0 imputed, clean `processStatus`→DONE; trainer lazy-init 41 nodes, insufficient-data guard fired, pushed v1; forecaster connected for publish, adopted v1 (1107 rows = 27×41), 0 forecasts, clean shutdown. **End-to-end live plumbing validated; training/forecasting/publish pending the non-realtime fix.**

## 12. Code artifacts / repo layout (under `prototype/`)
- **`forecaster_single.py`** — validated single-process version; correctness reference + shared logic (`read_json_records`, `make_imputer`, `impute_missing_records` wrapper, `_pq_value`, `RollingBuffer`, `DNN`, incremental scalers, `assemble_input_vector`, `build_forecast_json`, scaler-state helpers, config).
- **`forecaster_multi.py`** — three-process app (feeder/trainer/forecaster) + GridAPPS-D integration; imports shared logic from `forecaster_single.py`. Feeder selects file vs. bus driver on `gappsd_simid` (None = file).
- **`util/`** — `csv_to_jsonl.py` (legacy), `simple_to_real_json.py`, `drop_timestamps.py`, `check_imputation.py`.
- Input data: real-format files (13/123-node, 1/5/15-min). GitHub repo; this summary stored there.

## 13. Next steps / roadmap
**Blocked on GridAPPS-D platform (non-realtime sims):**
- Accumulate a full training block on live data → first real train→forecast cycle → first live `gapps.send` (publish smoke test: does `fc_json` serialize/publish cleanly, does the topic accept it).

**Independent of the blocker:**
- **Finalize published-forecast message structure** with consumers (envelope under `Forecast` is a first guess; layout may change; possible `base_time` duplication).
- **Productionization pass** (deferred by design): merge `forecaster_single.py` + `forecaster_multi.py` into one module; possibly convert feeder/trainer/forecaster closures → classes; unify the duplicated SvEstVoltages-entry parse; decide whether file input is dropped entirely.
- **Optional:** accuracy-vs-actual telemetry; imputed/real validity feature + retrain; per-snapshot re-normalize optimization beyond ~thousands of nodes; synthetic-gap feeder option to demo imputation on small models (which don't naturally gap); mRID→node-name mapping for human-readable output.

**Design conventions to preserve:** validate each change against byte-identical baselines where possible; keep the app **query-free** (no CIM dependency) with the node set stream-derived; keep the **feeder as the single integration seam** (file vs. bus selected by `gappsd_simid`); **defer structural elegance to productionization, don't defer functional correctness**; prefer short/simple code (closures over classes until complexity warrants).

## 14. Key configuration parameters
| Param | Default | Meaning / note |
|---TS_INCREMENT_SEC` | 60 | estimate spacing; validated 300/900; **set 3 for live realtime testing**; must match data cadence (cadence-guard enforces) and divide 86,400 & 604,800 |
| `HIST` | 15 | input history samples (position-based) |
| `FUT` | 15 | forecast horizon samples (horizon = FUT × increment) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | 86,400 / 604,800 | lag features |
| `BLOCK_DAYS` | 2 | training block size |
| `RETENTION_DAYS` | 10 | rolling buffer horizon (≥ week-lag + block span; see note) |
| `EPOCHS_PER_BLOCK` | 8 | + early stopping (patience 2) |
| `MAX_WINDOW_SAMPLES` | 500,000 | per-block train-set cap (memory + training time) |
| `VAL_FRACTION` | 0.05 | validation split |
| `DROPOUT_P` | 0.03 | training regularization only |
| `CADENCE_CHECK_SAMPLES` | 20 | real records sampled for the startup cadence-guard |
| `FEED_RATE_HZ` | (test) | file-driver pacing; dev convenience only (bus driver doesn't pace) |
| `FEEDER_POLL_SEC` | 0.05 | bus-driver stay-alive poll interval while awaiting messages |
| `FORECAST_LOG_EVERY` | (tunable) | forecaster log throttle (bounds log volume; publish is NOT throttled) |
| `gappsd_simid` | (CLI arg) | GridAPPS-D simulation ID; **also the file-vs-bus flag** (None → file input) |

**Note on `RETENTION_DAYS = 10`:** Distribution load has a **weekly trend**, so ≥7 days must be retained to keep the 1-week lag feature populated. But a sample near the *start* of the current 2-day block needs its week-lag to reach 7 days before the block *start* — ~9 days before the block *end*. The original 8-day value silently dropped those rows, degrading the week-lag for early-in-block samples. Raised to **10** to guarantee no week-lag degradation across the full block *and* keep retention an even multiple of the 2-day block (5 blocks). Enforced by a config assertion (`retention ≥ week-lag + HIST × increment`).

---

That completes the v4 summary. Closing notes:

**Coverage** — v4 is fully self-contained and captures everything through today: the complete GridAPPS-D bus integration (sub + publish-out), the per-record imputer refactor (`make_imputer`), the feeder driver-split (file/bus with shared `emit`), the `Forecast`-wrapped output message, and the live-data validation results. This plus the source files reconstitutes the project with no dependence on prior summaries or this conversation. The `gappsd_simid`-as-flag detail and the non-realtime blocker are both prominently captured.

**For a new chat** (if this one ends): upload the v4 summary **plus** `forecaster_single.py`, `forecaster_multi.py`, and the `util/` scripts. Use **paste-in-a-code-fence** for any code you need me to review in full (uploads get chunked/partially retrieved, as we learned — the code fence is the reliable channel; preview before sending to catch mangling).

**Honest caveat** — as always, this reflects my understanding; read it through before relying on it with colleagues and correct anything I've mischaracterized. You know the code and domain better than the summary does.

**Where you're leaving things:** a genuinely satisfying milestone — the state-forecaster is now a bus-connected ADMS service on both ends, with every code path implemented and validated as far as the platform allows. The one thing standing between "plumbing complete" and "producing live forecasts" is the non-realtime-simulation fix, which is squarely in the GridAPPS-D platform team's court, not yours. That's a great place to hand off to the weekend (or the platform devs) and pick up when non-realtime is restored.

