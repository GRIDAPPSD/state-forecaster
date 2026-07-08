
# State Forecaster App — Prototype Summary

- Last updated: July 7, 2026
- Authors: Gary Black and Anthropic Claude Opus 4.8 (AI used for design reasoning and code development starting from forecasting neural network algorithm code by Avijit Das)

## 1. Purpose of this document
Captures the design, rationale, current status, features, assumptions, and limitations of the prototype **State Forecaster** application. Intended to (a) understand *what was built and why*; (b) future development sessions, to resume AI work with full context. It also frames the decision of whether the app is ready for GridAPPS-D integration or whether specific features/limitations should be addressed first.

## 2. What the app is and where it sits
The **State Forecaster** is a Python application that forecasts near-future distribution-system state (per-phase-node **Voltage magnitude (pu)**, **Angle (rad)**) from a stream of state estimates. Production data pipeline:

```
GridLAB-D / OpenDSS (simulation measurements)
  → State Estimator (existing C++ app; sparse-matrix EKF; publishes state estimates to ActiveMQ bus)
     → State Forecaster (THIS app; Python; subscribes to estimates, forecasts, publishes to bus)
        → other ADMS apps consume forecasts
```
- The State Estimator is the only C++ app in GridAPPS-D (chosen because Python/Julia sparse-matrix libraries didn't scale to required matrix dimensions). It uses its own queue-draining/**averaging** design to avoid falling behind when estimates are slow to produce.
- The values the Forecaster ingests are **State Estimator outputs**, one bus message per timestamp.
- **Origin:** started from a validated single-process NN script (written by Avijit Das to validate the forecasting approach and support a journal paper). This project's deliverable is to turn that into a **bus-integrated, streaming ADMS service**.

**Test grids:** IEEE 13 Node feeder → **41 phase-nodes**; IEEE 123 Node feeder → **274 phase-nodes**. ("Node" is the official IEEE name for *locations*; expanding to per-phase electrical points gives the larger phase-node counts the model actually operates on.) A 9500-node model exists but is **explicitly out of scope** (see §10).

## 3. Current status (headline)
- **Changes 1, 2 (Steps A & B), and 3 are complete and validated.** The prototype app now runs as **three concurrent processes** (feeder → trainer + forecaster) simulating streaming ingestion, with training and forecasting happening simultaneously and forecasts produced per incoming estimate.
- **Not yet done:** GridAPPS-D bus integration (the feeder's file-read would be replaced by a bus subscription; forecasts would be published to the bus).
- Validated on both 13-node and 123-node feeders.

## 4. Why the single-process → multi-process transformation was a major effort
The original code and the final architecture solve the *same forecasting math*, but under **fundamentally different systems constraints**. The original script had the luxury of **global, upfront knowledge**; the production shape has **none of it**. Nearly every subsystem had to be re-engineered as a result:

| Concern | Original (single process, read whole file) | Now (streaming, three processes) |
|---|---|---|
| **Data availability** | Entire dataset in memory upfront; any time window instantly accessible | One timestamp at a time; must maintain a **rolling buffer** with **retention + eviction** to bound memory |
| **Normalization (scalers)** | Fit once on the full training period (needs all data) | **Incremental/causal scalers** (running min/max, Welford) updated per block; no future peeking |
| **Node set discovery** | Known by scanning the whole file | Discovered **lazily from the first streamed record**; no file/queries |
| **Control flow** | Simple sequential loop: train blocks, then forecast | **Concurrent processes** with queues, lifecycle, clean shutdown (sentinel), per-process logging |
| **Train vs. forecast timing** | Interleaved in one loop (forecasting blocked while training) | **Decoupled processes**: training (~48 min/block) and forecasting (~per second) run in parallel |
| **Model sharing** | Same in-memory object | Model **weights + scaler state serialized and handed across the process boundary** (bytes blob, not shared memory) |
| **Forecast unit** | One big batched forecast over a whole 2-day block, 6× per run | **One forecast per incoming estimate**, on demand — requiring **incremental normalization** to stay fast |
| **History for a forecast** | Trivially available | Forecaster must maintain its **own gapless recent-history buffer** (model needs recent trajectory, not just latest estimate) |

The recurring theme: **removing "global knowledge" forces every component that relied on it to be redesigned.** Additionally, the concurrency itself introduced categories of work that simply didn't exist before — inter-process communication semantics (what to keep vs. discard on each channel), process lifecycle/shutdown, cross-process model transport (which hit a real PyTorch shared-memory pitfall, see §7), and the need to prove no behavioral drift was introduced at each step. That verification burden (re-validating against the trusted single-process outputs after each change) was itself a significant part of the effort — and is why the result is trustworthy.

## 5. Development history (stages, in order)
1. **Change 1 — forecast after every block.** Original trained 6 two-day blocks then forecast once; changed to forecast after each block (cumulative model). Restructured into functions.
2. **Removed MC-dropout + Excel export.** Forecast became a fast single forward pass (`eval()`+`no_grad()`). Dropout kept only as training regularization.
3. **Change 2, Step A — data source CSV+pandas → JSON**, all other logic held identical. Routed through a `read_json_records()` generator (the "streaming seam"). Removed `deg2rad` (angle already radians in JSON). **Validation: 13-node output byte-identical to the CSV version.**
   - A standalone `csv_to_jsonl.py` converter was written (stdlib only) producing line-delimited JSON: `{"timestamp": epoch, "nodes": {node: {P,Q,V,Angle}}}`, Angle in radians.
4. **Change 2, Step B — true streaming ingestion.** pandas removed from data path. Added: **RollingBuffer** (per-node raw ring, retention/eviction), **incremental scalers** (RunningMinMax for P/Q/V, RunningStandardizer/Welford for Angle), **forecast-then-train** ordering per block, emergent warm-up. Continual train/forecast over all data (7 blocks on the 14-day set) — the correct shape for an open-ended live run. **Validation: matched Step A accuracy on both 13- and 123-node feeders** (not byte-identical—incremental scalers legitimately differ—but Voltage MAE within a few percent per aligned forecast).
5. **Per-node forecast JSON output.** Added `build_forecast_json()` producing the single-base-timestamp, all-nodes, FUT-horizon structure in **physical units, epoch seconds** — the eventual bus-publish unit. Validated via three-phase angle structure (~0 / −120° / +120°).
6. **Change 3 — three-process split** (delivered in Pieces 1→3, each validated before the next):
   - **Piece 1:** three-process scaffolding with stubbed bodies — proved concurrency plumbing (spawn, queues, model handoff, DONE shutdown, per-process logs).
   - **Piece 2:** real training loop in the trainer process. (Block-1 epoch losses came out **byte-identical** to single-process Step B → proved the split introduced zero drift.)
   - **Piece 3:** real forecaster — incremental-normalize store, keep-all gapless ingestion + latest-only forecasting, full re-normalize on snapshot arrival, eviction, batched forward pass → JSON. A shared `assemble_input_vector()` function was extracted so trainer and forecaster build inputs *identically* (re-validated byte-identical in single-process before use — retiring the biggest correctness risk of the whole rework).

## 6. Key runtime cadence facts
- Timestamp increment: **60 s** (1-min data). In the live app, estimates arrive at **~1/sec** (real-world requirement); accelerated feed rates (4, 20, 500/sec) are **development conveniences only**, not performance targets.
- A 2-day training block ≈ **2,880 records** ≈ **~48 min** of real time to accumulate; training a block also takes minutes. Forecasting is a single forward pass (milliseconds).
- **Forecast-to-training-update ratio ≈ 2,880:1** — the asymmetry that *requires* separate processes.

## 7. Non-trivial architecture & design decisions (consolidated)

**Process/communication design (Change 3):**
- **Three processes:** `feeder` (reads records, paces, distributes), `trainer` (accumulates blocks, trains, publishes model snapshots), `forecaster` (ingests, forecasts each newest estimate).
- **Four channels, with deliberately different retention semantics:**

| Channel | Producer | Consumer | Semantics |
|---|---|---|---|
| Model snapshot | Trainer | Forecaster | **latest-only** (consumer drains, keeps newest; honors DONE) |
| Data → Trainer | Feeder | Trainer | **keep-all FIFO** (every record needed; no gaps) |
| Data → Forecaster | Feeder | Forecaster | **keep-all FIFO** (gapless history), but forecasts only the **latest** timestamp |

- **"Latest-only" moved *inside* the forecaster.** Early design had the forecaster's *data queue* be latest-only. **Corrected:** the forecaster must ingest **every** estimate to keep its recent-history window **gapless** (the model needs the recent trajectory, e.g. up to ~2 days of estimates accumulate after the last model snapshot). So it *keeps all data* but *forecasts only the newest timestamp* — discarding stale **forecast opportunities**, not stale **data**.
- **Model handoff = serialized bytes blob, NOT `share_memory()`.** `share_memory()` would let the forecaster read half-updated weights mid-optimizer-step and would require locking the training loop. Instead the trainer serializes a **consistent snapshot at a clean block boundary** (weights + scaler state) via `torch.save` to bytes. **A real bug was hit and fixed here:** transporting live torch tensors across a Queue uses shared-memory file descriptors owned by the sender; when the trainer exited with a snapshot still in flight, the receiver got `FileNotFoundError`. Bytes-blob transport eliminated this.
- **Scaler state travels *with* the model.** The forecaster does **not** fit scalers; it applies the trainer's. This keeps train/forecast normalization identical and keeps the forecaster minimal (a stated design goal: "as much as possible in the trainer").
- **Sentinel `DONE`** is a distinguished **queue item** marking end-of-stream for clean shutdown. In production the feeder derives this from the simulation's "closed" log message (or a State-Estimator-supplied end marker).
- **`spawn` start method** (required for CUDA + multiprocessing); each process inits CUDA independently.
- **Per-process log files** (`feeder.log`, `trainer.log`, `forecaster.log`) so each is independently `tail -f`-able. Forecaster logs judiciously (not per-poll) to bound log volume.

**Forecaster efficiency design (for scalability into thousands of nodes):**
- **Incremental normalization.** Normalizing one incoming row is O(nodes); the expensive **full re-normalize of the whole buffer happens only when a new snapshot's scalers arrive** (~per block, rare). This keeps per-estimate cost tiny and independent of buffer depth — the key to scaling.
- **Position-based HIST window + timestamp-based lag lookups**, matching the trainer exactly (see gap-handling in §9).

**Lazy, stream-self-describing initialization (deliberate loose coupling):**
- Node set is discovered from the **first streamed record** (`sorted(node keys)` → identical mapping across processes, robust to which record each sees first). **The app makes NO CIM/topology queries.** Rationale: the State Estimator queries CIM because it builds a physical network model; the Forecaster only needs "what nodes exist, in what order," which the estimate stream already carries. This minimizes ADMS coupling, keeps it testable offline, and keeps it portable to real field grids. (Escape hatch noted in §9.)

## 8. The retention-horizon change: 8 → 10 days (why it was necessary)
This is worth explaining carefully because the reasoning is subtle and is a deviation from Avijit's original code.

**Original value (8 days), and its logic:** Distribution load has a strong **weekly trend**, so the buffer must retain **≥ 7 days** for the **1-week lag feature** to be populated (rather than zero-flagged). Eight days was chosen as "7 days for the weekly trend, rounded up to an even multiple of the 2-day training block" → exactly **4 blocks**. This reasoning was correct *for training-sample construction*, but it only accounted for the lag needed at the **most recent** point in the buffer.

**Why 8 was not enough:** A forecast (or training sample) built for a base timestamp near the **start of the current 2-day block** needs its 1-week-lag lookup to reach **7 days before that block-start** — not 7 days before the block-*end*. Since the block itself spans 2 days, the oldest lag lookup a current-block sample can require reaches back:

```
(7 days week-lag) + (2 days block span) = 9 days before the block END
```

With only 8 days retained, rows in that 9th-day region have already been **evicted**, so the week-lag feature **silently degrades to the zero-flag** for samples early in the block — exactly the degradation the retention was supposed to prevent. In other words, 8 days protected the week-lag only at the newest edge of the buffer, not across the full span of samples the current block actually generates.

**Why 10 (not 9):** 9 days is the true minimum to avoid week-lag degradation across the whole current block. It was rounded up to **10** to keep retention an **even multiple of the 2-day block** (→ **5 blocks**), preserving the clean block-divisibility property the original 8-day choice valued, while guaranteeing the week-lag never degrades at block boundaries. A config assertion enforces `retention ≥ week-lag + HIST`.

**Net:** the bump from 8→10 fixes a real correctness issue (week-lag feature quietly disappearing for early-in-block samples) at the cost of ~25% more retained history — comfortably within the memory budget (§10).

## 9. Assumptions & limitations (know these before integration)
1. **Fixed, complete node set, consistent every timestamp.** Lazy init from the first record assumes every timestamp carries all nodes. **True for GridLAB-D-derived estimates; may not hold for real field data** (partial node sets, appearing/disappearing nodes). If violated, the escape hatch is a one-time CIM topology query at startup — deliberately *not* built now.
2. **Regular, gapless timestamps.** Both the original file-based code *and* the streaming version assume evenly-spaced consecutive estimates. **The real State Estimator can produce timestamp gaps** at large node counts (its averaging/queue-drain assigns the latest timestamp to an estimate spanning several). Current handling: **no crash** (HIST is position-based = "last 15 estimates"; missing lags hit the zero-flag path; a forecast base timestamp with no matching row per node is skipped). But gappy data is silently treated as regular → **forecast-quality degradation, not a failure.** A proper policy (skip/pad/interpolate) is deferred; mainly relevant at large node counts.
3. **Forecast horizon FUT = 15** (15 min). Short by design; belief (from prior testing) is that with continuous per-estimate forecasting there won't be prediction gaps. May be increased after evaluation.
4. **`MAX_WINDOW_SAMPLES = 500,000`** caps per-block training-set size — bounds both memory and **training time**. Confirmed sufficient at 123-node (accuracy held—actually better than 13-node—despite the cap engaging from block 1 on <10% of available data). Raising it is deferred (would increase training time → staler model between updates).
5. **Multi-process forecast *accuracy* not independently re-measured.** It's trustworthy *by construction* (shared `assemble_input_vector` proven byte-identical; scalers handed over intact; JSON physically valid via three-phase check). No ground-truth MAE is computed in the live forecaster (it forecasts the front, before actuals exist). An optional "compare-to-actual-when-it-arrives" evaluation mode could be added later.
6. **Recurring data anomaly (07-01→07-03 window):** elevated forecast error in this window across **all four runs** (13A, 13B, 123A, 123B). Confirmed **data-driven** (shared load-profile schedule), not a code artifact. Note for evaluation/paper.
7. **Multi-process validation (13- and 123-node):** The three-process app was run end-to-end on both feeders. Trainer epoch losses matched single-process Step B (block-1 byte-identical), confirming no drift from the split. Forecast JSON validated via three-phase angle structure at both 41 and 274 nodes. **Memory bounding confirmed:** forecaster normalized-store row count plateaus at the retention horizon (~590K at 41 nodes, ~3.95M at 274 nodes) regardless of run length. **Per-snapshot full re-normalize cost:** ~tens of seconds at 274 nodes (a brief forecasting pause, rare — ~per training block); scales with node count, flagged for the thousands-node revisit.
8. **Forecaster keep-up is rate-dependent (as designed).** At the real ~1/sec estimate rate, the forecaster keeps up comfortably (forward pass is ms; only the rare re-normalize is heavy). At accelerated *test* rates (e.g., 4/sec on 274 nodes) it drifts behind the data front — forecasting fewer timestamps (latest-only), but never losing data or breaking history (keep-all ingestion). This degrades forecast *frequency*, not correctness, and only under artificial test loads. **Recommend a one-time rate-1 confirmation run** to observe base_time tracking the front, though the design guarantees it.

## 10. Scope boundary (deliberate)
The **centralized** Forecaster is appropriate up through ~123-node/274-phase-node scale (and likely low thousands). It is **not** designed for the 9500-node model. Reason: the State Estimator scales centrally by exploiting the grid's physical **sparsity**; the Forecaster is a **dense** learned model spanning the whole network with no sparsity to exploit, so training time becomes the binding constraint. The 9500-node case would require **regional decomposition** (partition, train sub-forecasters in parallel, reconcile) — a different architecture, out of scope. Memory target: hosts with **32–48 GB** (comfortable at 274 nodes; retention bounds memory regardless of run length).

## 11. Readiness assessment & suggested prioritization
**Ready now:** the concurrency architecture, streaming ingestion, model/scaler handoff, and forecast output format are validated end-to-end on both feeders. The feeder is explicitly the **GridAPPS-D integration seam** (swap file-read for a bus subscription; publish forecasts to the bus).

**Candidate work, in rough priority order — for the team to decide:**
- **(A) Proceed to GridAPPS-D integration (Change 3c)?** The architecture is shaped for it. Gating question: are the §9 assumptions (esp. #1 fixed/complete node set, #2 gapless timestamps) acceptable for the *initial* integration target (GridLAB-D-driven, where they hold)? If yes, integration can start.
- **(B) Timestamp-gap robustness (§9.2)** — needed before trusting large-node or field data; not needed for small-feeder GridLAB-D integration.
- **(C) Optional accuracy-vs-actual evaluation mode (§9.5)** — have the forecaster log/compare its forecast against the real estimate when that timestamp later arrives, producing live MAE. Valuable for the *evaluation* phase; not required for integration to function.
- **(D) Node-set-from-CIM escape hatch (§9.1)** — only if/when targeting real field data or non-uniform node sets. Deliberately deferred.
- **(E) Forecast horizon tuning (`FUT`, §9.3)** and **retention/cap tuning (§9.4)** — evaluation-phase knobs, best tuned once running against the live (or live-like) feed where their real impact is measurable.
- **(F) Full re-normalize cost at scale (§7)** — the per-snapshot full buffer re-normalize is O(retained rows); fine at 274 nodes, worth profiling if pushing toward thousands. Micro-optimizations (cache time-features, pre-tensorize lags) are structured to be easy to add if profiling ever demands it.

**Recommendation:** items **A and B are the fork in the road.** If the initial GridAPPS-D target is GridLAB-D-driven small/medium feeders (where the gapless/complete-node assumptions hold), **integration (A) can proceed now**, with (B), (C), (D) scheduled as the target expands toward larger or field data. If the team wants the forecaster robust to the State Estimator's real timestamp-gap behavior *before* any integration, do **(B) first**.

## 12. Code artifacts / file layout
- **`forecaster_single.py`** — the validated single-process streaming version (Step B + shared `assemble_input_vector` + JSON output). Still runnable standalone; serves as the **correctness reference** and the source of shared logic (`RollingBuffer`, `DNN`, scalers, `assemble_input_vector`, `build_forecast_json`, scaler-state helpers, config constants).
- **`forecaster_multi.py`** — the three-process app (feeder/trainer/forecaster), importing shared logic from `forecaster_single.py`.
- **`csv_to_jsonl.py`** — standalone CSV→line-delimited-JSON converter (stdlib only; Angle from `Angle_rad`).
- **Input data:** `results_data_forecasting_13.json`, `results_data_forecasting_123.json` (line-delimited; one timestamp record per line).
- **Logs:** `feeder.log`, `trainer.log`, `forecaster.log`.

## 13. Key config parameters (all explicit/configurable)
| Param | Value | Meaning / note |
|---|---|---|
| `TS_INCREMENT_SEC` | 60 | timestamp spacing |
| `HIST` | 15 | input history samples (15 min) |
| `FUT` | 15 | forecast horizon samples (15 min) |
| `DAY_LAG_SEC` / `WEEK_LAG_SEC` | 1d / 7d | lag features |
| `BLOCK_DAYS` | 2 | training block size |
| `RETENTION_DAYS` | 10 | rolling buffer horizon (see §8) |
| `EPOCHS_PER_BLOCK` | 8 | + early stopping (patience 2) |
| `MAX_WINDOW_SAMPLES` | 500,000 | per-block train-set cap (§9.4) |
| `VAL_FRACTION` | 0.05 | validation split |
| `FEED_RATE_HZ` | (test) | feeder pacing; dev convenience only |
| `DROPOUT_P` | 0.03 | training regularization only |

---

That completes the summary. A few notes on using it:

**For resuming in a new chat:** paste this summary *plus* your current `forecaster_single.py` and `forecaster_multi.py`. The summary gives the "why"; the code gives the "what." One caveat worth remembering — given the paste-mangling issue (underscores→Markdown, lost indentation) when copying out of your VM, use the workarounds we discussed (preview the code block before sending, or wrap in triple-backticks) so a fresh session gets clean code.


