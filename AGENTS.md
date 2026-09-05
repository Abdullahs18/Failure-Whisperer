# Failure Whisperer — Agent Instructions

This file is the single source of truth for any AI coding agent working on
this project (Claude, Cursor, Kiro, Antigravity, etc.). Read this fully
before writing any code. Update Section 4 and Section 8 before ending your
session — the next agent depends on it.

---

## 1. What This Is

Real-time acoustic anomaly detection for machine condition monitoring.
Healthy-sound baseline → anomaly score → evidence → LLM explanation →
dashboard. Modeled on DCASE 2026 Task 2 (unsupervised, first-shot
anomalous sound detection).

Reference: https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

## 2. Core Concept (do not violate)

- Baseline = embedding centroid (or kNN set) from HEALTHY audio only.
- Anomaly score = distance of a new window's embedding to that baseline.
- **Convention: HIGHER score = MORE anomalous.** Keep this direction
  consistent everywhere.
- No training on faulty data, ever.

## 3. Stack & Models

- Python 3.10+, numpy/scipy/scikit-learn/matplotlib (installed, working)
- torch/torchaudio — NOT yet installed anywhere in this project. Required
  only once you wire in real BEATs (see Section 4).
- Primary encoder (target): BEATs — https://arxiv.org/abs/2212.09058
- Evidence-layer encoder (target): CLAP — https://arxiv.org/abs/2206.04769
- LLM explanation layer: Claude API (not yet built — Phase 4)
- Dashboard: not yet built — Phase 5

## 4. Where We Are

**Phase 1 & 2 pipeline is built and verified working end-to-end, but
running on STAND-IN data, not real inputs.** Read this carefully before
continuing — two things are placeholders and must be swapped before
results mean anything:

1. **Encoder:** Using `src/encoders/reference_encoder.py` (hand-crafted
   spectral features), NOT real BEATs. `src/encoders/beats_encoder.py` is
   a filled-in skeleton with exact setup steps in its docstring — do that
   on a machine with GPU + internet access, not in a sandboxed dev
   environment without network access to Microsoft's checkpoint host.
2. **Audio:** Using synthetic generated audio (`src/audio/synth.py`), not
   real recordings or a real physical fault. Swap the moment mic/rig
   access exists — nothing else in the pipeline changes.

**What's actually done and tested (Days 1, 2, 3, 5, 6 from the plan):**
- `scripts/day1_embedding_shape.py` — loads encoder, embeds a clip, prints
  shape. PASSED (68-dim embedding, ~4.6ms inference).
- `scripts/day2_chunk_and_cluster.py` — chunks into 1s/3s windows, checks
  clustering tightness. Both tight on synthetic data (expected — real
  audio will differ).
- `scripts/day3_baseline_score.py` — centroid baseline, scores held-out
  healthy audio. PASSED (low, stable scores).
- `scripts/day5_separation_test.py` — the critical checkpoint. AUC = 1.00
  on synthetic imbalance fault (expected to be this clean only because
  synthetic data is artificially easy — NOT evidence the real approach
  will work this well).
- `scripts/day6_compare_methods.py` — centroid vs kNN(3,5,10), all tied at
  AUC 1.00 on synthetic data (can't differentiate methods until real data
  is used).

**Not started:** Day 4 (real controlled fault — needs physical
access), Phase 3 (robustness/noise testing — needs real environment),
Phase 4, Phase 5.

**Next concrete step:** Someone with a machine (GPU optional but faster)
and network access needs to: (a) install torch/torchaudio, (b) download a
real BEATs checkpoint and fill in `beats_encoder.py`, (c) record real
healthy audio per Day 2's instructions, and re-run
`day2_chunk_and_cluster.py` and `day3_baseline_score.py` against real
data before trusting any results.

## 5. The 15-Day Plan (see docs/phase-notes/ for detail — create as you go)

- **Phase 1 (Days 1–3):** Environment + tiny working pipeline — BUILT
  (on stand-in data, see Section 4).
- **Phase 2 (Days 4–6):** Real controlled fault, separation test, method
  comparison — SCRIPTS BUILT (Day 4 needs physical hardware; 5 & 6 need
  re-running on real data once available).
- **Phase 3 (Days 7–9):** Robustness — noise, temporal smoothing,
  calibration length. NOT STARTED.
- **Phase 4 (Days 10–12):** Evidence layer, LLM explanation, integration.
  NOT STARTED.
- **Phase 5 (Days 13–15):** Dashboard, dry run, buffer. NOT STARTED.

## 6. Folder Structure & Ownership

```
src/
  audio/
    synth.py     — synthetic audio generator (STAND-IN, replace with real recordings)
    io.py        — wav load/save, windowing/chunking (real, keep using this)
  encoders/
    base.py             — AudioEncoder interface (real, keep using this)
    reference_encoder.py — stand-in encoder (works, but not BEATs — see Section 4)
    beats_encoder.py     — real BEATs skeleton, fill in locally
  scoring/
    baseline.py  — CentroidScorer, KNNScorer (real, tested, keep using this)
  evaluate/
    separation.py — AUC + histogram plotting (real, tested, keep using this)
scripts/
  day1_embedding_shape.py
  day2_chunk_and_cluster.py
  day3_baseline_score.py
  day5_separation_test.py
  day6_compare_methods.py
data/     — generated .wav files land here (gitignored, do not commit raw audio)
outputs/  — generated plots land here
docs/
  DECISIONS.md — read this, it explains exactly what's real vs. stand-in right now
```

**Interface contract (do not break without updating this file):**
- `encoders.embed(window, sr) -> np.ndarray` (fixed-length embedding)
- `scoring.score_batch(embeddings) -> np.ndarray` (higher = more anomalous)
- `evaluate.evaluate_separation(healthy_scores, faulty_scores, title, path) -> (auc, verdict)`

## 7. Non-Negotiable Constraints

- Never commit raw audio files >5MB — see `data/.gitignore`.
- Never fine-tune BEATs/CLAP from scratch — pretrained inference only.
- Log every threshold/window-size/method choice in `docs/DECISIONS.md`.
- Keep anomaly-score direction (higher = worse) consistent everywhere.
- **Do not treat any AUC number in this repo as real evidence until the
  encoder is real BEATs and the audio is real recordings.** Synthetic-data
  AUC = 1.00 is a pipeline sanity check, not a result.

## 8. Decisions Log

See `docs/DECISIONS.md` for the full log with reasoning. Latest entries
(as of this build session):
- Built full pipeline against `ReferenceEncoder` stand-in due to no
  GPU/network/mic in dev sandbox — interface designed so swapping to real
  BEATs requires no changes elsewhere.
- Window size and scoring method (centroid vs kNN) both inconclusive on
  synthetic data — genuinely decide these once real data is available,
  don't carry forward the synthetic "1.00 AUC for everything" result.

## 9. Before Ending Any Session

- [ ] Update Section 4 ("Where We Are")
- [ ] Append new choices to `docs/DECISIONS.md`
- [ ] Re-run affected scripts, paste real output — don't just claim it works
- [ ] Update Section 6 if folder structure changed
