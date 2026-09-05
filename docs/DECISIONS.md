# Decisions Log

Append-only. Never delete or rewrite history — if a decision gets reversed
later, add a new entry saying so; don't erase the old one.

Format: `Date — decision — what was tried — why this won`

---

## [build session 1] — Used a reference (non-BEATs) encoder to build the pipeline

**Decision:** Built and validated the full pipeline (windowing → embedding →
centroid/kNN scoring → AUC evaluation) using `ReferenceEncoder`
(hand-crafted spectral features), NOT real BEATs.

**Why:** The dev environment for this session had no GPU, no network
access to Microsoft's checkpoint host, and no microphone. Rather than
block on those, built every OTHER piece of the pipeline against a fixed
interface (`AudioEncoder` in `src/encoders/base.py`) so real BEATs can be
dropped in later with zero changes anywhere else.

**Action required before trusting real results:** Swap `ReferenceEncoder`
for `BEATsEncoder` (see `src/encoders/beats_encoder.py` for setup steps)
before drawing any real conclusions about separation quality. The AUC
numbers below are pipeline sanity checks, not real evidence about whether
BEATs will separate your actual machine's healthy/faulty sound.

---

## [build session 1] — Used synthetic audio to stand in for Day 2/Day 4 recordings

**Decision:** `src/audio/synth.py` generates synthetic "healthy motor" and
"faulty motor" audio (imbalance/looseness/bearing fault types) instead of
using real recordings.

**Why:** No microphone or physical rig available in this session.

**Action required:** Replace with real recordings the moment you have mic
access (Day 2) and a real physical fault (Day 4). The pipeline code does
not change — only swap what feeds into `chunk_into_windows()`.

---

## [build session 1] — Window size: tested 1s vs 3s on synthetic healthy audio

**Tried:** 1.0s windows (60 windows from 60s audio) vs 3.0s windows (20
windows from 60s audio).

**Result:** Both clustered extremely tightly (mean pairwise cosine
distance ~0.0001 and ~0.00003 respectively) — expected, since the
synthetic healthy audio has no real-world variability (mic noise, room
acoustics, etc.).

**Decision:** Defaulted to 3.0s windows for Phase 1–2 scripts, matching
the plan's suggestion. **This is not yet a real decision** — re-run
`scripts/day2_chunk_and_cluster.py` once real recorded audio is available;
real-world noise will likely show a meaningful gap between 1s and 3s that
synthetic audio can't reveal.

---

## [build session 1] — Scoring method: centroid vs kNN (k=3,5,10)

**Tried:** Centroid distance vs kNN distance at k=3, 5, 10, all against
synthetic healthy vs. synthetic "imbalance" fault.

**Result:** All four methods scored AUC = 1.0000.

**Decision:** Not a real decision — synthetic imbalance fault is too
cleanly separable to differentiate scoring methods (this is expected;
synthetic data is deliberately easy to validate plumbing, not to reveal
subtle method trade-offs). **Re-run `scripts/day6_compare_methods.py`
with real BEATs embeddings and real recorded fault audio** — that's where
centroid vs kNN will actually start to differ.

---

## Template for your next real entry

```
## Day X — [decision title]
**Tried:** ...
**Result:** ...
**Decision:** ...
**Owner:** [name/agent]
```
