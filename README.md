# Failure Whisperer

Acoustic anomaly detection for machine condition monitoring — listen to a
machine, learn what "healthy" sounds like, flag and explain deviations.
Modeled on [DCASE 2026 Challenge Task 2](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
(unsupervised, first-shot anomalous sound detection).

> **If you're an AI coding agent (Claude Code, Cursor, Kiro, etc.):**
> read `AGENTS.md` first, not this file. It has the full technical
> context, current build status, and constraints you must follow.
> This README is the human-friendly summary.

---

## TL;DR — where things stand right now

**Phase 1 and Phase 2 of the 15-day plan are built and running
end-to-end — but on placeholder data, not real inputs.** Two things need
to be swapped in before any results here mean anything real:

| Placeholder in use now | Needs to become | Why it's a placeholder |
|---|---|---|
| `ReferenceEncoder` (hand-crafted spectral features) | Real **BEATs** model | No GPU/network access to download the pretrained checkpoint in the dev sandbox this was built in |
| Synthetic generated audio | Real recorded machine audio + a real physical fault | No microphone or physical rig available in that sandbox |

Everything else — windowing, chunking, centroid/kNN scoring, AUC
evaluation, plotting — is real, tested code that will keep working
unchanged once you plug in the real encoder and real audio.

**AUC = 1.00 in the current test runs is NOT a real result.** It just
proves the pipeline plumbing works. Synthetic data is deliberately easy
to separate — don't get excited about that number, and don't skip the
real Day 5 separation test once real data is in.

---

## What to do next (in order)

1. **Get a machine with internet + ideally a GPU.**
2. **Install real dependencies:**
   ```bash
   pip install torch torchaudio
   ```
3. **Get a real BEATs checkpoint** — follow the exact steps written inside
   `src/encoders/beats_encoder.py` (docstring at the top of the file).
   Fill in the TODOs in that file.
4. **Record real healthy audio** (Day 2 of the plan — 5-10 min of a
   fan/motor, various clip positions). Replace the call to
   `generate_healthy_motor()` in the scripts with `load_wav()` on your
   real recording (see `src/audio/io.py`).
5. **Re-run in order:**
   ```bash
   cd scripts
   python3 day1_embedding_shape.py     # confirms real BEATs loads + embedding shape
   python3 day2_chunk_and_cluster.py   # real clustering tightness, 1s vs 3s
   python3 day3_baseline_score.py      # real baseline stability check
   ```
6. **Create a real physical fault** (Day 4 — imbalance a fan blade with
   tape, loosen something). Record it, replace the faulty-audio generator
   call the same way.
7. **Re-run the real separation checkpoint:**
   ```bash
   python3 day5_separation_test.py     # THIS is the real go/no-go decision
   python3 day6_compare_methods.py     # centroid vs kNN, on real data
   ```
8. Log every real decision (window size, which method won, why) in
   `docs/DECISIONS.md` — the template is at the bottom of that file.

If real Day 5 separation looks weak (AUC well below the synthetic 1.00
you'll be used to seeing) — that's expected and useful information, not a
failure. Try a different window size, kNN instead of centroid, or CLAP
instead of BEATs, exactly as the original plan anticipates.

---

## Project structure

```
AGENTS.md              ← full technical context for AI agents (read this, not just this README)
CLAUDE.md               ← one-line pointer to AGENTS.md for Claude Code
docs/
  DECISIONS.md          ← running log of every design choice + reasoning
src/
  audio/
    synth.py            ← synthetic audio generator (SWAP for real recordings)
    io.py                ← real wav load/save + windowing utilities
  encoders/
    base.py              ← the interface every encoder must implement
    reference_encoder.py ← current stand-in (works, but isn't BEATs)
    beats_encoder.py      ← real BEATs skeleton — fill in locally, see its docstring
  scoring/
    baseline.py           ← centroid + kNN anomaly scoring (real, tested)
  evaluate/
    separation.py          ← AUC computation + histogram plotting (real, tested)
scripts/
  day1_embedding_shape.py
  day2_chunk_and_cluster.py
  day3_baseline_score.py
  day5_separation_test.py
  day6_compare_methods.py
data/     ← generated/recorded .wav files (gitignore raw audio >5MB)
outputs/  ← generated plots (e.g. day5_separation.png)
```

## Setup

```bash
pip install -r requirements.txt   # numpy/scipy/sklearn/matplotlib only, for now
cd scripts
python3 day1_embedding_shape.py   # sanity check: should print an embedding shape in <1s
```

To move to real BEATs, also run:
```bash
pip install torch torchaudio
```
then follow the setup steps in `src/encoders/beats_encoder.py`.

## Design principles this project follows

- **Anomaly score convention: HIGHER = more anomalous.** Never flip this
  without updating it everywhere (scoring, thresholds, dashboard).
- **Never train on faulty data** — the whole point is detecting unknown
  fault types from healthy-only training, per DCASE Task 2 rules.
- **Every encoder implements the same interface** (`src/encoders/base.py`)
  so swapping BEATs ↔ CLAP ↔ anything else never touches downstream code.
- **Log decisions, don't just make them** — `docs/DECISIONS.md` exists so
  nobody re-derives (or contradicts) a choice that was already tested.

## Multi-agent / team workflow

This repo is set up so any team member can use any AI coding tool without
re-explaining the project:
- **Claude Code:** reads `CLAUDE.md`, which imports `AGENTS.md`.
- **Cursor:** reads `AGENTS.md` natively from repo root, no config needed.
- **Kiro:** paste `AGENTS.md` content into its steering setup on first use.

The only rule that keeps this working: **whoever finishes a session
updates Section 4 ("Where We Are") and the Decisions Log in `AGENTS.md`
before stopping.** That's the entire mechanism that lets you close this
project for weeks and pick it back up (with any tool) without confusion.
