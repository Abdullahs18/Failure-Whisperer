"""
Day 3 — First anomaly score, no fault yet

Build: compute centroid of healthy embeddings, compute distance of new
healthy windows to that centroid.
Test: does the score stay low and stable for held-out healthy audio?
Decide: is centroid distance giving stable low scores on healthy-only
data? Yes -> proceed. No -> debug window size/preprocessing.
"""

import sys
import numpy as np

sys.path.insert(0, "..")
from src.audio.synth import generate_healthy_motor
from src.audio.io import chunk_into_windows
from src.encoders.reference_encoder import ReferenceEncoder
from src.scoring.baseline import CentroidScorer

SR = 16000
WINDOW_S = 3.0  # <- update this if Day 2 picked a different winner


def main():
    encoder = ReferenceEncoder(sr=SR)

    # "Training" healthy audio -> builds the baseline
    train_audio = generate_healthy_motor(duration_s=60.0, sr=SR, seed=0)
    train_windows = chunk_into_windows(train_audio, SR, window_s=WINDOW_S)
    train_embeddings = np.array([encoder.embed(w, SR) for w in train_windows])

    scorer = CentroidScorer()
    scorer.fit(train_embeddings)

    # Held-out healthy audio, different random seed -> simulates a new recording session
    test_audio = generate_healthy_motor(duration_s=30.0, sr=SR, seed=42)
    test_windows = chunk_into_windows(test_audio, SR, window_s=WINDOW_S)
    test_embeddings = np.array([encoder.embed(w, SR) for w in test_windows])
    scores = scorer.score_batch(test_embeddings)

    print(f"Held-out healthy scores (n={len(scores)}):")
    print(f"  mean:  {scores.mean():.4f}")
    print(f"  std:   {scores.std():.4f}")
    print(f"  min:   {scores.min():.4f}")
    print(f"  max:   {scores.max():.4f}")

    cv = scores.std() / (scores.mean() + 1e-8)
    print(f"\nCoefficient of variation: {cv:.3f}")
    if cv < 0.3:
        print("PASS: scores are low and stable on held-out healthy data. Proceed to Phase 2.")
    else:
        print("WARNING: high variance in healthy scores. Consider a different window size or "
              "check for preprocessing inconsistencies before Phase 2.")


if __name__ == "__main__":
    main()
