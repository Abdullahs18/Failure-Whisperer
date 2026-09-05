"""
Day 5 — Evaluate separation (the single most important checkpoint)

Plot healthy scores vs. faulty scores.
Test: is there a visible gap? Or do they overlap?
Decide: clear separation -> move forward. Muddy -> try different window
size, kNN instead of centroid, or CLAP instead of BEATs.

(Sandbox substitute: synthetic "imbalance" fault — see src/audio/synth.py.
 Swap in your real recorded fault the moment you have one, Day 4.)
"""

import sys
import numpy as np

sys.path.insert(0, "..")
from src.audio.synth import generate_healthy_motor, generate_faulty_motor
from src.audio.io import chunk_into_windows, save_wav
from src.encoders.reference_encoder import ReferenceEncoder
from src.scoring.baseline import CentroidScorer
from src.evaluate.separation import evaluate_separation

SR = 16000
WINDOW_S = 3.0


def main():
    encoder = ReferenceEncoder(sr=SR)

    train_audio = generate_healthy_motor(duration_s=60.0, sr=SR, seed=0)
    train_windows = chunk_into_windows(train_audio, SR, window_s=WINDOW_S)
    train_embeddings = np.array([encoder.embed(w, SR) for w in train_windows])
    scorer = CentroidScorer()
    scorer.fit(train_embeddings)

    healthy_test = generate_healthy_motor(duration_s=30.0, sr=SR, seed=42)
    healthy_windows = chunk_into_windows(healthy_test, SR, window_s=WINDOW_S)
    healthy_embeddings = np.array([encoder.embed(w, SR) for w in healthy_windows])
    healthy_scores = scorer.score_batch(healthy_embeddings)

    faulty_audio = generate_faulty_motor(duration_s=30.0, sr=SR, seed=7, fault_type="imbalance")
    save_wav("../data/faulty_sample.wav", faulty_audio, SR)
    faulty_windows = chunk_into_windows(faulty_audio, SR, window_s=WINDOW_S)
    faulty_embeddings = np.array([encoder.embed(w, SR) for w in faulty_windows])
    faulty_scores = scorer.score_batch(faulty_embeddings)

    auc, verdict = evaluate_separation(
        healthy_scores, faulty_scores,
        title="Day 5: Healthy vs Faulty anomaly score separation",
        save_path="../outputs/day5_separation.png",
    )

    print(f"Healthy scores  -> mean {healthy_scores.mean():.4f}, std {healthy_scores.std():.4f}")
    print(f"Faulty scores   -> mean {faulty_scores.mean():.4f}, std {faulty_scores.std():.4f}")
    print(f"\nAUC: {auc:.4f}")
    print(f"Verdict: {verdict}")
    print("Plot saved to outputs/day5_separation.png")


if __name__ == "__main__":
    main()
