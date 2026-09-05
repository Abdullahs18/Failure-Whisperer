"""
Day 6 — Compare methods

Try centroid vs kNN, whichever separation looks better/more stable.
Decide: lock in scoring method based on evidence, not preference.
"""

import sys
import numpy as np

sys.path.insert(0, "..")
from src.audio.synth import generate_healthy_motor, generate_faulty_motor
from src.audio.io import chunk_into_windows
from src.encoders.reference_encoder import ReferenceEncoder
from src.scoring.baseline import CentroidScorer, KNNScorer
from src.evaluate.separation import evaluate_separation

SR = 16000
WINDOW_S = 3.0


def build_data():
    encoder = ReferenceEncoder(sr=SR)

    train_audio = generate_healthy_motor(duration_s=60.0, sr=SR, seed=0)
    train_windows = chunk_into_windows(train_audio, SR, window_s=WINDOW_S)
    train_embeddings = np.array([encoder.embed(w, SR) for w in train_windows])

    healthy_test = generate_healthy_motor(duration_s=30.0, sr=SR, seed=42)
    healthy_windows = chunk_into_windows(healthy_test, SR, window_s=WINDOW_S)
    healthy_embeddings = np.array([encoder.embed(w, SR) for w in healthy_windows])

    faulty_audio = generate_faulty_motor(duration_s=30.0, sr=SR, seed=7, fault_type="imbalance")
    faulty_windows = chunk_into_windows(faulty_audio, SR, window_s=WINDOW_S)
    faulty_embeddings = np.array([encoder.embed(w, SR) for w in faulty_windows])

    return train_embeddings, healthy_embeddings, faulty_embeddings


def main():
    train_emb, healthy_emb, faulty_emb = build_data()

    results = {}

    centroid_scorer = CentroidScorer()
    centroid_scorer.fit(train_emb)
    h_scores = centroid_scorer.score_batch(healthy_emb)
    f_scores = centroid_scorer.score_batch(faulty_emb)
    auc, verdict = evaluate_separation(h_scores, f_scores, "Centroid method",
                                         "../outputs/day6_centroid.png")
    results["centroid"] = auc
    print(f"Centroid AUC: {auc:.4f} — {verdict}")

    for k in [3, 5, 10]:
        knn_scorer = KNNScorer(k=k)
        knn_scorer.fit(train_emb)
        h_scores = knn_scorer.score_batch(healthy_emb)
        f_scores = knn_scorer.score_batch(faulty_emb)
        auc, verdict = evaluate_separation(h_scores, f_scores, f"kNN (k={k}) method",
                                             f"../outputs/day6_knn_k{k}.png")
        results[f"knn_k{k}"] = auc
        print(f"kNN (k={k}) AUC: {auc:.4f} — {verdict}")

    print("\n--- Summary ---")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:15s} AUC = {auc:.4f}")
    best = max(results, key=results.get)
    print(f"\nBest method: {best} (AUC={results[best]:.4f})")
    print("Log this choice + numbers in docs/DECISIONS.md.")


if __name__ == "__main__":
    main()
