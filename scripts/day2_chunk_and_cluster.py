"""
Day 2 — Record real data
(Sandbox substitute: synthetic healthy motor audio — see src/audio/synth.py.
 Swap in a real recording the moment you have mic access.)

Build: chunk audio into windows (try 1s, 3s), embed each window.
Test: do windows from the same healthy recording cluster tightly?
(Eyeball cosine similarity between them.)
"""

import sys
import numpy as np

sys.path.insert(0, "..")
from src.audio.synth import generate_healthy_motor
from src.audio.io import chunk_into_windows, save_wav
from src.encoders.reference_encoder import ReferenceEncoder
from src.scoring.baseline import cosine_distance

SR = 16000


def cluster_tightness(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance within a set. Lower = tighter cluster."""
    n = len(embeddings)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(cosine_distance(embeddings[i], embeddings[j]))
    return float(np.mean(dists))


def main():
    healthy_audio = generate_healthy_motor(duration_s=60.0, sr=SR, seed=0)
    save_wav("../data/healthy_sample.wav", healthy_audio, SR)
    print(f"Generated {len(healthy_audio) / SR:.0f}s of stand-in healthy audio -> data/healthy_sample.wav")

    encoder = ReferenceEncoder(sr=SR)

    for window_s in [1.0, 3.0]:
        windows = chunk_into_windows(healthy_audio, SR, window_s=window_s)
        embeddings = np.array([encoder.embed(w, SR) for w in windows])
        tightness = cluster_tightness(embeddings)
        print(f"\nWindow size: {window_s}s -> {len(windows)} windows")
        print(f"Mean pairwise cosine distance (lower = tighter): {tightness:.4f}")

    print("\nDecide: whichever window size gave the lower distance clusters tighter.")
    print("Log the winner + reasoning in docs/DECISIONS.md.")


if __name__ == "__main__":
    main()
