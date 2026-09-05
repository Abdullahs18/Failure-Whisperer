"""
Day 1 — Concepts + environment
Build: script that loads an encoder, feeds it a random audio clip, prints
the embedding shape.
Test: does it run? What's the embedding dimensionality? How long does
inference take?
"""

import sys
import time
import numpy as np

sys.path.insert(0, "..")
from src.encoders.reference_encoder import ReferenceEncoder

SR = 16000


def main():
    encoder = ReferenceEncoder(sr=SR)

    rng = np.random.default_rng(0)
    random_clip = rng.standard_normal(SR * 3).astype(np.float32)  # 3s of noise

    start = time.time()
    embedding = encoder.embed(random_clip, SR)
    elapsed = time.time() - start

    print(f"Encoder: {type(encoder).__name__}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Declared embedding_dim: {encoder.embedding_dim}")
    print(f"Inference time: {elapsed * 1000:.2f} ms")
    assert embedding.shape[0] == encoder.embedding_dim, "Shape mismatch!"
    print("PASS: embedding shape matches declared embedding_dim")


if __name__ == "__main__":
    main()
