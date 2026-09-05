"""
Anomaly scoring: centroid-distance and kNN-distance methods.

CONVENTION (see AGENTS.md Section 2): HIGHER score = MORE anomalous.
Keep this direction consistent everywhere.
"""

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = identical direction, 2 = opposite direction. Higher = more different."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_norm, b_norm))


class CentroidScorer:
    """Baseline = mean of healthy embeddings. Score = distance to that centroid."""

    def __init__(self):
        self.centroid: np.ndarray | None = None

    def fit(self, healthy_embeddings: np.ndarray) -> None:
        self.centroid = healthy_embeddings.mean(axis=0)

    def score(self, embedding: np.ndarray) -> float:
        if self.centroid is None:
            raise RuntimeError("Call fit() with healthy embeddings first.")
        return cosine_distance(embedding, self.centroid)

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        return np.array([self.score(e) for e in embeddings])


class KNNScorer:
    """Baseline = all healthy embeddings. Score = mean distance to k nearest healthy points."""

    def __init__(self, k: int = 5):
        self.k = k
        self.healthy_embeddings: np.ndarray | None = None

    def fit(self, healthy_embeddings: np.ndarray) -> None:
        self.healthy_embeddings = healthy_embeddings

    def score(self, embedding: np.ndarray) -> float:
        if self.healthy_embeddings is None:
            raise RuntimeError("Call fit() with healthy embeddings first.")
        distances = np.array([cosine_distance(embedding, h) for h in self.healthy_embeddings])
        k = min(self.k, len(distances))
        nearest = np.sort(distances)[:k]
        return float(nearest.mean())

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        return np.array([self.score(e) for e in embeddings])
