"""
Day 5's "single most important checkpoint": does the anomaly score
actually separate healthy from faulty windows?

Uses AUC (see DCASE Task 2 scoring) rather than eyeballing alone, so the
decision is backed by a number, not a vibe.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_auc(healthy_scores: np.ndarray, faulty_scores: np.ndarray) -> float:
    """
    1.0 = perfect separation, 0.5 = random guessing, <0.5 = scores are
    inverted (check your HIGHER-is-worse convention if this happens).
    """
    y_true = np.concatenate([np.zeros(len(healthy_scores)), np.ones(len(faulty_scores))])
    y_score = np.concatenate([healthy_scores, faulty_scores])
    return float(roc_auc_score(y_true, y_score))


def plot_separation(healthy_scores: np.ndarray, faulty_scores: np.ndarray,
                      title: str, save_path: str) -> None:
    plt.figure(figsize=(7, 4))
    bins = np.linspace(
        min(healthy_scores.min(), faulty_scores.min()),
        max(healthy_scores.max(), faulty_scores.max()),
        30,
    )
    plt.hist(healthy_scores, bins=bins, alpha=0.6, label="Healthy", color="#2a9d8f")
    plt.hist(faulty_scores, bins=bins, alpha=0.6, label="Faulty", color="#e76f51")
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def evaluate_separation(healthy_scores: np.ndarray, faulty_scores: np.ndarray,
                          title: str, save_path: str) -> Tuple[float, str]:
    auc = compute_auc(healthy_scores, faulty_scores)
    plot_separation(healthy_scores, faulty_scores, title, save_path)

    if auc >= 0.9:
        verdict = "Strong separation — proceed to Phase 3."
    elif auc >= 0.75:
        verdict = "Moderate separation — usable, but consider tuning window size or k before Phase 3."
    else:
        verdict = "Weak separation — do not proceed yet. Try a different window size, distance method, or encoder."

    return auc, verdict
