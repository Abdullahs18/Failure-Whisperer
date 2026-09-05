"""
Encoder interface contract (see AGENTS.md Section 6).

Every encoder — the reference stand-in used for now, or real BEATs/CLAP
wired in later — MUST implement this exact interface so nothing else in
the pipeline (scoring, evaluation) needs to change when you swap encoders.
"""

from abc import ABC, abstractmethod
import numpy as np


class AudioEncoder(ABC):
    @abstractmethod
    def embed(self, window: np.ndarray, sr: int) -> np.ndarray:
        """
        Args:
            window: 1D float32 audio array, single window (e.g. 3s of audio)
            sr: sample rate of `window`
        Returns:
            1D np.ndarray embedding, fixed length for a given encoder instance
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError
