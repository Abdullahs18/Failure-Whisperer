"""
Audio loading and windowing (chunking) utilities.

This is where Day 2's "chunk audio into windows (try 1s, 3s)" logic lives.
"""

from typing import List
import numpy as np
from scipy.io import wavfile


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a .wav file. Returns (mono float32 signal in [-1,1], sample_rate)."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return data, sr


def save_wav(path: str, signal: np.ndarray, sr: int) -> None:
    """Save a float32 [-1,1] signal to a 16-bit PCM wav file."""
    clipped = np.clip(signal, -1.0, 1.0)
    int_signal = (clipped * 32767).astype(np.int16)
    wavfile.write(path, sr, int_signal)


def chunk_into_windows(signal: np.ndarray, sr: int, window_s: float,
                         hop_s: float | None = None) -> List[np.ndarray]:
    """
    Split a 1D signal into fixed-length windows.

    window_s: window length in seconds (Day 2 asks you to try 1s and 3s)
    hop_s: step between window starts. Defaults to window_s (no overlap).
           Use hop_s < window_s for overlapping windows (more windows, smoother scoring).
    """
    window_len = int(window_s * sr)
    hop_len = int((hop_s if hop_s is not None else window_s) * sr)

    windows = []
    start = 0
    while start + window_len <= len(signal):
        windows.append(signal[start:start + window_len])
        start += hop_len
    return windows
