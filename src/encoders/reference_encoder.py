"""
Reference encoder — a lightweight, dependency-free stand-in for BEATs.

WHY THIS EXISTS:
Real BEATs needs: torch + torchaudio installed, a pretrained checkpoint
downloaded from Microsoft's release (not reachable from this sandbox's
network allowlist), and ideally a GPU for reasonable inference speed.
None of that is available here.

This encoder computes a deterministic, hand-crafted feature vector from a
log-mel-style spectrogram (band-energy summary + spectral shape stats).
It is NOT a learned model and will NOT match BEATs' semantic quality —
but it satisfies the exact same `AudioEncoder` interface, which means
every downstream script (windowing, centroid scoring, AUC evaluation)
built against it will keep working unmodified once you swap in
`beats_encoder.py` on your own machine (see that file for setup steps).

Use this to validate pipeline PLUMBING (does scoring/evaluation logic
work end-to-end), not to validate whether BEATs itself will separate
healthy vs. faulty sound well — that judgment still needs the real model.
"""

import numpy as np
from .base import AudioEncoder


class ReferenceEncoder(AudioEncoder):
    def __init__(self, sr: int = 16000, n_bands: int = 32):
        self.sr = sr
        self.n_bands = n_bands

    @property
    def embedding_dim(self) -> int:
        # n_bands mean + n_bands std + 4 spectral shape stats
        return self.n_bands * 2 + 4

    def embed(self, window: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sr:
            # naive resample by linear interpolation (fine for a stand-in encoder)
            duration = len(window) / sr
            new_len = int(duration * self.sr)
            window = np.interp(
                np.linspace(0, len(window), new_len, endpoint=False),
                np.arange(len(window)), window,
            )

        # STFT magnitude spectrogram
        n_fft = 1024
        hop = 512
        window_fn = np.hanning(n_fft)
        n_frames = max(1, (len(window) - n_fft) // hop + 1)

        spec = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * hop
            frame = window[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec[:, i] = np.abs(np.fft.rfft(frame * window_fn))

        # collapse into log-spaced frequency bands (mel-ish, without needing librosa)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
        band_edges = np.logspace(np.log10(20), np.log10(self.sr / 2), self.n_bands + 1)
        band_means = np.zeros(self.n_bands)
        band_stds = np.zeros(self.n_bands)
        for b in range(self.n_bands):
            mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
            if mask.sum() == 0:
                continue
            band_energy = spec[mask, :].mean(axis=0)
            band_means[b] = np.log1p(band_energy.mean())
            band_stds[b] = np.log1p(band_energy.std())

        # global spectral shape stats
        overall = spec.mean(axis=1)
        total_energy = overall.sum() + 1e-8
        centroid = (freqs * overall).sum() / total_energy
        spread = np.sqrt(((freqs - centroid) ** 2 * overall).sum() / total_energy)
        rms = np.sqrt(np.mean(window ** 2))
        zero_cross = np.mean(np.abs(np.diff(np.sign(window)))) / 2.0

        embedding = np.concatenate([
            band_means, band_stds,
            [centroid / 1000.0, spread / 1000.0, rms, zero_cross],
        ]).astype(np.float32)

        return embedding
