"""
Synthetic machine-sound generator.

WHY THIS EXISTS:
Day 2 of the plan calls for recording 5-10 min of real healthy machine audio,
and Day 4 calls for a real controlled fault. This sandbox has no microphone
and no physical fan to imbalance, so this module generates realistic-enough
stand-in audio so the REST of the pipeline (windowing, embedding, scoring,
evaluation) can be built and verified end-to-end right now.

>>> Swap this out for real recordings the moment you have mic access. <<<
The rest of the codebase does not care where the audio came from.
"""

import numpy as np


def generate_healthy_motor(duration_s: float = 60.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """
    Simulates a healthy small motor/fan: a fundamental hum + a few harmonics,
    stable amplitude, plus a low noise floor. Mimics steady mechanical rotation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr

    fundamental = 120.0  # Hz, rough fan/motor rotation-induced hum
    signal = np.zeros_like(t)
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.15), (4, 0.05)]
    for mult, amp in harmonics:
        signal += amp * np.sin(2 * np.pi * fundamental * mult * t)

    # stable amplitude with tiny natural jitter (real motors aren't perfectly steady)
    jitter = 1.0 + 0.01 * rng.standard_normal(len(t))
    signal *= jitter

    noise_floor = 0.02 * rng.standard_normal(len(t))
    signal = signal + noise_floor

    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def generate_faulty_motor(duration_s: float = 60.0, sr: int = 16000, seed: int = 1,
                            fault_type: str = "imbalance") -> np.ndarray:
    """
    Simulates a faulty version of the same motor.

    fault_type:
      - "imbalance": periodic amplitude wobble (simulates a taped/unbalanced fan blade)
      - "looseness": extra broadband rattle noise bursts (simulates loose hardware)
      - "bearing":   added high-frequency grinding component
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr

    fundamental = 120.0
    signal = np.zeros_like(t)
    harmonics = [(1, 1.0), (2, 0.4), (3, 0.15), (4, 0.05)]
    for mult, amp in harmonics:
        signal += amp * np.sin(2 * np.pi * fundamental * mult * t)

    if fault_type == "imbalance":
        wobble_freq = 4.0  # Hz, slow amplitude wobble from an unbalanced blade
        wobble = 1.0 + 0.35 * np.sin(2 * np.pi * wobble_freq * t)
        signal *= wobble

    elif fault_type == "looseness":
        rattle = np.zeros_like(t)
        n_bursts = int(duration_s * 3)
        for _ in range(n_bursts):
            start = rng.integers(0, len(t) - sr // 10)
            length = sr // 20
            rattle[start:start + length] += 0.6 * rng.standard_normal(length)
        signal += rattle

    elif fault_type == "bearing":
        grind_freq = 3200.0
        grind = 0.25 * np.sin(2 * np.pi * grind_freq * t)
        grind *= (0.5 + 0.5 * rng.random(len(t)))  # irregular amplitude, not pure tone
        signal += grind

    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    jitter = 1.0 + 0.01 * rng.standard_normal(len(t))
    signal *= jitter
    noise_floor = 0.02 * rng.standard_normal(len(t))
    signal = signal + noise_floor

    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)
