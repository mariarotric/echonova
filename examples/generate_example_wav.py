"""
generate_example_wav.py — creates ONE small synthetic example WAV file so the
pipeline can be demonstrated end-to-end without requiring an externally
supplied recording.

IMPORTANT: this is a synthetic test signal (a vowel-like harmonic tone with a
vibrato-style pitch wobble, background hiss, and silence padding) — it is
NOT a real human speech recording. It exists only to prove the pipeline runs
correctly on an actual WAV file on disk. Validating the pipeline against real
recorded speech (and, later, real vs. cloned samples) is a separate step
tracked in PROGRESS.md.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_wav", "example_synthetic_vowel.wav"
)


def generate(path: str = OUTPUT_PATH) -> str:
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    base_f0 = 140.0
    vibrato = 4.0 * np.sin(2 * np.pi * 5.0 * t)  # slight natural-sounding pitch wobble
    instantaneous_f0 = base_f0 + vibrato
    phase = 2 * np.pi * np.cumsum(instantaneous_f0) / sr

    voice = (
        1.00 * np.sin(phase)
        + 0.45 * np.sin(2 * phase)
        + 0.20 * np.sin(3 * phase)
        + 0.10 * np.sin(4 * phase)
    )

    envelope = np.clip(np.sin(np.pi * (t / duration)), 0.05, 1.0)
    voice = voice * envelope
    voice = voice / np.max(np.abs(voice)) * 0.85

    rng = np.random.default_rng(7)
    hiss = rng.normal(0, 0.01, size=voice.shape)
    signal = voice + hiss

    pad = np.zeros(int(sr * 0.4))
    full = np.concatenate([pad, signal, pad]).astype(np.float64)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, full, sr, subtype="PCM_16")
    return path


if __name__ == "__main__":
    out = generate()
    print(f"Synthetic example WAV written to: {out}")
