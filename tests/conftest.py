"""
conftest.py — shared pytest fixtures.

All fixtures generate small SYNTHETIC WAV files on the fly (sine waves / noise),
written to a pytest tmp_path. This keeps tests fully self-contained and
reproducible without depending on any external/real audio asset. These synthetic
signals are deliberately simple and are NOT a substitute for validating against
real speech recordings — that validation happens separately, on real files
(see PROGRESS.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def _write_wav(path, audio, sr, subtype="PCM_16"):
    sf.write(str(path), audio, sr, subtype=subtype)
    return str(path)


@pytest.fixture
def speech_like_wav(tmp_path):
    """
    A synthetic 'speech-like' mono WAV: a fundamental sine (~150 Hz) plus two
    harmonics with a slowly varying amplitude envelope, padded with silence at
    start and end, plus a small amount of noise. Sample rate: 22050 Hz (an
    intentionally non-target rate, to exercise resampling).
    """
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    f0 = 150.0
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))  # slow amplitude variation
    voice = (
        1.0 * np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    voice = voice * envelope
    voice = voice / np.max(np.abs(voice)) * 0.8

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=voice.shape)
    signal = voice + noise

    silence_pad = np.zeros(int(sr * 0.3))
    full = np.concatenate([silence_pad, signal, silence_pad])

    path = tmp_path / "speech_like.wav"
    return _write_wav(path, full, sr)


@pytest.fixture
def silent_wav(tmp_path):
    """A WAV that is pure digital silence (all zeros)."""
    sr = 16000
    audio = np.zeros(sr * 1)  # 1 second of silence
    path = tmp_path / "silent.wav"
    return _write_wav(path, audio, sr)


@pytest.fixture
def clipped_wav(tmp_path):
    """A sine wave hard-clipped at full scale, to exercise clipping detection."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 2.0 * np.sin(2 * np.pi * 200 * t)  # amplitude > 1 before clipping
    clipped = np.clip(tone, -1.0, 1.0)
    path = tmp_path / "clipped.wav"
    return _write_wav(path, clipped, sr)


@pytest.fixture
def stereo_wav(tmp_path):
    """A short stereo WAV (identical sine in both channels) for mono-downmix tests."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([tone, tone], axis=1)
    path = tmp_path / "stereo.wav"
    return _write_wav(path, stereo, sr)


@pytest.fixture
def tiny_wav(tmp_path):
    """An extremely short WAV (a handful of samples) to exercise 'too short' guards."""
    sr = 16000
    audio = np.array([0.1, -0.1, 0.05, -0.05, 0.0], dtype=np.float64)
    path = tmp_path / "tiny.wav"
    return _write_wav(path, audio, sr)
