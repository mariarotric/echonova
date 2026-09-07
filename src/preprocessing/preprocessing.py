"""
preprocessing.py — Stage [1] Preprocessing (ARCHITECTURE.md).

Responsibilities:
    - Convert multi-channel audio to mono.
    - Resample to a documented target sample rate.
    - Trim leading/trailing silence.
    - Apply safe peak amplitude normalization.
    - Never modify the original file on disk (this module only ever operates on
      an in-memory NumPy array loaded by io_utils.load_audio).

Documented heuristics / defaults (all overridable, all recorded in the returned
`PreprocessingResult.parameters` dict so the JSON report is self-describing):

    TARGET_SAMPLE_RATE_HZ = 16000
        Rationale: 16 kHz is a standard target for speech analysis — it retains
        the frequency range relevant to voice (fundamental frequency and the
        first several formants) while keeping downstream compute cheap. This is
        a documented engineering choice, not a measured property of any file.

    SILENCE_TRIM_TOP_DB = 30
        Rationale: passed to librosa.effects.trim as the threshold (in dB below
        the signal's peak) below which a frame is considered silence. 30 dB is
        librosa's own commonly used default for speech-like material. Recorded
        explicitly here because it materially affects duration-based downstream
        metrics.

    NORMALIZATION_PEAK_TARGET = 0.98
        Rationale: peak-normalize so the loudest sample reaches 0.98 of full
        scale (not 1.0), leaving headroom to avoid introducing new clipping
        during normalization itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import librosa

TARGET_SAMPLE_RATE_HZ = 16000
SILENCE_TRIM_TOP_DB = 30
NORMALIZATION_PEAK_TARGET = 0.98
# Below this peak amplitude, a signal is considered effectively silent/empty and
# normalization is skipped (dividing by a near-zero peak would amplify noise
# to an arbitrary, meaningless degree).
MIN_PEAK_FOR_NORMALIZATION = 1e-6


@dataclass
class PreprocessingResult:
    audio: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    was_resampled: bool
    was_downmixed_to_mono: bool
    silence_trimmed_seconds_start: float
    silence_trimmed_seconds_end: float
    was_normalized: bool
    duration_after_seconds: float
    parameters: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_sample_rate_hz": self.parameters.get("target_sample_rate_hz"),
            "was_resampled": self.was_resampled,
            "was_downmixed_to_mono": self.was_downmixed_to_mono,
            "silence_trim_top_db": self.parameters.get("silence_trim_top_db"),
            "silence_trimmed_seconds_start": round(self.silence_trimmed_seconds_start, 6),
            "silence_trimmed_seconds_end": round(self.silence_trimmed_seconds_end, 6),
            "was_normalized": self.was_normalized,
            "normalization_peak_target": self.parameters.get("normalization_peak_target"),
            "duration_after_seconds": round(self.duration_after_seconds, 6),
        }


def to_mono(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Downmix to mono by averaging channels. If already mono, returns input unchanged.
    Returns (mono_audio, was_downmixed).
    """
    if audio.ndim == 1:
        return audio, False
    if audio.ndim == 2:
        return np.mean(audio, axis=1), True
    raise ValueError(f"Unsupported audio array shape for to_mono: {audio.shape}")


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE_HZ
) -> tuple[np.ndarray, bool]:
    """
    Resample mono audio to target_sr using librosa (high-quality polyphase resampler).
    Returns (resampled_audio, was_resampled).
    """
    if orig_sr == target_sr:
        return audio, False
    if orig_sr <= 0:
        raise ValueError(f"Invalid original sample rate: {orig_sr}")
    resampled = librosa.resample(
        audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr
    )
    return resampled, True


def trim_silence(
    audio: np.ndarray, sr: int, top_db: int = SILENCE_TRIM_TOP_DB
) -> tuple[np.ndarray, float, float]:
    """
    Trim leading/trailing silence using librosa.effects.trim.
    Returns (trimmed_audio, seconds_trimmed_from_start, seconds_trimmed_from_end).

    If the signal is entirely silence (or trimming would remove everything),
    the original audio is returned unchanged rather than producing an empty array,
    since a zero-length signal cannot be meaningfully analyzed downstream.
    """
    if audio.size == 0:
        return audio, 0.0, 0.0

    trimmed, index = librosa.effects.trim(audio, top_db=top_db)
    if trimmed.size == 0:
        # Trimming would remove the entire signal (e.g. pure silence/noise floor).
        # Keep the original rather than returning nothing.
        return audio, 0.0, 0.0

    start_sample, end_sample = index
    seconds_trimmed_start = start_sample / sr
    seconds_trimmed_end = (audio.shape[0] - end_sample) / sr
    return trimmed, seconds_trimmed_start, seconds_trimmed_end


def normalize_amplitude(
    audio: np.ndarray, peak_target: float = NORMALIZATION_PEAK_TARGET
) -> tuple[np.ndarray, bool]:
    """
    Safe peak normalization: scales audio so its maximum absolute sample equals
    `peak_target`. If the signal's peak is at/near zero (effective silence),
    normalization is skipped to avoid dividing by (near) zero and amplifying
    noise to an arbitrary level.
    Returns (normalized_audio, was_normalized).
    """
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak < MIN_PEAK_FOR_NORMALIZATION:
        return audio, False
    scale = peak_target / peak
    return audio * scale, True


def preprocess(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE_HZ,
    silence_trim_top_db: int = SILENCE_TRIM_TOP_DB,
    normalization_peak_target: float = NORMALIZATION_PEAK_TARGET,
) -> PreprocessingResult:
    """
    Run the full Milestone 1 preprocessing chain: mono -> resample -> trim silence
    -> normalize. Operates entirely on the in-memory array; never touches the
    original file.
    """
    warnings: List[str] = []
    original_channels = 1 if audio.ndim == 1 else audio.shape[1]
    original_sample_rate = sample_rate

    mono_audio, was_downmixed = to_mono(audio)

    resampled_audio, was_resampled = resample_audio(
        mono_audio, orig_sr=sample_rate, target_sr=target_sample_rate
    )
    working_sr = target_sample_rate if was_resampled else sample_rate

    if resampled_audio.size == 0:
        warnings.append("Audio is empty after mono conversion; skipping trim/normalize.")
        trimmed_audio = resampled_audio
        trim_start_s, trim_end_s = 0.0, 0.0
        normalized_audio, was_normalized = resampled_audio, False
    else:
        trimmed_audio, trim_start_s, trim_end_s = trim_silence(
            resampled_audio, working_sr, top_db=silence_trim_top_db
        )
        if trimmed_audio.size == 0:
            warnings.append(
                "Silence trimming would have removed the entire signal; "
                "kept untrimmed audio instead."
            )
        normalized_audio, was_normalized = normalize_amplitude(
            trimmed_audio, peak_target=normalization_peak_target
        )
        if not was_normalized:
            warnings.append(
                "Signal peak amplitude is effectively zero; normalization skipped "
                "(audio may be silent or near-silent)."
            )

    duration_after = normalized_audio.shape[0] / working_sr if working_sr else 0.0

    return PreprocessingResult(
        audio=normalized_audio,
        sample_rate=working_sr,
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
        was_resampled=was_resampled,
        was_downmixed_to_mono=was_downmixed,
        silence_trimmed_seconds_start=trim_start_s,
        silence_trimmed_seconds_end=trim_end_s,
        was_normalized=was_normalized,
        duration_after_seconds=duration_after,
        parameters={
            "target_sample_rate_hz": target_sample_rate,
            "silence_trim_top_db": silence_trim_top_db,
            "normalization_peak_target": normalization_peak_target,
        },
        warnings=warnings,
    )
