"""
source_extraction.py — Voice-Source Analysis (EchoNova).

Purpose:
    Estimate the underlying vocal-source signal from speech using a
    source-filter analysis.

This is the first stage of EchoNova's Voice-Source Temporal Dynamics branch.

Pipeline:
    speech waveform
        ↓
    pre-emphasis
        ↓
    short-time LPC analysis
        ↓
    inverse filtering
        ↓
    estimated source/residual signal

Important:
    Source extraction is an analysis aid, not a perfect reconstruction of
    the physical glottal flow. The resulting signal is therefore described
    as an estimated vocal-source / excitation signal.

The output is intentionally transparent so later stages can inspect the
actual source waveform rather than receiving a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import lfilter


# Speech-analysis defaults.
DEFAULT_LPC_ORDER = 16
DEFAULT_FRAME_MS = 25
DEFAULT_HOP_MS = 10
DEFAULT_PREEMPHASIS = 0.97

MIN_FRAME_SAMPLES = 200


@dataclass
class SourceExtractionResult:
    """Result of vocal-source estimation."""

    source_signal: np.ndarray
    sample_rate: int
    lpc_order: int
    frame_length_samples: int
    hop_length_samples: int
    preemphasis_coefficient: float
    frames_processed: int
    valid_frames: int
    reliability: Optional[float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_rate_hz": self.sample_rate,
            "lpc_order": self.lpc_order,
            "frame_length_samples": self.frame_length_samples,
            "hop_length_samples": self.hop_length_samples,
            "preemphasis_coefficient": self.preemphasis_coefficient,
            "frames_processed": self.frames_processed,
            "valid_frames": self.valid_frames,
            "reliability": (
                None
                if self.reliability is None
                else round(float(self.reliability), 4)
            ),
            "warnings": self.warnings,
        }


def _autocorrelation_lpc(
    frame: np.ndarray,
    order: int,
) -> Optional[np.ndarray]:
    """
    Estimate LPC coefficients using autocorrelation and Levinson-style
    recursion.

    Returns coefficients [1, a1, ..., ap] for the prediction filter:

        A(z) = 1 + a1*z^-1 + ... + ap*z^-p

    Returns None when the frame does not contain enough usable energy.
    """
    frame = np.asarray(frame, dtype=np.float64)

    if frame.size <= order:
        return None

    frame = frame - np.mean(frame)

    energy = float(np.dot(frame, frame))
    if not np.isfinite(energy) or energy <= 1e-12:
        return None

    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[frame.size - 1 : frame.size + order]

    if autocorr.size < order + 1 or autocorr[0] <= 1e-12:
        return None

    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[0] = 1.0

    error = autocorr[0]

    for i in range(1, order + 1):
        if error <= 1e-12:
            return None

        reflection = autocorr[i]

        if i > 1:
            reflection -= np.dot(
                coefficients[1:i],
                autocorr[i - 1 : 0 : -1],
            )

        reflection /= error
        reflection = float(np.clip(reflection, -0.999, 0.999))

        previous = coefficients.copy()

        coefficients[i] = reflection

        if i > 1:
            coefficients[1:i] = (
                previous[1:i] - reflection * previous[i - 1 : 0 : -1]
            )

        error *= 1.0 - reflection * reflection

    if not np.all(np.isfinite(coefficients)):
        return None

    return coefficients


def estimate_source_signal(
    audio: np.ndarray,
    sr: int,
    lpc_order: int = DEFAULT_LPC_ORDER,
    frame_ms: float = DEFAULT_FRAME_MS,
    hop_ms: float = DEFAULT_HOP_MS,
    preemphasis: float = DEFAULT_PREEMPHASIS,
) -> SourceExtractionResult:
    """
    Estimate the vocal-source/excitation signal using frame-wise LPC
    inverse filtering.

    The residual is produced by applying the LPC prediction filter to
    the speech frame:

        e[n] = A(z) * x[n]

    where x[n] is the speech signal and e[n] is the estimated excitation.

    Frames are overlap-added to produce a continuous source estimate.
    """
    warnings: List[str] = []

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1:
        raise ValueError("estimate_source_signal expects a mono 1-D waveform.")

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    if audio.size == 0:
        warnings.append("Audio is empty; source extraction was not performed.")

        return SourceExtractionResult(
            source_signal=np.array([], dtype=np.float64),
            sample_rate=sr,
            lpc_order=lpc_order,
            frame_length_samples=0,
            hop_length_samples=0,
            preemphasis_coefficient=preemphasis,
            frames_processed=0,
            valid_frames=0,
            reliability=None,
            warnings=warnings,
        )

    frame_length = max(int(sr * frame_ms / 1000.0), MIN_FRAME_SAMPLES)
    hop_length = max(int(sr * hop_ms / 1000.0), 1)

    if audio.size < frame_length:
        warnings.append(
            "Audio is shorter than one analysis frame; "
            "source extraction was not performed."
        )

        return SourceExtractionResult(
            source_signal=np.zeros_like(audio),
            sample_rate=sr,
            lpc_order=lpc_order,
            frame_length_samples=frame_length,
            hop_length_samples=hop_length,
            preemphasis_coefficient=preemphasis,
            frames_processed=0,
            valid_frames=0,
            reliability=None,
            warnings=warnings,
        )

    # Pre-emphasis makes the speech spectrum more suitable for LPC analysis.
    emphasized = lfilter(
        [1.0, -preemphasis],
        [1.0],
        audio,
    )

    n_frames = 1 + (audio.size - frame_length) // hop_length

    source_sum = np.zeros(audio.size, dtype=np.float64)
    source_count = np.zeros(audio.size, dtype=np.float64)

    window = np.hanning(frame_length)

    valid_frames = 0

    for frame_index in range(n_frames):
        start = frame_index * hop_length
        end = start + frame_length

        frame = emphasized[start:end]

        if frame.size != frame_length:
            continue

        # Windowing reduces discontinuities at frame boundaries.
        windowed = frame * window

        coefficients = _autocorrelation_lpc(
            windowed,
            order=lpc_order,
        )

        if coefficients is None:
            continue

        residual = lfilter(
            coefficients,
            [1.0],
            windowed,
        )

        if not np.all(np.isfinite(residual)):
            continue

        source_sum[start:end] += residual
        source_count[start:end] += window

        valid_frames += 1

    source_signal = np.zeros_like(audio)

    valid = source_count > 1e-12
    source_signal[valid] = source_sum[valid] / source_count[valid]

    if valid_frames == 0:
        warnings.append(
            "No valid LPC frames were available for source estimation."
        )
        reliability = 0.0
    else:
        reliability = valid_frames / float(n_frames)

    return SourceExtractionResult(
        source_signal=source_signal,
        sample_rate=sr,
        lpc_order=lpc_order,
        frame_length_samples=frame_length,
        hop_length_samples=hop_length,
        preemphasis_coefficient=preemphasis,
        frames_processed=n_frames,
        valid_frames=valid_frames,
        reliability=float(np.clip(reliability, 0.0, 1.0)),
        warnings=warnings,
    )