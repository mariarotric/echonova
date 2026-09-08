"""
pulse_detection.py — Vocal-cycle candidate detection for EchoNova.

This module detects candidate excitation pulses from an estimated
voice-source signal.

It reports measurements only. It does not classify speech as human
or synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import find_peaks


DEFAULT_MIN_F0_HZ = 70.0
DEFAULT_MAX_F0_HZ = 400.0
DEFAULT_PROMINENCE_RATIO = 0.15
INTERVAL_TOLERANCE = 1.15


@dataclass
class PulseDetectionResult:
    pulse_samples: np.ndarray
    pulse_times_seconds: np.ndarray
    pulse_amplitudes: np.ndarray
    inter_pulse_intervals_seconds: np.ndarray
    estimated_f0_hz: np.ndarray
    valid_interval_mask: np.ndarray
    sample_rate: int
    pulses_detected: int
    valid_intervals: int
    reliability: float | None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_rate_hz": self.sample_rate,
            "pulses_detected": self.pulses_detected,
            "valid_intervals": self.valid_intervals,
            "pulse_times_seconds": [
                round(float(x), 6)
                for x in self.pulse_times_seconds
            ],
            "pulse_amplitudes": [
                round(float(x), 6)
                for x in self.pulse_amplitudes
            ],
            "inter_pulse_intervals_seconds": [
                round(float(x), 6)
                for x in self.inter_pulse_intervals_seconds
            ],
            "estimated_f0_hz": [
                None if not np.isfinite(x) else round(float(x), 3)
                for x in self.estimated_f0_hz
            ],
            "reliability": (
                None
                if self.reliability is None
                else round(float(self.reliability), 4)
            ),
            "warnings": self.warnings,
        }


def _empty_result(
    sr: int,
    warning: str,
    reliability: float | None = 0.0,
) -> PulseDetectionResult:
    """Create a consistent empty detection result."""

    empty_float = np.array([], dtype=np.float64)
    empty_int = np.array([], dtype=np.int64)
    empty_bool = np.array([], dtype=bool)

    return PulseDetectionResult(
        pulse_samples=empty_int,
        pulse_times_seconds=empty_float,
        pulse_amplitudes=empty_float,
        inter_pulse_intervals_seconds=empty_float,
        estimated_f0_hz=empty_float,
        valid_interval_mask=empty_bool,
        sample_rate=sr,
        pulses_detected=0,
        valid_intervals=0,
        reliability=reliability,
        warnings=[warning],
    )


def detect_vocal_pulses(
    source_signal: np.ndarray,
    sr: int,
    min_f0_hz: float = DEFAULT_MIN_F0_HZ,
    max_f0_hz: float = DEFAULT_MAX_F0_HZ,
    min_prominence_ratio: float = DEFAULT_PROMINENCE_RATIO,
) -> PulseDetectionResult:
    """
    Detect candidate vocal-source pulses.

    The detector:
        1. Cleans non-finite samples.
        2. Uses positive excitation polarity.
        3. Enforces a minimum pulse spacing.
        4. Computes inter-pulse intervals.
        5. Marks intervals compatible with the configured F0 range.

    Large gaps between voiced regions are therefore not treated as
    valid vocal-cycle intervals.
    """

    if sr <= 0:
        raise ValueError(f"Invalid sample rate: {sr}")

    if min_f0_hz <= 0 or max_f0_hz <= 0:
        raise ValueError("F0 limits must be positive.")

    if min_f0_hz >= max_f0_hz:
        raise ValueError(
            "min_f0_hz must be smaller than max_f0_hz."
        )

    source_signal = np.asarray(
        source_signal,
        dtype=np.float64,
    )

    if source_signal.ndim != 1:
        raise ValueError(
            "detect_vocal_pulses expects a 1-D source signal."
        )

    if source_signal.size == 0:
        return _empty_result(
            sr,
            "Source signal is empty; no pulses detected.",
            None,
        )

    signal = np.nan_to_num(
        source_signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    peak = float(
        np.max(np.abs(signal))
    )

    if peak <= 1e-12:
        return _empty_result(
            sr,
            "Source signal has negligible amplitude; no pulses detected.",
        )

    # Use one excitation polarity.
    #
    # Using abs(signal) would allow positive and negative excursions
    # from the same residual cycle to become separate candidates.
    detection_signal = np.maximum(
        signal,
        0.0,
    )

    detection_peak = float(
        np.max(detection_signal)
    )

    if detection_peak <= 1e-12:
        return _empty_result(
            sr,
            "No positive excitation energy was available for pulse detection.",
        )

    # Minimum distance between candidate pulses.
    min_distance_samples = max(
        1,
        int(sr / max_f0_hz),
    )

    # Robust amplitude estimate prevents one extreme spike from
    # completely determining the prominence threshold.
    median_level = float(
        np.median(detection_signal)
    )

    mad = float(
        np.median(
            np.abs(
                detection_signal - median_level
            )
        )
    )

    robust_level = (
        median_level + 3.0 * mad
    )

    prominence = max(
        detection_peak * min_prominence_ratio,
        robust_level * 0.5,
        1e-8,
    )

    pulse_samples, _ = find_peaks(
        detection_signal,
        distance=min_distance_samples,
        prominence=prominence,
    )

    pulse_samples = pulse_samples.astype(
        np.int64
    )

    if pulse_samples.size == 0:
        return _empty_result(
            sr,
            "No candidate vocal pulses were detected.",
        )

    pulse_times = (
        pulse_samples.astype(np.float64)
        / float(sr)
    )

    pulse_amplitudes = detection_signal[
        pulse_samples
    ].astype(np.float64)

    # ---------------------------------------------------------
    # Interval validation
    # ---------------------------------------------------------

    if pulse_samples.size < 2:

        return PulseDetectionResult(
            pulse_samples=pulse_samples,
            pulse_times_seconds=pulse_times,
            pulse_amplitudes=pulse_amplitudes,
            inter_pulse_intervals_seconds=np.array(
                [],
                dtype=np.float64,
            ),
            estimated_f0_hz=np.array(
                [],
                dtype=np.float64,
            ),
            valid_interval_mask=np.array(
                [],
                dtype=bool,
            ),
            sample_rate=sr,
            pulses_detected=int(
                pulse_samples.size
            ),
            valid_intervals=0,
            reliability=0.0,
            warnings=[
                "Only one pulse was detected; "
                "cycle-to-cycle dynamics cannot be estimated."
            ],
        )

    intervals = np.diff(
        pulse_times
    )

    theoretical_min = (
        1.0 / max_f0_hz
    )

    theoretical_max = (
        1.0 / min_f0_hz
    )

    min_interval = (
        theoretical_min
        / INTERVAL_TOLERANCE
    )

    max_interval = (
        theoretical_max
        * INTERVAL_TOLERANCE
    )

    valid_mask = (
        (intervals >= min_interval)
        & (intervals <= max_interval)
        & np.isfinite(intervals)
    )

    f0_values = np.full(
        intervals.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        valid_mask
        & (intervals > 0)
    )

    f0_values[valid] = (
        1.0 / intervals[valid]
    )

    valid_count = int(
        np.sum(valid_mask)
    )

    interval_reliability = (
        valid_count
        / float(intervals.size)
    )

    cycle_factor = float(
        np.clip(
            valid_count / 20.0,
            0.0,
            1.0,
        )
    )

    reliability = (
        0.7 * interval_reliability
        + 0.3 * cycle_factor
    )

    warnings: List[str] = []

    if valid_count == 0:
        warnings.append(
            "Detected pulses did not form plausible "
            "vocal-cycle intervals under the configured F0 range."
        )

    return PulseDetectionResult(
        pulse_samples=pulse_samples,
        pulse_times_seconds=pulse_times,
        pulse_amplitudes=pulse_amplitudes,
        inter_pulse_intervals_seconds=intervals,
        estimated_f0_hz=f0_values,
        valid_interval_mask=valid_mask,
        sample_rate=sr,
        pulses_detected=int(
            pulse_samples.size
        ),
        valid_intervals=valid_count,
        reliability=float(
            np.clip(
                reliability,
                0.0,
                1.0,
            )
        ),
        warnings=warnings,
    )