"""
temporal_features.py — Voice-source temporal dynamics for EchoNova.

Computes robust cycle-to-cycle statistics from detected vocal pulses.

Only physically plausible inter-pulse intervals are used for temporal
statistics. Large gaps between voiced regions are not treated as vocal
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TemporalDynamicsResult:

    pulse_count: int

    mean_interval_seconds: Optional[float]
    std_interval_seconds: Optional[float]
    interval_cv: Optional[float]

    mean_f0_hz: Optional[float]
    std_f0_hz: Optional[float]
    f0_cv: Optional[float]

    mean_pulse_amplitude: Optional[float]
    std_pulse_amplitude: Optional[float]
    amplitude_cv: Optional[float]

    valid_interval_count: int
    temporal_reliability: Optional[float]

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:

        def rounded(value, digits=6):
            if value is None:
                return None
            return round(float(value), digits)

        return {
            "pulse_count": self.pulse_count,
            "mean_interval_seconds": rounded(
                self.mean_interval_seconds
            ),
            "std_interval_seconds": rounded(
                self.std_interval_seconds
            ),
            "interval_cv": rounded(
                self.interval_cv,
                4,
            ),
            "mean_f0_hz": rounded(
                self.mean_f0_hz,
                3,
            ),
            "std_f0_hz": rounded(
                self.std_f0_hz,
                3,
            ),
            "f0_cv": rounded(
                self.f0_cv,
                4,
            ),
            "mean_pulse_amplitude": rounded(
                self.mean_pulse_amplitude
            ),
            "std_pulse_amplitude": rounded(
                self.std_pulse_amplitude
            ),
            "amplitude_cv": rounded(
                self.amplitude_cv,
                4,
            ),
            "valid_interval_count": (
                self.valid_interval_count
            ),
            "temporal_reliability": rounded(
                self.temporal_reliability,
                4,
            ),
            "warnings": self.warnings,
        }


def _coefficient_of_variation(
    values: np.ndarray,
) -> Optional[float]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if values.size < 2:
        return None

    mean = float(np.mean(values))

    if mean <= 1e-12:
        return None

    return float(
        np.std(values) / mean
    )


def compute_temporal_dynamics(
    pulse_times_seconds: np.ndarray,
    pulse_amplitudes: np.ndarray,
    estimated_f0_hz: np.ndarray,
    valid_interval_mask: np.ndarray | None = None,
) -> TemporalDynamicsResult:

    warnings: List[str] = []

    pulse_times = np.asarray(
        pulse_times_seconds,
        dtype=np.float64,
    )

    amplitudes = np.asarray(
        pulse_amplitudes,
        dtype=np.float64,
    )

    f0 = np.asarray(
        estimated_f0_hz,
        dtype=np.float64,
    )

    pulse_count = int(
        pulse_times.size
    )

    # ---------------------------------------------------------
    # Interval dynamics
    # ---------------------------------------------------------

    if pulse_count >= 2:

        intervals = np.diff(
            pulse_times
        )

        if valid_interval_mask is not None:

            mask = np.asarray(
                valid_interval_mask,
                dtype=bool,
            )

            if mask.shape == intervals.shape:
                intervals = intervals[mask]

        valid_intervals = intervals[
            np.isfinite(intervals)
            & (intervals > 0)
        ]

    else:

        valid_intervals = np.array(
            [],
            dtype=np.float64,
        )

    if valid_intervals.size >= 2:

        mean_interval = float(
            np.mean(valid_intervals)
        )

        std_interval = float(
            np.std(valid_intervals)
        )

        interval_cv = (
            _coefficient_of_variation(
                valid_intervals
            )
        )

    else:

        mean_interval = None
        std_interval = None
        interval_cv = None

        warnings.append(
            "Insufficient valid pulse intervals "
            "for reliable interval dynamics."
        )

    # ---------------------------------------------------------
    # F0 dynamics
    # ---------------------------------------------------------

    valid_f0 = f0[
        np.isfinite(f0)
        & (f0 > 0)
    ]

    if valid_f0.size >= 2:

        mean_f0 = float(
            np.mean(valid_f0)
        )

        std_f0 = float(
            np.std(valid_f0)
        )

        f0_cv = (
            _coefficient_of_variation(
                valid_f0
            )
        )

    else:

        mean_f0 = None
        std_f0 = None
        f0_cv = None

        warnings.append(
            "Insufficient valid F0 estimates "
            "for reliable F0 dynamics."
        )

    # ---------------------------------------------------------
    # Amplitude dynamics
    # ---------------------------------------------------------

    valid_amplitudes = amplitudes[
        np.isfinite(amplitudes)
        & (amplitudes > 0)
    ]

    if valid_amplitudes.size >= 2:

        mean_amplitude = float(
            np.mean(valid_amplitudes)
        )

        std_amplitude = float(
            np.std(valid_amplitudes)
        )

        amplitude_cv = (
            _coefficient_of_variation(
                valid_amplitudes
            )
        )

    else:

        mean_amplitude = None
        std_amplitude = None
        amplitude_cv = None

        warnings.append(
            "Insufficient pulse amplitudes "
            "for reliable amplitude dynamics."
        )

    # ---------------------------------------------------------
    # Reliability
    # ---------------------------------------------------------

    if pulse_count < 2:

        reliability = 0.0

    else:

        valid_fraction = (
            valid_intervals.size
            / float(pulse_count - 1)
        )

        count_factor = float(
            np.clip(
                valid_intervals.size / 20.0,
                0.0,
                1.0,
            )
        )

        reliability = (
            0.7 * valid_fraction
            + 0.3 * count_factor
        )

    return TemporalDynamicsResult(

        pulse_count=pulse_count,

        mean_interval_seconds=mean_interval,
        std_interval_seconds=std_interval,
        interval_cv=interval_cv,

        mean_f0_hz=mean_f0,
        std_f0_hz=std_f0,
        f0_cv=f0_cv,

        mean_pulse_amplitude=mean_amplitude,
        std_pulse_amplitude=std_amplitude,
        amplitude_cv=amplitude_cv,

        valid_interval_count=int(
            valid_intervals.size
        ),

        temporal_reliability=float(
            np.clip(
                reliability,
                0.0,
                1.0,
            )
        ),

        warnings=warnings,
    )