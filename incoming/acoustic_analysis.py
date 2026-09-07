"""
acoustic_analysis.py — Stage [3] Acoustic Analysis (ARCHITECTURE.md).

Computes standard acoustic-level features from the preprocessed waveform:
    - F0 (pitch) statistics, restricted to frames the pitch tracker itself
      considers voiced.
    - MFCC statistics.
    - Spectral centroid.
    - Spectral bandwidth.
    - Zero-crossing rate.

This module produces evidence for later fusion (future milestone). It does NOT
compute any authenticity score, and does NOT perform glottal/voice-source
analysis (that is Milestone 3+ per PROGRESS.md).

Documented heuristics:

    F0_MIN_HZ = 65, F0_MAX_HZ = 400
        Search range passed to librosa.pyin. Covers typical human speech
        fundamental frequency for both lower and higher voices with some
        margin. This is a conventional speech-analysis range, not derived
        from the specific input file.

    N_MFCC = 13
        Standard number of MFCC coefficients for speech analysis.

    MIN_SAMPLES_FOR_SPECTRAL_ANALYSIS
        Below this many samples, frame-based spectral features (which need at
        least one full analysis frame) are not computed; the function returns
        None for those fields plus a warning instead of producing a value from
        an inadequately short / zero-padded window.

Reliability handling:
    - If librosa.pyin finds no voiced frames at all (e.g. non-speech audio,
      pure noise, or a signal too degraded to track pitch), all F0 statistics
      are returned as None with an explanatory warning — never a fabricated
      pitch value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import librosa

F0_MIN_HZ = 65.0
F0_MAX_HZ = 400.0
N_MFCC = 13
FRAME_LENGTH = 2048
HOP_LENGTH = 512
# Need at least one full FFT frame to compute spectral features meaningfully.
MIN_SAMPLES_FOR_SPECTRAL_ANALYSIS = FRAME_LENGTH


@dataclass
class AcousticAnalysisResult:
    f0_mean_hz: Optional[float]
    f0_median_hz: Optional[float]
    f0_std_hz: Optional[float]
    f0_min_hz: Optional[float]
    f0_max_hz: Optional[float]
    voiced_frame_ratio: Optional[float]
    mfcc_mean: Optional[List[float]]
    mfcc_std: Optional[List[float]]
    spectral_centroid_mean_hz: Optional[float]
    spectral_centroid_std_hz: Optional[float]
    spectral_bandwidth_mean_hz: Optional[float]
    spectral_bandwidth_std_hz: Optional[float]
    zero_crossing_rate_mean: Optional[float]
    zero_crossing_rate_std: Optional[float]
    parameters: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "f0": {
                "mean_hz": _r(self.f0_mean_hz, 2),
                "median_hz": _r(self.f0_median_hz, 2),
                "std_hz": _r(self.f0_std_hz, 2),
                "min_hz": _r(self.f0_min_hz, 2),
                "max_hz": _r(self.f0_max_hz, 2),
                "voiced_frame_ratio": _r(self.voiced_frame_ratio, 4),
                "search_range_hz": [self.parameters.get("f0_min_hz"), self.parameters.get("f0_max_hz")],
            },
            "mfcc": {
                "n_mfcc": self.parameters.get("n_mfcc"),
                "mean_per_coefficient": self.mfcc_mean,
                "std_per_coefficient": self.mfcc_std,
            },
            "spectral_centroid": {
                "mean_hz": _r(self.spectral_centroid_mean_hz, 2),
                "std_hz": _r(self.spectral_centroid_std_hz, 2),
            },
            "spectral_bandwidth": {
                "mean_hz": _r(self.spectral_bandwidth_mean_hz, 2),
                "std_hz": _r(self.spectral_bandwidth_std_hz, 2),
            },
            "zero_crossing_rate": {
                "mean": _r(self.zero_crossing_rate_mean, 6),
                "std": _r(self.zero_crossing_rate_std, 6),
            },
        }


def _r(value: Optional[float], ndigits: int) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def compute_f0_statistics(
    audio: np.ndarray,
    sr: int,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
) -> dict:
    """
    Estimate F0 using librosa.pyin (probabilistic YIN) and summarize statistics
    over frames the tracker marks as voiced. Returns a dict with keys mean,
    median, std, min, max, voiced_frame_ratio, and 'warning' (str or None).
    All numeric fields are None if no voiced frames were found.
    """
    if audio.size < FRAME_LENGTH:
        return {
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "voiced_frame_ratio": None,
            "warning": "Audio too short for pitch tracking (fewer samples than one analysis frame).",
        }

    try:
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            audio, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "voiced_frame_ratio": None,
            "warning": f"Pitch tracking failed: {exc}",
        }

    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    f0 = np.asarray(f0, dtype=np.float64)
    valid = voiced_flag & np.isfinite(f0)

    voiced_ratio = float(np.sum(voiced_flag)) / float(voiced_flag.size) if voiced_flag.size else None

    if not np.any(valid):
        return {
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "voiced_frame_ratio": voiced_ratio,
            "warning": "No voiced frames detected; pitch estimation unreliable for this input.",
        }

    voiced_f0 = f0[valid]
    return {
        "mean": float(np.mean(voiced_f0)),
        "median": float(np.median(voiced_f0)),
        "std": float(np.std(voiced_f0)),
        "min": float(np.min(voiced_f0)),
        "max": float(np.max(voiced_f0)),
        "voiced_frame_ratio": voiced_ratio,
        "warning": None,
    }


def compute_mfcc_statistics(
    audio: np.ndarray, sr: int, n_mfcc: int = N_MFCC
) -> tuple[Optional[List[float]], Optional[List[float]], Optional[str]]:
    if audio.size < MIN_SAMPLES_FOR_SPECTRAL_ANALYSIS:
        return None, None, "Audio too short to compute MFCCs reliably."
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    return (
        [float(v) for v in np.mean(mfcc, axis=1)],
        [float(v) for v in np.std(mfcc, axis=1)],
        None,
    )


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if audio.size < MIN_SAMPLES_FOR_SPECTRAL_ANALYSIS:
        return None, None, "Audio too short to compute spectral centroid reliably."
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    return float(np.mean(centroid)), float(np.std(centroid)), None


def compute_spectral_bandwidth(audio: np.ndarray, sr: int) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if audio.size < MIN_SAMPLES_FOR_SPECTRAL_ANALYSIS:
        return None, None, "Audio too short to compute spectral bandwidth reliably."
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    return float(np.mean(bandwidth)), float(np.std(bandwidth)), None


def compute_zero_crossing_rate(audio: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[str]]:
    if audio.size < HOP_LENGTH:
        return None, None, "Audio too short to compute zero-crossing rate reliably."
    zcr = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    return float(np.mean(zcr)), float(np.std(zcr)), None


def analyze_acoustics(audio: np.ndarray, sr: int) -> AcousticAnalysisResult:
    warnings: List[str] = []

    f0_stats = compute_f0_statistics(audio, sr)
    if f0_stats["warning"]:
        warnings.append(f0_stats["warning"])

    mfcc_mean, mfcc_std, mfcc_warn = compute_mfcc_statistics(audio, sr)
    if mfcc_warn:
        warnings.append(mfcc_warn)

    centroid_mean, centroid_std, centroid_warn = compute_spectral_centroid(audio, sr)
    if centroid_warn:
        warnings.append(centroid_warn)

    bandwidth_mean, bandwidth_std, bandwidth_warn = compute_spectral_bandwidth(audio, sr)
    if bandwidth_warn:
        warnings.append(bandwidth_warn)

    zcr_mean, zcr_std, zcr_warn = compute_zero_crossing_rate(audio)
    if zcr_warn:
        warnings.append(zcr_warn)

    return AcousticAnalysisResult(
        f0_mean_hz=f0_stats["mean"],
        f0_median_hz=f0_stats["median"],
        f0_std_hz=f0_stats["std"],
        f0_min_hz=f0_stats["min"],
        f0_max_hz=f0_stats["max"],
        voiced_frame_ratio=f0_stats["voiced_frame_ratio"],
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        spectral_centroid_mean_hz=centroid_mean,
        spectral_centroid_std_hz=centroid_std,
        spectral_bandwidth_mean_hz=bandwidth_mean,
        spectral_bandwidth_std_hz=bandwidth_std,
        zero_crossing_rate_mean=zcr_mean,
        zero_crossing_rate_std=zcr_std,
        parameters={
            "f0_min_hz": F0_MIN_HZ,
            "f0_max_hz": F0_MAX_HZ,
            "n_mfcc": N_MFCC,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
        },
        warnings=warnings,
    )
