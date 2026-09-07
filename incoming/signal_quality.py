"""
signal_quality.py — Stage [2] Signal Quality Assessment (ARCHITECTURE.md).

Per DECISIONS.md D-003, this stage's output is what later milestones will use to
weight the voice-source (glottal) branch during fusion. Milestone 1 only computes
and reports the metrics — no fusion/weighting logic exists yet.

Every metric here is computed directly from the waveform. Where a metric cannot be
reliably computed (e.g. the signal is silent, or too short for frame-based
estimation), the function returns None and the caller is expected to add a warning
— we never substitute a default/fabricated number.

Documented heuristics (all are deliberate engineering choices, not measured facts,
and are recorded in the JSON output's `parameters` so the report is self-describing):

    CLIPPING_THRESHOLD = 0.99
        A sample is counted as "clipped" if |sample| >= 0.99 (relative to full
        scale, where full scale is 1.0 for float PCM). This threshold is applied
        to a caller-supplied reference array — see note in `compute_signal_quality`
        about WHICH array should be used for clipping detection.

    FRAME_LENGTH_SAMPLES / HOP_LENGTH_SAMPLES
        Frame-based metrics (silence ratio, SNR estimate) use 25 ms frames with a
        10 ms hop, standard short-term speech analysis windowing, computed from
        the actual sample rate at call time (not hardcoded in samples).

    SILENCE_FRAME_DB_THRESHOLD = -40 dBFS
        A frame is considered "silent" if its RMS level is below -40 dBFS. This is
        a conventional speech/silence heuristic threshold, not a measured property.

    SNR_NOISE_PERCENTILE = 10, SNR_SIGNAL_PERCENTILE = 95
        The SNR estimate treats the 10th percentile of per-frame RMS energy as an
        approximation of the noise floor, and the 95th percentile as an
        approximation of active-signal energy. This is a coarse, documented proxy
        — NOT a lab-grade SNR measurement (which would require a known noise-only
        reference segment). It is reported as "snr_estimate_db" to make this
        explicit, and a `snr_method` string is included in the JSON explaining it.

        KNOWN LIMITATION: this proxy only works when the recording actually
        contains some quiet/pause frames for the low percentile to sample as
        "noise floor". A continuous, gap-free tone or utterance with no pauses
        gives the proxy nothing to measure against and will report a low/
        near-zero SNR even if the signal is genuinely clean — this is a known,
        tested limitation (see tests/test_signal_quality.py), not a bug. Real
        speech recordings normally contain enough pauses for this to be a
        reasonable proxy; this should be re-evaluated against real voice
        samples once available (see PROGRESS.md).

    BANDWIDTH_ENERGY_FRACTION = 0.95
        The bandwidth proxy is the frequency below which 95% of the signal's total
        spectral energy (via Welch's power spectral density estimate) is
        contained. This approximates the "effective occupied bandwidth" and acts
        as a proxy for narrowband/telephony-like degradation in later work
        (DECISIONS.md D-004) — it is not a formal telecom bandwidth measurement.

    QUALITY_SCORE weights
        The overall quality score (0-1) is a documented weighted combination of
        normalized clipping, silence ratio, SNR estimate, and bandwidth proxy.
        If any input component could not be computed, it is excluded from the
        weighted sum and the weights of the remaining components are renormalized
        — the score is never computed from a partially fabricated component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import welch

CLIPPING_THRESHOLD = 0.99
FRAME_MS = 25
HOP_MS = 10
SILENCE_FRAME_DB_THRESHOLD = -40.0
SNR_NOISE_PERCENTILE = 10
SNR_SIGNAL_PERCENTILE = 95
BANDWIDTH_ENERGY_FRACTION = 0.95
MIN_SAMPLES_FOR_FRAME_ANALYSIS = 400  # ~25ms at 16kHz; below this, frame stats are unreliable

# Weights for the composite quality score, applied only to components that were
# successfully computed (see _combine_quality_score).
QUALITY_SCORE_WEIGHTS = {
    "snr": 0.4,
    "clipping": 0.25,
    "silence_ratio": 0.15,
    "bandwidth": 0.20,
}


@dataclass
class SignalQualityResult:
    duration_seconds: Optional[float]
    rms_energy: Optional[float]
    peak_amplitude: Optional[float]
    clipping_percentage: Optional[float]
    silence_ratio: Optional[float]
    snr_estimate_db: Optional[float]
    snr_method: str
    estimated_bandwidth_hz: Optional[float]
    nyquist_hz: Optional[float]
    quality_score: Optional[float]
    parameters: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "duration_seconds": _round_or_none(self.duration_seconds, 6),
            "rms_energy": _round_or_none(self.rms_energy, 6),
            "peak_amplitude": _round_or_none(self.peak_amplitude, 6),
            "clipping_percentage": _round_or_none(self.clipping_percentage, 4),
            "silence_ratio": _round_or_none(self.silence_ratio, 4),
            "snr_estimate_db": _round_or_none(self.snr_estimate_db, 2),
            "snr_method": self.snr_method,
            "estimated_bandwidth_hz": _round_or_none(self.estimated_bandwidth_hz, 1),
            "nyquist_hz": self.nyquist_hz,
            "quality_score": _round_or_none(self.quality_score, 4),
            "parameters": self.parameters,
        }


def _round_or_none(value: Optional[float], ndigits: int) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def compute_rms(audio: np.ndarray) -> Optional[float]:
    if audio.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(audio))))


def compute_peak_amplitude(audio: np.ndarray) -> Optional[float]:
    if audio.size == 0:
        return None
    return float(np.max(np.abs(audio)))


def compute_clipping_percentage(
    audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD
) -> Optional[float]:
    """
    Percentage of samples whose absolute amplitude is >= threshold.
    NOTE: this should be measured on audio that has NOT been rescaled by
    normalization (normalization can shrink an originally-clipped peak below
    the threshold and hide real clipping). The pipeline passes the
    pre-normalization array in for this specific metric — see pipeline.py.
    """
    if audio.size == 0:
        return None
    clipped = np.sum(np.abs(audio) >= threshold)
    return float(clipped) / float(audio.size) * 100.0


def _frame_rms_db(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    frame_length = max(int(sr * FRAME_MS / 1000), 1)
    hop_length = max(int(sr * HOP_MS / 1000), 1)
    if audio.size < frame_length:
        return None

    n_frames = 1 + (audio.size - frame_length) // hop_length
    if n_frames < 1:
        return None

    frame_rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        rms = np.sqrt(np.mean(np.square(frame))) if frame.size else 0.0
        frame_rms[i] = rms

    with np.errstate(divide="ignore"):
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    return frame_db


def compute_silence_ratio(
    audio: np.ndarray, sr: int, db_threshold: float = SILENCE_FRAME_DB_THRESHOLD
) -> Optional[float]:
    frame_db = _frame_rms_db(audio, sr)
    if frame_db is None or frame_db.size == 0:
        return None
    silent_frames = np.sum(frame_db < db_threshold)
    return float(silent_frames) / float(frame_db.size)


def estimate_snr_db(
    audio: np.ndarray,
    sr: int,
    noise_percentile: float = SNR_NOISE_PERCENTILE,
    signal_percentile: float = SNR_SIGNAL_PERCENTILE,
) -> Optional[float]:
    """
    Coarse SNR proxy: difference (in dB) between a high percentile and a low
    percentile of per-frame RMS energy. This assumes the quietest frames are
    dominated by noise/background and the loudest frames are dominated by
    active speech — a reasonable approximation for many recordings, but NOT
    a substitute for a true noise-reference SNR measurement. See module
    docstring.
    """
    frame_db = _frame_rms_db(audio, sr)
    if frame_db is None or frame_db.size < 5:
        return None
    noise_floor = np.percentile(frame_db, noise_percentile)
    signal_level = np.percentile(frame_db, signal_percentile)
    snr = signal_level - noise_floor
    if not np.isfinite(snr):
        return None
    return float(snr)


def estimate_bandwidth_hz(
    audio: np.ndarray, sr: int, energy_fraction: float = BANDWIDTH_ENERGY_FRACTION
) -> Optional[float]:
    """
    Frequency below which `energy_fraction` of total spectral energy (via Welch
    PSD) is contained. Proxy for effective occupied bandwidth / narrowband
    degradation. Returns None if the signal is too short for a meaningful PSD
    estimate.
    """
    if audio.size < 256:
        return None
    nperseg = min(2048, audio.size)
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg)
    total_energy = np.sum(psd)
    if total_energy <= 0 or not np.isfinite(total_energy):
        return None
    cumulative = np.cumsum(psd) / total_energy
    idx = np.searchsorted(cumulative, energy_fraction)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _combine_quality_score(
    snr_db: Optional[float],
    clipping_pct: Optional[float],
    silence_ratio: Optional[float],
    bandwidth_hz: Optional[float],
    nyquist_hz: Optional[float],
) -> Optional[float]:
    """
    Combine available (non-None) normalized components into a single 0-1 score
    using QUALITY_SCORE_WEIGHTS, renormalized over whichever components were
    actually computed. Returns None only if NO component could be computed.
    """
    components: dict[str, float] = {}

    if snr_db is not None:
        # Normalize: 0 dB or below -> 0.0, 40 dB or above -> 1.0 (documented clamp range)
        components["snr"] = float(np.clip(snr_db / 40.0, 0.0, 1.0))

    if clipping_pct is not None:
        # 0% clipping -> 1.0, 5% or more clipped samples -> 0.0
        components["clipping"] = float(np.clip(1.0 - (clipping_pct / 5.0), 0.0, 1.0))

    if silence_ratio is not None:
        # Some silence is normal in real speech; only penalize once silence
        # ratio exceeds 70% of all frames.
        components["silence_ratio"] = float(np.clip(1.0 - max(0.0, silence_ratio - 0.7) / 0.3, 0.0, 1.0))

    if bandwidth_hz is not None and nyquist_hz:
        components["bandwidth"] = float(np.clip(bandwidth_hz / nyquist_hz, 0.0, 1.0))

    if not components:
        return None

    total_weight = sum(QUALITY_SCORE_WEIGHTS[k] for k in components)
    score = sum(QUALITY_SCORE_WEIGHTS[k] * v for k, v in components.items()) / total_weight
    return float(np.clip(score, 0.0, 1.0))


def compute_signal_quality(
    audio: np.ndarray,
    sr: int,
    clipping_reference_audio: Optional[np.ndarray] = None,
) -> SignalQualityResult:
    """
    Compute the full Milestone 1 signal-quality report for `audio` at sample
    rate `sr`.

    Args:
        audio: the (typically preprocessed) waveform used for RMS, peak,
            silence ratio, SNR estimate, and bandwidth proxy.
        clipping_reference_audio: if provided, clipping percentage is computed
            on THIS array instead of `audio`. Use the original, pre-normalization
            waveform here so normalization cannot mask real clipping. If not
            provided, clipping is computed on `audio` itself (with a warning).
    """
    warnings: List[str] = []

    duration = audio.shape[0] / sr if sr else None
    rms = compute_rms(audio)
    peak = compute_peak_amplitude(audio)

    if clipping_reference_audio is None:
        clipping_reference_audio = audio
        warnings.append(
            "No pre-normalization reference supplied for clipping detection; "
            "measured clipping on the provided (possibly normalized) array, "
            "which may under-report clipping present in the original recording."
        )
    clipping_pct = compute_clipping_percentage(clipping_reference_audio)

    if audio.size < MIN_SAMPLES_FOR_FRAME_ANALYSIS:
        warnings.append(
            "Audio too short for reliable frame-based analysis "
            f"(< {MIN_SAMPLES_FOR_FRAME_ANALYSIS} samples); "
            "silence ratio and SNR estimate not computed."
        )
        silence_ratio = None
        snr_db = None
    else:
        silence_ratio = compute_silence_ratio(audio, sr)
        snr_db = estimate_snr_db(audio, sr)
        if snr_db is None:
            warnings.append("SNR estimate could not be computed for this signal.")

    bandwidth_hz = estimate_bandwidth_hz(audio, sr)
    nyquist_hz = sr / 2.0 if sr else None
    if bandwidth_hz is None:
        warnings.append("Bandwidth proxy could not be computed (signal too short).")

    quality_score = _combine_quality_score(
        snr_db, clipping_pct, silence_ratio, bandwidth_hz, nyquist_hz
    )
    if quality_score is None:
        warnings.append(
            "Overall quality score could not be computed: no underlying "
            "component metric was available."
        )

    return SignalQualityResult(
        duration_seconds=duration,
        rms_energy=rms,
        peak_amplitude=peak,
        clipping_percentage=clipping_pct,
        silence_ratio=silence_ratio,
        snr_estimate_db=snr_db,
        snr_method=(
            "Percentile-based proxy: difference between the "
            f"{SNR_SIGNAL_PERCENTILE}th and {SNR_NOISE_PERCENTILE}th percentile "
            "of per-frame RMS energy (25ms frames, 10ms hop). Not a substitute "
            "for a true noise-reference SNR measurement."
        ),
        estimated_bandwidth_hz=bandwidth_hz,
        nyquist_hz=nyquist_hz,
        quality_score=quality_score,
        parameters={
            "clipping_threshold": CLIPPING_THRESHOLD,
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "silence_frame_db_threshold": SILENCE_FRAME_DB_THRESHOLD,
            "snr_noise_percentile": SNR_NOISE_PERCENTILE,
            "snr_signal_percentile": SNR_SIGNAL_PERCENTILE,
            "bandwidth_energy_fraction": BANDWIDTH_ENERGY_FRACTION,
            "quality_score_weights": QUALITY_SCORE_WEIGHTS,
        },
        warnings=warnings,
    )
