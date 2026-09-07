"""
io_utils.py — WAV loading and validation.

Responsibilities (Milestone 1, per ARCHITECTURE.md stage [1] Preprocessing input step):
    - Validate that a path exists and is a readable WAV file.
    - Report raw file metadata: sample rate, channel count, duration, sample count,
      bit depth (if available from the container).
    - Load audio samples as a float64 NumPy array WITHOUT modifying the original file.

Design notes:
    - We use `soundfile` (libsndfile) for both metadata inspection and sample loading,
      since it reports subtype/bit-depth information that raw `wave` module handling
      of float/extended formats does not always expose reliably.
    - This module never writes to the input file. The original file is only opened
      for reading.
    - Any inability to determine a value results in `None` plus an entry in the
      returned warnings list — never a fabricated/default value standing in for
      real data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf


class AudioLoadError(Exception):
    """Raised when a file cannot be validated/loaded as a usable WAV file."""


@dataclass
class AudioMetadata:
    file: str
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_seconds: Optional[float] = None
    sample_count: Optional[int] = None
    bit_depth: Optional[int] = None
    subtype: Optional[str] = None
    format: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_rate_hz": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "bit_depth": self.bit_depth,
            "subtype": self.subtype,
            "format": self.format,
        }


# soundfile subtype strings -> bit depth, where unambiguous.
# Anything not in this map yields bit_depth = None + a warning, rather than a guess.
_SUBTYPE_BIT_DEPTH = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


def validate_wav_path(path: str) -> None:
    """
    Validate that `path` exists, is a file, and is readable by libsndfile as a WAV.
    Raises AudioLoadError with a clear message on failure. Does not load samples.
    """
    if not isinstance(path, str) or not path:
        raise AudioLoadError("No file path provided.")
    if not os.path.exists(path):
        raise AudioLoadError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise AudioLoadError(f"Path is not a file: {path}")
    if os.path.getsize(path) == 0:
        raise AudioLoadError(f"File is empty: {path}")

    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001 - we want to convert any backend error
        raise AudioLoadError(f"File could not be read as audio ({path}): {exc}") from exc

    if info.format != "WAV":
        raise AudioLoadError(
            f"File is not a WAV container (detected format: {info.format}): {path}"
        )


def get_metadata(path: str) -> AudioMetadata:
    """
    Read WAV container metadata WITHOUT decoding all samples.
    Assumes validate_wav_path(path) has already succeeded (or will re-raise if not).
    """
    validate_wav_path(path)
    info = sf.info(path)
    warnings: List[str] = []

    bit_depth = _SUBTYPE_BIT_DEPTH.get(info.subtype)
    if bit_depth is None:
        warnings.append(
            f"Could not determine exact bit depth for subtype '{info.subtype}'; "
            "reporting subtype string only."
        )

    duration = info.frames / info.samplerate if info.samplerate else None
    if duration is None:
        warnings.append("Sample rate reported as 0; duration could not be computed.")

    return AudioMetadata(
        file=path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=duration,
        sample_count=info.frames,
        bit_depth=bit_depth,
        subtype=info.subtype,
        format=info.format,
        warnings=warnings,
    )


def load_audio(path: str) -> tuple[np.ndarray, int, AudioMetadata]:
    """
    Load a WAV file's samples as float64 in range [-1.0, 1.0] (soundfile's default
    float normalization for integer PCM subtypes), plus its metadata.

    Returns:
        audio: np.ndarray, shape (n_samples,) for mono or (n_samples, n_channels) for
               multi-channel, dtype float64. The array is a fresh in-memory copy;
               the original file on disk is never modified.
        sample_rate: int
        metadata: AudioMetadata describing the ORIGINAL file as read from disk.

    Raises:
        AudioLoadError if the file is missing, empty, unreadable, or not a WAV.
    """
    metadata = get_metadata(path)
    try:
        audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise AudioLoadError(f"Failed to decode samples from {path}: {exc}") from exc

    if audio.size == 0:
        metadata.warnings.append("Decoded audio contains zero samples.")

    return audio, sample_rate, metadata
