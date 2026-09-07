"""
pipeline.py — Milestone 1 orchestration.

Implements exactly the pipeline documented in ARCHITECTURE.md for Milestone 1:

    WAV -> validation -> preprocessing -> signal quality -> acoustic analysis -> JSON report

No authenticity scoring, no ML, no glottal/voice-source analysis, no fusion —
those are explicitly out of scope for this milestone (see PROGRESS.md / CLAUDE.md).

Output contract (see README/examples for a sample):

{
  "schema_version": "1.0.0",
  "milestone": "1",
  "file": "<path>",
  "audio": { ... original file metadata from io_utils ... },
  "preprocessing": { ... },
  "signal_quality": { ... },
  "acoustic_analysis": { ... },
  "warnings": [ "<stage>: <message>", ... ],
  "error": null | "<message>"
}

If the file cannot even be validated/loaded, the report contains only
`schema_version`, `milestone`, `file`, `error`, and `warnings` — the other
sections are omitted rather than filled with fabricated placeholder values.
"""

from __future__ import annotations

from typing import List, Optional

from . import io_utils
from . import preprocessing as preprocessing_mod
from . import signal_quality as signal_quality_mod
from . import acoustic_analysis as acoustic_analysis_mod

SCHEMA_VERSION = "1.0.0"
MILESTONE = "1"


def _tag(stage: str, messages: List[str]) -> List[str]:
    return [f"{stage}: {m}" for m in messages]


def run_pipeline(path: str) -> dict:
    """
    Run the full Milestone 1 pipeline on a single WAV file and return a JSON-
    serializable dict following the documented schema. Never raises for
    ordinary "bad audio" conditions (unreadable file, silent file, non-speech
    audio, etc.) — those are reported via `error` and/or `warnings`. Only
    programmer errors (e.g. bad arguments) propagate as exceptions.
    """
    warnings: List[str] = []

    # --- Stage: validation + load -----------------------------------------
    try:
        raw_audio, sample_rate, metadata = io_utils.load_audio(path)
    except io_utils.AudioLoadError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "file": path,
            "error": str(exc),
            "warnings": [],
        }

    warnings.extend(_tag("validation", metadata.warnings))

    # Keep a mono, pre-normalization copy of the ORIGINAL-sample-rate audio
    # purely for clipping detection, so normalization can't mask real clipping
    # present in the source file (see signal_quality.py docstring).
    mono_original, _ = preprocessing_mod.to_mono(raw_audio)

    # --- Stage: preprocessing ------------------------------------------------
    prep_result = preprocessing_mod.preprocess(raw_audio, sample_rate)
    warnings.extend(_tag("preprocessing", prep_result.warnings))

    # --- Stage: signal quality ------------------------------------------------
    sq_result = signal_quality_mod.compute_signal_quality(
        prep_result.audio,
        prep_result.sample_rate,
        clipping_reference_audio=mono_original,
    )
    warnings.extend(_tag("signal_quality", sq_result.warnings))

    # --- Stage: acoustic analysis ----------------------------------------------
    ac_result = acoustic_analysis_mod.analyze_acoustics(prep_result.audio, prep_result.sample_rate)
    warnings.extend(_tag("acoustic_analysis", ac_result.warnings))

    report = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "file": path,
        "audio": metadata.to_dict(),
        "preprocessing": prep_result.to_dict(),
        "signal_quality": sq_result.to_dict(),
        "acoustic_analysis": ac_result.to_dict(),
        "warnings": warnings,
        "error": None,
    }
    return report
