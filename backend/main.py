from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Allow the backend to import the project-level src package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import io_utils
from src.preprocessing import preprocess
from src.signal_quality import compute_signal_quality
from src.acoustic_analysis import analyze_acoustics
from src.voice_source.source_extraction import estimate_source_signal
from src.voice_source.pulse_detection import detect_vocal_pulses
from src.voice_source.temporal_features import compute_temporal_dynamics


app = FastAPI(
    title="EchoNova Voice Forensics API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "EchoNova Voice Forensics Engine",
    }


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyse one uploaded WAV file.

    Returns:
        Milestone-1 analysis plus the separate voice-source
        temporal-dynamics experiment.
    """

    filename = Path(file.filename or "audio.wav").name

    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="EchoNova currently accepts WAV audio files.",
        )

    temp_path: Path | None = None

    try:
        suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp_path = Path(temp.name)
            shutil.copyfileobj(file.file, temp)

        # ---------------------------------------------------------
        # Existing Milestone 1 pipeline stages
        # ---------------------------------------------------------

        try:
            raw_audio, sample_rate, metadata = io_utils.load_audio(
                str(temp_path)
            )
        except io_utils.AudioLoadError as exc:
            return {
                "schema_version": "1.0.0",
                "milestone": "1",
                "file": filename,
                "error": str(exc),
                "warnings": [],
            }

        warnings: list[str] = []

        warnings.extend(
            f"validation: {message}"
            for message in metadata.warnings
        )

        mono_original, _ = preprocess_to_mono(raw_audio)

        prep_result = preprocess(
            raw_audio,
            sample_rate,
        )

        warnings.extend(
            f"preprocessing: {message}"
            for message in prep_result.warnings
        )

        sq_result = compute_signal_quality(
            prep_result.audio,
            prep_result.sample_rate,
            clipping_reference_audio=mono_original,
        )

        warnings.extend(
            f"signal_quality: {message}"
            for message in sq_result.warnings
        )

        ac_result = analyze_acoustics(
            prep_result.audio,
            prep_result.sample_rate,
        )

        warnings.extend(
            f"acoustic_analysis: {message}"
            for message in ac_result.warnings
        )

        # ---------------------------------------------------------
        # Separate voice-source experiment
        # ---------------------------------------------------------

        source_result = estimate_source_signal(
            prep_result.audio,
            prep_result.sample_rate,
        )

        warnings.extend(
            f"voice_source: {message}"
            for message in source_result.warnings
        )

        pulse_result = detect_vocal_pulses(
            source_result.source_signal,
            source_result.sample_rate,
        )

        warnings.extend(
            f"pulse_detection: {message}"
            for message in pulse_result.warnings
        )

        temporal_result = compute_temporal_dynamics(
            pulse_result.inter_pulse_intervals_seconds,
            pulse_result.pulse_amplitudes,
            pulse_result.estimated_f0_hz,
            valid_interval_mask=pulse_result.valid_interval_mask,
        )

        warnings.extend(
            f"temporal_dynamics: {message}"
            for message in temporal_result.warnings
        )

        return {
            "schema_version": "1.0.0",
            "milestone": "1",
            "file": filename,
            "audio": metadata.to_dict(),
            "preprocessing": prep_result.to_dict(),
            "signal_quality": sq_result.to_dict(),
            "acoustic_analysis": ac_result.to_dict(),
            "voice_source": {
                "source_extraction": source_result.to_dict(),
                "pulse_detection": pulse_result.to_dict(),
                "temporal_dynamics": temporal_result.to_dict(),
            },
            "warnings": warnings,
            "error": None,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def preprocess_to_mono(audio):
    """
    Keep the original Milestone 1 clipping-reference behaviour.
    """
    import numpy as np

    audio = np.asarray(audio)

    if audio.ndim == 1:
        return audio.astype(float), None

    if audio.ndim == 2:
        # io_utils may return channels x samples or samples x channels.
        if audio.shape[0] <= 8:
            mono = np.mean(audio, axis=0)
        else:
            mono = np.mean(audio, axis=1)

        return mono.astype(float), None

    raise ValueError("Unsupported audio dimensions.")


@app.get("/")
def root():
    return {
        "name": "EchoNova",
        "description": "Voice Forensics Engine",
        "status": "ready",
    }