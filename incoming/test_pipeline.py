import json

from src import pipeline


def test_run_pipeline_missing_file_returns_error_report():
    report = pipeline.run_pipeline("/nonexistent/file.wav")
    assert report["error"] is not None
    assert report["file"] == "/nonexistent/file.wav"
    assert report["schema_version"] == pipeline.SCHEMA_VERSION
    # No analysis sections should be fabricated when the file couldn't be loaded.
    assert "audio" not in report
    assert "signal_quality" not in report


def test_run_pipeline_schema_keys_present_on_valid_file(speech_like_wav):
    report = pipeline.run_pipeline(speech_like_wav)
    assert report["error"] is None
    for key in ("schema_version", "milestone", "file", "audio", "preprocessing",
                "signal_quality", "acoustic_analysis", "warnings"):
        assert key in report


def test_run_pipeline_output_is_json_serializable(speech_like_wav):
    report = pipeline.run_pipeline(speech_like_wav)
    # Will raise if any value (e.g. a numpy type) is not JSON-serializable.
    text = json.dumps(report)
    assert isinstance(text, str)
    round_tripped = json.loads(text)
    assert round_tripped["file"] == speech_like_wav


def test_run_pipeline_reports_pitch_for_speech_like_signal(speech_like_wav):
    report = pipeline.run_pipeline(speech_like_wav)
    f0 = report["acoustic_analysis"]["f0"]
    assert f0["mean_hz"] is not None
    assert 100 < f0["mean_hz"] < 250  # fixture's fundamental is 150 Hz


def test_run_pipeline_on_silent_file_has_null_pitch_and_warning(silent_wav):
    report = pipeline.run_pipeline(silent_wav)
    assert report["error"] is None
    f0 = report["acoustic_analysis"]["f0"]
    assert f0["mean_hz"] is None
    assert any("voiced" in w.lower() or "silen" in w.lower() for w in report["warnings"])


def test_run_pipeline_on_clipped_file_reports_clipping(clipped_wav):
    report = pipeline.run_pipeline(clipped_wav)
    assert report["error"] is None
    clipping_pct = report["signal_quality"]["clipping_percentage"]
    assert clipping_pct is not None
    assert clipping_pct > 0.0


def test_run_pipeline_on_stereo_file_downmixes(stereo_wav):
    report = pipeline.run_pipeline(stereo_wav)
    assert report["error"] is None
    assert report["audio"]["channels"] == 2  # original metadata
    assert report["preprocessing"]["was_downmixed_to_mono"] is True


def test_run_pipeline_on_tiny_file_does_not_crash_and_reports_warnings(tiny_wav):
    report = pipeline.run_pipeline(tiny_wav)
    assert report["error"] is None
    assert len(report["warnings"]) > 0
    # Pitch/MFCC/spectral fields should be null rather than fabricated.
    assert report["acoustic_analysis"]["f0"]["mean_hz"] is None
