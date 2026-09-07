import numpy as np
import pytest

from src import io_utils


def test_validate_wav_path_missing_file():
    with pytest.raises(io_utils.AudioLoadError, match="not found"):
        io_utils.validate_wav_path("/nonexistent/path/does_not_exist.wav")


def test_validate_wav_path_empty_string():
    with pytest.raises(io_utils.AudioLoadError):
        io_utils.validate_wav_path("")


def test_get_metadata_mono(speech_like_wav):
    meta = io_utils.get_metadata(speech_like_wav)
    assert meta.sample_rate == 22050
    assert meta.channels == 1
    assert meta.sample_count is not None and meta.sample_count > 0
    assert meta.duration_seconds is not None and meta.duration_seconds > 2.0
    assert meta.bit_depth == 16
    assert meta.format == "WAV"


def test_get_metadata_stereo(stereo_wav):
    meta = io_utils.get_metadata(stereo_wav)
    assert meta.channels == 2


def test_load_audio_returns_expected_shapes(speech_like_wav):
    audio, sr, meta = io_utils.load_audio(speech_like_wav)
    assert sr == 22050
    assert audio.ndim == 1
    assert audio.shape[0] == meta.sample_count
    assert audio.dtype == np.float64


def test_load_audio_stereo_shape(stereo_wav):
    audio, sr, meta = io_utils.load_audio(stereo_wav)
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_load_audio_does_not_modify_original_file(speech_like_wav):
    import os

    original_size = os.path.getsize(speech_like_wav)
    original_mtime = os.path.getmtime(speech_like_wav)
    io_utils.load_audio(speech_like_wav)
    assert os.path.getsize(speech_like_wav) == original_size
    assert os.path.getmtime(speech_like_wav) == original_mtime


def test_load_audio_nonexistent_raises():
    with pytest.raises(io_utils.AudioLoadError):
        io_utils.load_audio("/nonexistent/file.wav")


def test_get_metadata_tiny_wav(tiny_wav):
    meta = io_utils.get_metadata(tiny_wav)
    assert meta.sample_count == 5
