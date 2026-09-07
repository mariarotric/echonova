import numpy as np

from src import preprocessing


def test_to_mono_already_mono():
    audio = np.array([0.1, 0.2, -0.1])
    mono, was_downmixed = preprocessing.to_mono(audio)
    assert was_downmixed is False
    np.testing.assert_array_equal(mono, audio)


def test_to_mono_averages_channels():
    left = np.array([1.0, 1.0, 1.0])
    right = np.array([-1.0, -1.0, -1.0])
    stereo = np.stack([left, right], axis=1)
    mono, was_downmixed = preprocessing.to_mono(stereo)
    assert was_downmixed is True
    np.testing.assert_allclose(mono, np.zeros(3))


def test_resample_audio_same_rate_no_op():
    audio = np.random.randn(1000)
    out, was_resampled = preprocessing.resample_audio(audio, orig_sr=16000, target_sr=16000)
    assert was_resampled is False
    np.testing.assert_array_equal(out, audio)


def test_resample_audio_changes_length():
    sr_in, sr_out = 22050, 16000
    audio = np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, sr_in, endpoint=False))
    out, was_resampled = preprocessing.resample_audio(audio, orig_sr=sr_in, target_sr=sr_out)
    assert was_resampled is True
    expected_len = int(round(len(audio) * sr_out / sr_in))
    assert abs(len(out) - expected_len) <= 2  # resampler may be off by a sample or two


def test_trim_silence_removes_leading_trailing_zeros():
    sr = 16000
    tone = 0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False))
    padded = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)])
    trimmed, start_s, end_s = preprocessing.trim_silence(padded, sr)
    assert trimmed.shape[0] < padded.shape[0]
    assert start_s > 0
    assert end_s > 0


def test_trim_silence_pure_silence_returns_original():
    sr = 16000
    silence = np.zeros(sr)
    trimmed, start_s, end_s = preprocessing.trim_silence(silence, sr)
    assert trimmed.shape[0] == silence.shape[0]
    assert start_s == 0.0 and end_s == 0.0


def test_normalize_amplitude_scales_peak_to_target():
    audio = np.array([0.1, -0.4, 0.2])
    normalized, was_normalized = preprocessing.normalize_amplitude(audio, peak_target=0.98)
    assert was_normalized is True
    assert np.isclose(np.max(np.abs(normalized)), 0.98)


def test_normalize_amplitude_skips_near_silent_signal():
    audio = np.zeros(100)
    normalized, was_normalized = preprocessing.normalize_amplitude(audio)
    assert was_normalized is False
    np.testing.assert_array_equal(normalized, audio)


def test_preprocess_full_chain_speech_like(speech_like_wav):
    from src import io_utils

    audio, sr, _ = io_utils.load_audio(speech_like_wav)
    result = preprocessing.preprocess(audio, sr)

    assert result.was_resampled is True  # source is 22050Hz, target is 16000Hz
    assert result.sample_rate == preprocessing.TARGET_SAMPLE_RATE_HZ
    assert result.was_downmixed_to_mono is False  # already mono
    assert result.duration_after_seconds > 0
    # Silence padding at start/end of the fixture should be at least partly trimmed.
    assert result.silence_trimmed_seconds_start > 0
    assert result.silence_trimmed_seconds_end > 0


def test_preprocess_stereo_input_downmixes(stereo_wav):
    from src import io_utils

    audio, sr, _ = io_utils.load_audio(stereo_wav)
    result = preprocessing.preprocess(audio, sr)
    assert result.was_downmixed_to_mono is True
    assert result.audio.ndim == 1


def test_preprocess_silent_input_skips_normalization(silent_wav):
    from src import io_utils

    audio, sr, _ = io_utils.load_audio(silent_wav)
    result = preprocessing.preprocess(audio, sr)
    assert result.was_normalized is False
    assert any("silen" in w.lower() or "zero" in w.lower() for w in result.warnings)
