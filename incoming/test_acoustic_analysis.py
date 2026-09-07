import numpy as np

from src import acoustic_analysis as aa


def test_compute_f0_statistics_detects_known_pitch():
    sr = 16000
    f0_true = 150.0
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * f0_true * t)
    stats = aa.compute_f0_statistics(tone, sr)
    assert stats["warning"] is None
    assert stats["mean"] is not None
    # Pitch tracking on a pure tone should land close to the true frequency.
    assert abs(stats["mean"] - f0_true) < 5.0


def test_compute_f0_statistics_no_voiced_frames_on_noise():
    sr = 16000
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1.0, size=sr)
    stats = aa.compute_f0_statistics(noise, sr)
    # Pure white noise should not be confidently voiced; mean may be None.
    if stats["mean"] is None:
        assert stats["warning"] is not None


def test_compute_f0_statistics_too_short_returns_warning():
    sr = 16000
    audio = np.zeros(100)
    stats = aa.compute_f0_statistics(audio, sr)
    assert stats["mean"] is None
    assert "short" in stats["warning"].lower()


def test_compute_mfcc_statistics_shape():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    mean, std, warning = aa.compute_mfcc_statistics(tone, sr, n_mfcc=13)
    assert warning is None
    assert len(mean) == 13
    assert len(std) == 13


def test_compute_mfcc_statistics_too_short():
    mean, std, warning = aa.compute_mfcc_statistics(np.zeros(10), 16000)
    assert mean is None and std is None
    assert warning is not None


def test_compute_spectral_centroid_higher_for_higher_frequency_tone():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    low_tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    high_tone = 0.5 * np.sin(2 * np.pi * 2000 * t)

    low_mean, _, _ = aa.compute_spectral_centroid(low_tone, sr)
    high_mean, _, _ = aa.compute_spectral_centroid(high_tone, sr)
    assert high_mean > low_mean


def test_compute_zero_crossing_rate_higher_for_higher_frequency():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    low_tone = np.sin(2 * np.pi * 100 * t)
    high_tone = np.sin(2 * np.pi * 3000 * t)

    low_zcr, _, _ = aa.compute_zero_crossing_rate(low_tone)
    high_zcr, _, _ = aa.compute_zero_crossing_rate(high_tone)
    assert high_zcr > low_zcr


def test_analyze_acoustics_end_to_end_on_speech_like_signal():
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 150.0
    voice = (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    )
    voice = voice / np.max(np.abs(voice)) * 0.7

    result = aa.analyze_acoustics(voice, sr)
    assert result.f0_mean_hz is not None
    assert abs(result.f0_mean_hz - f0) < 10.0
    assert result.mfcc_mean is not None and len(result.mfcc_mean) == aa.N_MFCC
    assert result.spectral_centroid_mean_hz is not None
    assert result.spectral_bandwidth_mean_hz is not None
    assert result.zero_crossing_rate_mean is not None
