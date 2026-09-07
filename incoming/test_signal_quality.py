import numpy as np

from src import signal_quality as sq


def test_compute_rms_known_value():
    # A constant-amplitude signal's RMS equals its amplitude.
    audio = np.full(1000, 0.5)
    assert np.isclose(sq.compute_rms(audio), 0.5)


def test_compute_rms_empty_returns_none():
    assert sq.compute_rms(np.array([])) is None


def test_compute_peak_amplitude():
    audio = np.array([0.1, -0.9, 0.3])
    assert np.isclose(sq.compute_peak_amplitude(audio), 0.9)


def test_compute_clipping_percentage_all_clipped():
    audio = np.full(100, 1.0)
    pct = sq.compute_clipping_percentage(audio, threshold=0.99)
    assert pct == 100.0


def test_compute_clipping_percentage_none_clipped():
    audio = np.full(100, 0.1)
    pct = sq.compute_clipping_percentage(audio, threshold=0.99)
    assert pct == 0.0


def test_compute_clipping_percentage_partial():
    audio = np.concatenate([np.full(50, 1.0), np.full(50, 0.0)])
    pct = sq.compute_clipping_percentage(audio, threshold=0.99)
    assert np.isclose(pct, 50.0)


def test_silence_ratio_all_silent():
    sr = 16000
    audio = np.zeros(sr)
    ratio = sq.compute_silence_ratio(audio, sr)
    assert ratio == 1.0


def test_silence_ratio_loud_tone_is_low():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * 200 * t)
    ratio = sq.compute_silence_ratio(tone, sr)
    assert ratio < 0.1


def test_estimate_snr_db_returns_none_for_too_short_signal():
    sr = 16000
    audio = np.zeros(10)
    assert sq.estimate_snr_db(audio, sr) is None


def test_estimate_snr_db_higher_for_cleaner_signal_with_pauses():
    """
    The percentile-based SNR proxy (see signal_quality.py docstring) estimates
    the noise floor from the QUIETEST frames. That assumption only holds when
    the recording actually contains quiet/pause frames for it to sample from.
    A continuous, gap-free tone (loud in every frame) does not give the proxy
    anything to measure as "noise floor", so this test — correctly — uses a
    signal with silent gaps, mirroring real speech containing pauses.
    """
    sr = 16000
    t_tone = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * 200 * t_tone)
    pause = np.zeros(sr // 2)

    rng = np.random.default_rng(0)
    clean = np.concatenate([pause, tone + rng.normal(0, 0.001, size=tone.shape), pause])
    noisy = np.concatenate([pause, tone + rng.normal(0, 0.3, size=tone.shape), pause]) \
        + rng.normal(0, 0.05, size=clean.shape)  # noise floor present even during pauses

    snr_clean = sq.estimate_snr_db(clean, sr)
    snr_noisy = sq.estimate_snr_db(noisy, sr)
    assert snr_clean is not None and snr_noisy is not None
    assert snr_clean > snr_noisy


def test_estimate_snr_db_low_for_continuous_tone_with_no_pauses():
    """
    Documents a known limitation: with no quiet frames at all, the percentile
    proxy has no noise floor to measure against, so it reports a low/near-zero
    SNR even for a perfectly clean continuous tone. This is expected behaviour
    for this heuristic, not a bug — recorded here so the limitation is explicit
    and covered by a test rather than silently assumed.
    """
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * 200 * t)
    snr = sq.estimate_snr_db(tone, sr)
    assert snr is not None
    assert snr < 5.0


def test_estimate_bandwidth_hz_low_freq_tone_has_low_bandwidth_proxy():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = np.sin(2 * np.pi * 100 * t)  # pure single low-frequency tone
    bw = sq.estimate_bandwidth_hz(tone, sr)
    assert bw is not None
    assert bw < 1000  # should be well below Nyquist for a 100 Hz tone


def test_estimate_bandwidth_hz_too_short_returns_none():
    assert sq.estimate_bandwidth_hz(np.zeros(10), 16000) is None


def test_compute_signal_quality_end_to_end_on_tone():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    result = sq.compute_signal_quality(tone, sr)
    assert result.duration_seconds is not None and np.isclose(result.duration_seconds, 2.0)
    assert result.rms_energy is not None
    assert result.peak_amplitude is not None
    assert result.quality_score is not None
    assert 0.0 <= result.quality_score <= 1.0


def test_compute_signal_quality_uses_clipping_reference_when_provided():
    sr = 16000
    original = np.full(sr, 1.0)  # fully clipped in the "original"
    normalized = np.full(sr, 0.5)  # hypothetically rescaled down after normalization
    result = sq.compute_signal_quality(normalized, sr, clipping_reference_audio=original)
    assert result.clipping_percentage == 100.0


def test_compute_signal_quality_flags_missing_reference():
    sr = 16000
    audio = np.full(sr, 0.5)
    result = sq.compute_signal_quality(audio, sr)  # no reference supplied
    assert any("clipping" in w.lower() for w in result.warnings)
