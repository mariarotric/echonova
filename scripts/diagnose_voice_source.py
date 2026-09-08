import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import load_audio
from src.preprocessing import preprocess
from src.voice_source import (
    estimate_source_signal,
    detect_vocal_pulses,
)


AUDIO_PATH = "data/raw_wav/genuine_01.wav"


# ---------------------------------------------------------
# Load + preprocess
# ---------------------------------------------------------

audio, sr, _ = load_audio(AUDIO_PATH)

prep = preprocess(audio, sr)

source = estimate_source_signal(
    prep.audio,
    prep.sample_rate,
)

pulses = detect_vocal_pulses(
    source.source_signal,
    source.sample_rate,
)

x = source.source_signal
fs = source.sample_rate


# ---------------------------------------------------------
# 1. Full estimated source signal
# ---------------------------------------------------------

time = np.arange(len(x)) / fs

plt.figure(figsize=(14, 5))
plt.plot(time, x)

pulse_times = pulses.pulse_times_seconds

valid_pulse_times = pulse_times[
    pulse_times <= time[-1]
]

pulse_indices = (
    valid_pulse_times * fs
).astype(int)

pulse_indices = pulse_indices[
    pulse_indices < len(x)
]

plt.scatter(
    valid_pulse_times,
    x[pulse_indices],
    marker="x",
)

plt.xlabel("Time (seconds)")
plt.ylabel("Estimated source amplitude")
plt.title("EchoNova — Estimated Voice Source + Detected Pulses")
plt.tight_layout()

plt.savefig(
    "results/figures/source_pulses_full.png",
    dpi=150,
)

plt.close()


# ---------------------------------------------------------
# 2. Zoom into first 0.5 seconds containing pulses
# ---------------------------------------------------------

zoom_end = min(0.5, time[-1])

mask = time <= zoom_end

zoom_time = time[mask]
zoom_signal = x[mask]

zoom_pulse_mask = valid_pulse_times <= zoom_end
zoom_pulse_times = valid_pulse_times[zoom_pulse_mask]

zoom_indices = (
    zoom_pulse_times * fs
).astype(int)

zoom_indices = zoom_indices[
    zoom_indices < len(x)
]

plt.figure(figsize=(14, 5))
plt.plot(zoom_time, zoom_signal)

plt.scatter(
    zoom_pulse_times,
    x[zoom_indices],
    marker="x",
)

plt.xlabel("Time (seconds)")
plt.ylabel("Estimated source amplitude")
plt.title("EchoNova — Pulse Detection Zoom")
plt.tight_layout()

plt.savefig(
    "results/figures/source_pulses_zoom.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# 3. Inter-pulse interval distribution
# ---------------------------------------------------------

intervals = pulses.inter_pulse_intervals_seconds

intervals = intervals[
    np.isfinite(intervals) & (intervals > 0)
]

plt.figure(figsize=(10, 5))
plt.hist(intervals, bins=40)

plt.xlabel("Inter-pulse interval (seconds)")
plt.ylabel("Number of intervals")
plt.title("EchoNova — Inter-Pulse Interval Distribution")
plt.tight_layout()

plt.savefig(
    "results/figures/pulse_interval_distribution.png",
    dpi=150,
)

plt.close()


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("Diagnostic plots generated:")
print("  results/figures/source_pulses_full.png")
print("  results/figures/source_pulses_zoom.png")
print("  results/figures/pulse_interval_distribution.png")

print()
print("Pulses detected:", pulses.pulses_detected)
print("Pulse reliability:", pulses.reliability)

if intervals.size:
    print("Median interval:", np.median(intervals))
    print("Mean interval:", np.mean(intervals))
    print("Std interval:", np.std(intervals))