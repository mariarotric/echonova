from __future__ import annotations

import json
import sys
from pathlib import Path

from src import io_utils
from src.preprocessing import preprocess
from src.voice_source.source_extraction import estimate_source_signal
from src.voice_source.pulse_detection import detect_vocal_pulses
from src.voice_source.temporal_features import compute_temporal_dynamics


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_voice_source_analysis.py <wav>")
        raise SystemExit(1)

    path = Path(sys.argv[1])

    raw_audio, sample_rate, metadata = io_utils.load_audio(str(path))

    prep = preprocess(raw_audio, sample_rate)

    source = estimate_source_signal(
        prep.audio,
        prep.sample_rate,
    )

    pulses = detect_vocal_pulses(
        source.source_signal,
        source.sample_rate,
    )

    temporal = compute_temporal_dynamics(
    pulses.inter_pulse_intervals_seconds,
    pulses.pulse_amplitudes,
    pulses.estimated_f0_hz,
    valid_interval_mask=pulses.valid_interval_mask,
    )

    result = {
        "schema_version": "1.0.0",
        "analysis": "voice_source_temporal_dynamics",
        "file": str(path),
        "audio": metadata.to_dict(),
        "preprocessing": prep.to_dict(),
        "source_extraction": source.to_dict(),
        "pulse_detection": pulses.to_dict(),
        "temporal_dynamics": temporal.to_dict(),
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{path.stem}_voice_source.json"

    output_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()