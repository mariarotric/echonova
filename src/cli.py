"""
cli.py — Milestone 1 command-line entry point.

Usage:
    python -m src.cli path/to/file.wav
    python -m src.cli path/to/file.wav --out results/report.json
    python -m src.cli path/to/dir_of_wavs/ --out-dir results/

Prints the JSON report to stdout (and optionally writes it to disk). This is a
thin wrapper around src.pipeline.run_pipeline — no analysis logic lives here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import run_pipeline


def _iter_wav_files(path: str):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".wav"):
                yield os.path.join(path, name)
    else:
        yield path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="EchoNova Milestone 1: WAV -> validation -> preprocessing "
        "-> signal quality -> acoustic analysis -> JSON report."
    )
    parser.add_argument("input", help="Path to a WAV file or a directory of WAV files.")
    parser.add_argument(
        "--out", help="Write the JSON report to this path (single-file input only)."
    )
    parser.add_argument(
        "--out-dir",
        help="Write one JSON report per input WAV into this directory "
        "(used when --input is a directory, or to also save single-file output).",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indent level for printed/written output."
    )
    args = parser.parse_args(argv)

    wav_paths = list(_iter_wav_files(args.input))
    if not wav_paths:
        print(f"No .wav files found at: {args.input}", file=sys.stderr)
        return 1

    exit_code = 0
    for wav_path in wav_paths:
        report = run_pipeline(wav_path)
        text = json.dumps(report, indent=args.indent)
        print(text)

        if report.get("error"):
            exit_code = 2

        out_path = None
        if args.out and len(wav_paths) == 1:
            out_path = args.out
        elif args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(wav_path))[0]
            out_path = os.path.join(args.out_dir, f"{base}.json")

        if out_path:
            with open(out_path, "w") as f:
                f.write(text)
            print(f"# report written to {out_path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
