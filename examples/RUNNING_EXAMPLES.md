# Running the Milestone 1 Pipeline

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. (Optional) Generate a synthetic example WAV

No externally recorded WAV file was available when this milestone was built,
so this script creates one small **synthetic** example (a harmonic vowel-like
tone with vibrato, background hiss, and silence padding) purely to prove the
pipeline runs on a real file on disk. It is NOT real human speech.

```bash
python3 examples/generate_example_wav.py
# -> writes data/raw_wav/example_synthetic_vowel.wav
```

Replace this with real recorded WAV files as soon as they're available — the
pipeline works identically on any valid mono/stereo PCM or float WAV file.

## 3. Run the pipeline from the command line

Single file, print JSON to stdout:

```bash
python3 -m src.cli data/raw_wav/example_synthetic_vowel.wav
```

Single file, also save the JSON report to disk:

```bash
python3 -m src.cli data/raw_wav/example_synthetic_vowel.wav --out-dir results/
```

Every WAV file in a directory:

```bash
python3 -m src.cli data/raw_wav/ --out-dir results/
```

## 4. Run the pipeline from Python

```python
from src.pipeline import run_pipeline
import json

report = run_pipeline("data/raw_wav/example_synthetic_vowel.wav")
print(json.dumps(report, indent=2))
```

## 5. Run the test suite

```bash
python3 -m pytest tests/ -v
```

## What you get back

A single JSON dict with keys: `schema_version`, `milestone`, `file`, `audio`,
`preprocessing`, `signal_quality`, `acoustic_analysis`, `warnings`, `error`.
If a value could not be reliably computed (e.g. pitch on a silent or
non-speech file), it appears as `null` and an explanatory string is added to
`warnings` — never a fabricated number. See ARCHITECTURE.md for what each
pipeline stage does and why.
