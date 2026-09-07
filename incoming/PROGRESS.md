# PROGRESS.md — EchoNova Build Status

Last updated: after Milestone 1 (WAV → validation → preprocessing → signal quality → acoustic analysis → JSON report).

## Current Phase

**Phase 0: Project Foundation — DONE.**
**Phase 1 (Preprocessing) + Phase 2 (Signal Quality) + Phase 3 (Acoustic Analysis) — Milestone 1 implementation DONE, validated on synthetic test signals; not yet validated on real recorded speech or real cloned/genuine samples.**

## Milestone 1 — Completed Work

**Goal (as specified):** a reproducible Python pipeline: `WAV → validation → preprocessing → signal quality → acoustic analysis → JSON report`. No authenticity scoring, no ML, no glottal analysis, no fusion, no FastAPI, no React — all correctly out of scope for this milestone and not built.

### What was built

- `src/io_utils.py` — WAV validation and loading (sample rate, channels, duration, sample count, bit depth where determinable from the container subtype). Never modifies the original file; only reads it.
- `src/preprocessing.py` — mono downmix, resample to a documented target (16 kHz), leading/trailing silence trim (`librosa.effects.trim`, 30 dB threshold), safe peak normalization (skipped rather than fabricated when the signal is effectively silent).
- `src/signal_quality.py` — RMS energy, peak amplitude, clipping percentage (measured on the pre-normalization array specifically so normalization can't mask real clipping), silence ratio, a documented percentile-based SNR proxy, a Welch-PSD-based bandwidth/occupied-bandwidth proxy, and an overall quality score computed only from whichever of the above were actually available (renormalized weights if some are missing — never backfilled with a guess).
- `src/acoustic_analysis.py` — F0 statistics via `librosa.pyin` (mean/median/std/min/max, computed only over frames the tracker marks voiced; null + warning if no voiced frames are found), MFCC mean/std per coefficient, spectral centroid, spectral bandwidth, zero-crossing rate.
- `src/pipeline.py` — orchestrates the four stages above into the agreed JSON schema (`schema_version`, `milestone`, `file`, `audio`, `preprocessing`, `signal_quality`, `acoustic_analysis`, `warnings`, `error`). File-load failures produce an `error` field and omit the analysis sections entirely, rather than filling them with placeholder values.
- `src/cli.py` — command-line entry point (`python -m src.cli <file_or_dir> [--out PATH | --out-dir DIR]`).
- `requirements.txt` — numpy, scipy, librosa, soundfile, pytest. No ML libraries added (per DECISIONS.md D-001 — not yet justified).
- `tests/` — 52 tests across `test_io_utils.py`, `test_preprocessing.py`, `test_signal_quality.py`, `test_acoustic_analysis.py`, `test_pipeline.py`, using synthetic fixtures (`tests/conftest.py`): a speech-like harmonic tone, pure silence, a hard-clipped tone, stereo audio, and a near-empty file — chosen to exercise the "unreliable measurement → null + warning" paths as well as the normal path.
- `examples/generate_example_wav.py` + `examples/README.md` — since no externally recorded WAV was available, a small labeled **synthetic** example WAV (harmonic tone with vibrato, background hiss, silence padding — explicitly documented as NOT real speech) was generated to prove the pipeline runs end-to-end on a real file on disk, plus a short how-to-run guide.

### What was tested

- `python3 -m pytest tests/ -v` → **52 passed, 0 failed.**
- One test iteration surfaced a genuine limitation in the SNR proxy (a continuous, pause-free tone gives the percentile-based method no quiet frames to treat as a noise floor, so it reports a near-zero SNR even when the signal is clean). Rather than special-casing it away, this was kept as a documented, explicitly tested limitation (see `signal_quality.py` docstring and `tests/test_signal_quality.py::test_estimate_snr_db_low_for_continuous_tone_with_no_pauses`), and a second test confirms the proxy behaves correctly on signals that do contain pauses (closer to real speech).
- The CLI was run against the generated synthetic example WAV (`data/raw_wav/example_synthetic_vowel.wav`, 22050 Hz, mono, 16-bit, 3.8s including padding). Output confirmed: correct resampling to 16 kHz, correct silence trimming (~0.4s trimmed from each end), zero clipping detected, F0 mean recovered at 140.14 Hz against a true synthetic fundamental of 140 Hz, and a full set of populated MFCC/spectral/ZCR values with `"warnings": []` and `"error": null`. Full output was shown in-conversation and saved to `results/example_synthetic_vowel.json`.

### Known limitations (honest, not yet resolved)

- **No real recorded speech has been run through the pipeline yet** — only synthetic sine/harmonic test signals. Real speech has formant structure, breathiness, natural jitter/shimmer, and background acoustic conditions that synthetic tones don't fully exercise. This is the top priority before trusting any of these numbers on real voice-cloning detection work.
- The SNR estimate is a coarse percentile-based proxy, not a lab-grade measurement, and is unreliable on signals without pauses (see above).
- The bandwidth proxy has not yet been validated against real telephony/narrowband audio (that validation is explicitly deferred per DECISIONS.md D-004).
- The quality-score weighting scheme (`QUALITY_SCORE_WEIGHTS` in `signal_quality.py`) is an initial documented heuristic, not yet tuned against any labeled dataset.
- F0/MFCC/spectral parameters (frame/hop sizes, F0 search range) are conventional defaults for speech, not yet tuned or validated against this project's actual target audio.

### Explicitly NOT done in Milestone 1 (by design)

- No voice-source / glottal analysis (Phase 5 — next milestone).
- No speech-behaviour analysis (Phase 4).
- No speaker consistency (Phase 6).
- No evidence fusion (Phase 7).
- No risk scoring (Phase 8).
- No challenge-response (Phase 9).
- No FastAPI backend (Phase 11).
- No React frontend (Phase 12).
- No telephony/channel degradation testing (Phase 13).

## Next Milestone (Milestone 2 — proposed scope, not started)

Per the locked implementation order in project planning: before moving to the glottal/voice-source branch (the core differentiator), the recommended next step is **Phase 4: Speech Behaviour Analysis** (speaking rate, pause pattern/rhythm statistics, prosodic variability over the utterance), extending the same JSON schema with a new `"speech_behaviour"` section — plus, in parallel, sourcing or recording a small set of **real** WAV files (genuine and, if available, cloned/synthetic) to validate Milestone 1's numbers against actual speech rather than synthetic tones only. Milestone 3 after that is the voice-source temporal dynamics branch (Phase 5), which should not begin until Milestone 1's acoustic/signal-quality numbers have been sanity-checked against real recordings.

**Do not start Milestone 2 without explicit confirmation — Milestone 1 is complete and this session stops here.**

---

## Original Foundation Notes (retained for history)

We established persistent project state (CLAUDE.md, ARCHITECTURE.md, DECISIONS.md, README.md, this file) before writing any DSP, ML, backend, or frontend code.

## Status Legend

- ✅ Done
- 🔶 In progress
- ⬜ Not started
- 🚫 Explicitly deferred (not a current goal)

## Milestone Checklist

### Phase 0 — Foundation
- ✅ Define locked architecture (pipeline stages, tech stack)
- ✅ Write CLAUDE.md, PROGRESS.md, ARCHITECTURE.md, DECISIONS.md, README.md
- ✅ Create project folder structure on disk
- ✅ Set up Python environment / requirements.txt
- ⬜ Collect/organize an initial small WAV dataset (genuine + cloned/synthetic samples) for experimentation — **only a synthetic demo WAV exists so far; real samples still needed**

### Phase 1 — Preprocessing
- ✅ WAV loading + validation (format, sample rate, mono/stereo handling)
- ✅ Resampling to a consistent target rate (16 kHz)
- ✅ Amplitude normalization (safe, skipped on near-silent input)
- ✅ Basic trimming of leading/trailing silence

### Phase 2 — Signal Quality
- ✅ SNR estimation (documented percentile-based proxy; known limitation on pause-free audio)
- ✅ Clipping detection
- ✅ Silence ratio
- ✅ Bandwidth / narrowband detection proxy (not yet validated against real telephony audio)
- ✅ Overall "signal quality score" (heuristic, computed only from available components — not yet tuned against labeled data)

### Phase 3 — Acoustic Analysis
- ✅ Pitch (F0) extraction and statistics
- ✅ MFCCs / spectral features (centroid, bandwidth, zero-crossing rate)
- ⬜ Basic jitter/shimmer (frame-level, acoustic-only, distinct from glottal-cycle-level analysis in Phase 5) — **not yet implemented, candidate for Milestone 2/3**
- ⬜ Formant-related features (deferred; not yet found necessary)

### Phase 4 — Speech Behaviour Analysis
- ⬜ Speaking rate
- ⬜ Pause pattern / rhythm statistics
- ⬜ Prosodic variability (pitch/energy contour dynamics over the utterance)

### Phase 5 — Voice-Source Temporal Dynamics (Core Differentiator)
- ⬜ Glottal source signal estimation (e.g., inverse filtering method — method to be chosen after experimentation, recorded in DECISIONS.md)
- ⬜ Glottal closure instant / pulse detection
- ⬜ Per-cycle features: timing (pulse-to-pulse intervals), shape, amplitude
- ⬜ Temporal evolution features: how pulse timing/shape/amplitude drift or stay stable across the utterance
- ⬜ Validate on a small labeled set (genuine vs. cloned) before trusting this branch

### Phase 6 — Speaker Consistency
- ⬜ Define what "consistency" means for the prototype (e.g., comparison against an enrolled reference sample, or internal consistency across the utterance)
- ⬜ Implement a minimal version once Phases 3–5 are stable

### Phase 7 — Evidence Fusion
- ⬜ Define fusion input contract (what each branch outputs)
- ⬜ Implement quality-weighted fusion (signal quality score modulates voice-source and acoustic branch weight)
- ⬜ Start with a simple, interpretable fusion method (e.g., weighted scoring or a shallow classifier) before considering anything heavier

### Phase 8 — Authenticity Risk Score
- ⬜ Convert fused evidence into an interpretable risk score
- ⬜ Attach explanation output: which branches contributed, and how much
- ⬜ No fabricated confidence numbers — every value traceable to computed evidence

### Phase 9 — Challenge-Response (Optional Layer)
- 🚫 Deferred until core detection pipeline (Phases 1–8) is validated
- ⬜ Define a minimal rule-based trigger condition (e.g., risk score in an ambiguous band)

### Phase 10 — Final Action
- ⬜ Define decision thresholds/categories (e.g., accept / flag / reject) based on risk score + confidence in the evidence

### Phase 11 — Backend (FastAPI)
- 🚫 Deferred until pipeline stages above are validated on real data
- ⬜ Wrap pipeline in FastAPI once ready

### Phase 12 — Frontend (React + TypeScript + Tailwind + Framer Motion)
- 🚫 Deferred until backend exists and returns real computed values
- ⬜ Build UI once there is real data to display

### Phase 13 — Telephony / Channel Degradation Testing
- 🚫 Deferred — noted as a required future validation step, not a current build target
- ⬜ Test pipeline against codec-compressed / narrowband / VoIP-simulated audio once core pipeline is stable

## Open Experiments / Questions

- Which glottal inverse filtering approach performs adequately on our available data (to be decided and logged in DECISIONS.md once tested).
- What minimum dataset size/composition is needed to meaningfully validate the voice-source branch.
- Whether a classical classifier is sufficient for fusion or a small learned model is warranted — do not assume; test first.

## Notes for Next Session

- No application code exists yet. Next actionable step after this state-setup phase is Phase 0's remaining items (folder structure, environment, initial dataset) followed by Phase 1 (preprocessing).
- Do not skip ahead to fusion, scoring, backend, or frontend before earlier phases produce real, validated outputs.
