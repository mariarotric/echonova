# EchoNova

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**
Prototype for SIH 2026 — Problem Statement SIH26104.

## What This Is

EchoNova is a forensics-oriented voice authenticity analysis system. It takes a voice recording and produces an **evidence-based risk assessment** of whether the voice is genuine or synthetic/cloned — along with a transparent breakdown of which signals contributed to that assessment.

It is explicitly **not** a black-box "AI says X% fake" tool. Every value the system reports is meant to trace back to an actual computation performed on the audio.

## Current Status

**Milestone 1 complete.** The pipeline architecture is locked (see `ARCHITECTURE.md`). Milestone 1 implemented and tested: `WAV → validation → preprocessing → signal quality → acoustic analysis → JSON report`. See `PROGRESS.md` for full detail, what was tested, and known limitations.

- ✅ Architecture and project state documented
- ✅ Preprocessing, signal quality, and acoustic analysis modules — implemented and tested (52 passing tests), validated on synthetic test signals
- ⬜ Real recorded speech validation — not yet done (only a synthetic demo WAV exists so far)
- ⬜ Speech-behaviour, voice-source (glottal), speaker-consistency, fusion, risk scoring — not yet built
- 🚫 Frontend and backend — intentionally not started yet
- 🚫 Live microphone / telephony input — intentionally deferred
- 🚫 Telephony/channel degradation testing — intentionally deferred

See `PROGRESS.md` for the full milestone checklist.

## Why This Problem Matters

Voice cloning has become accessible enough that a few seconds of someone's speech can be used to generate convincing synthetic audio of them saying anything. This creates real risk for impersonation-based fraud (e.g., fake calls from a "relative" or "executive" requesting money or access). EchoNova aims to detect this by combining several independent layers of evidence rather than relying on any single signal.

## Core Idea

The detection pipeline combines multiple evidence branches:

1. **Acoustic analysis** — spectral and pitch-based features.
2. **Speech behaviour analysis** — rhythm, pacing, and prosodic patterns.
3. **Voice-source temporal dynamics** — the system estimates the underlying glottal source signal, identifies successive vocal pulses/cycles, and tracks how their timing, shape, and amplitude evolve over the utterance. This is used as an *additional* authenticity signal, not a standalone verdict.
4. **Speaker consistency** — internal and/or reference-based consistency checks.

All of this is combined via **evidence fusion**, weighted by an upfront **signal quality assessment** (so a poor-quality recording doesn't get treated as if it carries strong glottal-level evidence it can't actually support). The result is an interpretable **authenticity risk score** with an explanation, feeding into a **final action** decision — with an optional challenge-response step for ambiguous cases.

> Note on the voice-source technique: glottal inverse filtering and pulse-cycle analysis are established DSP methods, not something invented here. EchoNova's contribution is *how* this signal is reconstructed and fused with other evidence inside a real-time voice-cloning detector — not the base signal-processing technique itself.

## Pipeline

```
WAV → preprocessing → signal quality → acoustic analysis → speech behaviour analysis
   → voice-source temporal dynamics → evidence fusion → authenticity risk score
   → optional challenge-response → final action
```

Full detail in `ARCHITECTURE.md`. Milestone 1 implements the first three analytical stages (preprocessing, signal quality, acoustic analysis) plus validation and JSON reporting — see `PROGRESS.md`.

## Planned Technology Stack

| Layer | Technology | Status |
|---|---|---|
| Frontend | React + TypeScript + Tailwind + Framer Motion | Not started |
| Backend | Python + FastAPI | Not started |
| Audio / DSP | NumPy + SciPy + Librosa | In progress (Milestone 1 done) |
| ML | PyTorch/ONNX (only if justified by experiments) | Not started |

## Design Principles

- Lightweight, reproducible DSP and classical ML before reaching for large models.
- Every branch of evidence is fused — no single detector (including the glottal branch) decides the outcome alone.
- Signal quality actively controls how much trust each branch is given.
- Explainability and forensic traceability over raw "black box" accuracy claims.
- No fabricated numbers anywhere, including in early demos — unavailable measurements return `null` plus a warning.

## Project Documents

- `CLAUDE.md` — working rules and constraints for AI-assisted development on this project.
- `PROGRESS.md` — live milestone tracker (what's done, tested, and known limitations).
- `ARCHITECTURE.md` — full locked pipeline and module specification.
- `DECISIONS.md` — log of key decisions and their rationale.

## Getting Started (Milestone 1 code)

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 -m src.cli data/raw_wav/your_file.wav --out-dir results/
```

See `examples/RUNNING_EXAMPLES.md` for a full walkthrough, including how to generate a synthetic demo WAV if you don't have a real recording on hand yet.
