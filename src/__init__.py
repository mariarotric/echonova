"""
EchoNova Milestone 1 pipeline package.

Modules:
    io_utils            - WAV loading and validation
    preprocessing        - mono conversion, resampling, silence trim, normalization
    signal_quality       - measured signal-quality metrics
    acoustic_analysis    - F0, MFCC, spectral features
    pipeline             - orchestrates the above into a single JSON report

Scope (Milestone 1 only):
    WAV -> validation -> preprocessing -> signal quality -> acoustic analysis -> JSON report

Explicitly NOT implemented here (future milestones, see PROGRESS.md):
    voice-source / glottal analysis, speech-behaviour analysis, speaker consistency,
    evidence fusion, risk scoring, challenge-response, FastAPI backend, React frontend.
"""
