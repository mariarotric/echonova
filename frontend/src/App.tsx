import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AudioWaveform,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  Clock3,
  Download,
  FileAudio,
  FlaskConical,
  Gauge,
  Headphones,
  Pause,
  Play,
  Radio,
  ScanLine,
  Search,
  ShieldCheck,
  Signal,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Volume2,
  Waves,
  X,
  Zap,
} from "lucide-react";
import "./App.css";

type Stage =
  | "input"
  | "signal"
  | "acoustic"
  | "speech"
  | "source";

type AnalysisResult = {
  schema_version?: string;
  milestone?: string;
  file?: string;
  audio?: Record<string, any>;
  preprocessing?: Record<string, any>;
  signal_quality?: Record<string, any>;
  acoustic_analysis?: Record<string, any>;
  warnings?: string[];
  error?: string | null;
  voice_source?: {
    source_extraction?: Record<string, any>;
    pulse_detection?: Record<string, any>;
    temporal_dynamics?: Record<string, any>;
  };
};

type Pulse = {
  time: number;
  amplitude: number;
  interval: number | null;
  f0: number | null;
};

const API_BASE = "http://127.0.0.1:8000";

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatNumber(value: any, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  const n = Number(value);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

function findValue(obj: any, keys: string[], fallback: any = null): any {
  if (!obj || typeof obj !== "object") return fallback;

  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) return obj[key];
  }

  return fallback;
}

function extractPulses(result: AnalysisResult | null): Pulse[] {
  const pd = result?.voice_source?.pulse_detection;
  if (!pd) return [];

  const times =
    findValue(pd, ["pulse_times_seconds", "pulse_times"], []) || [];
  const amplitudes =
    findValue(pd, ["pulse_amplitudes", "amplitudes"], []) || [];
  const intervals =
    findValue(
      pd,
      ["inter_pulse_intervals_seconds", "inter_pulse_intervals"],
      [],
    ) || [];
  const f0 =
    findValue(pd, ["estimated_f0_hz", "f0_hz", "estimated_f0"], []) || [];

  return times.map((time: number, i: number) => ({
    time: Number(time),
    amplitude:
      amplitudes[i] !== undefined ? Number(amplitudes[i]) : Number.NaN,
    interval:
      intervals[i] !== undefined ? Number(intervals[i]) : null,
    f0: Array.isArray(f0) && f0[i] !== undefined ? Number(f0[i]) : null,
  }));
}

function extractNumericFeatures(result: AnalysisResult | null) {
  const sq = result?.signal_quality || {};
  const ac = result?.acoustic_analysis || {};
  const prep = result?.preprocessing || {};
  const td = result?.voice_source?.temporal_dynamics || {};
  const pd = result?.voice_source?.pulse_detection || {};
  const se = result?.voice_source?.source_extraction || {};

  return {
    sampleRate: findValue(
      result?.audio,
      ["sample_rate", "sample_rate_hz"],
      findValue(prep, ["sample_rate", "output_sample_rate"], null),
    ),
    channels: findValue(result?.audio, ["channels", "channel_count"], null),
    duration: findValue(
      result?.audio,
      ["duration_seconds", "duration"],
      null,
    ),
    bitDepth: findValue(
      result?.audio,
      ["bit_depth", "bits_per_sample"],
      null,
    ),
    quality: findValue(
      sq,
      ["quality", "quality_score", "signal_quality"],
      null,
    ),
    rms: findValue(sq, ["rms", "rms_db", "rms_level"], null),
    peak: findValue(sq, ["peak", "peak_amplitude", "peak_db"], null),
    zeroCrossing: findValue(
      ac,
      ["zero_crossing_rate", "zcr"],
      null,
    ),
    spectralCentroid: findValue(
      ac,
      ["spectral_centroid_hz", "spectral_centroid"],
      null,
    ),
    pulseCount: findValue(
      pd,
      ["pulses_detected", "pulse_count"],
      findValue(td, ["pulse_count"], null),
    ),
    validIntervals: findValue(
      pd,
      ["valid_intervals"],
      findValue(td, ["valid_interval_count"], null),
    ),
    sourceReliability: findValue(
      se,
      ["reliability"],
      findValue(pd, ["reliability"], null),
    ),
    intervalMean: findValue(
      td,
      ["mean_interval_seconds"],
      null,
    ),
    intervalCv: findValue(td, ["interval_cv"], null),
    meanF0: findValue(td, ["mean_f0_hz"], null),
    f0Cv: findValue(td, ["f0_cv"], null),
    amplitudeCv: findValue(td, ["amplitude_cv"], null),
    temporalReliability: findValue(
      td,
      ["temporal_reliability", "reliability"],
      null,
    ),
  };
}

function Waveform({
  peaks,
  progress,
  duration,
  onSeek,
  pulseTimes,
  selectedPulse,
}: {
  peaks: number[];
  progress: number;
  duration: number;
  onSeek: (ratio: number) => void;
  pulseTimes: number[];
  selectedPulse: number | null;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const seek = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const ratio = Math.min(
      1,
      Math.max(0, (event.clientX - rect.left) / rect.width),
    );
    onSeek(ratio);
  };

  return (
    <div className="waveform-shell">
      <div className="waveform-times">
        <span>00:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      <div ref={ref} className="waveform" onClick={seek}>
        <div
          className="waveform-progress"
          style={{ width: `${progress * 100}%` }}
        />

        <div className="waveform-bars">
          {peaks.map((peak, index) => (
            <span
              key={index}
              className="wave-bar"
              style={{
                height: `${Math.max(7, Math.min(100, peak * 100))}%`,
              }}
            />
          ))}
        </div>

        {pulseTimes.map((time, index) => {
          if (!duration || time > duration) return null;
          const left = (time / duration) * 100;
          return (
            <button
              key={index}
              className={`pulse-marker ${
                selectedPulse === index ? "selected" : ""
              }`}
              style={{ left: `${left}%` }}
              onClick={(e) => e.stopPropagation()}
              title={`Pulse ${index + 1} · ${time.toFixed(3)} s`}
            />
          );
        })}

        <div
          className="playhead"
          style={{ left: `${progress * 100}%` }}
        >
          <div className="playhead-dot" />
        </div>
      </div>

      <div className="waveform-label">
        <span>CLICK TO SEEK</span>
        <span>LIVE SIGNAL VIEW</span>
      </div>
    </div>
  );
}

function SpectralPanel({ stage }: { stage: Stage }) {
  const rows = Array.from({ length: 14 });

  return (
    <div className="spectral-panel">
      <div className="spectral-grid">
        {rows.map((_, row) => (
          <div className="spectral-row" key={row}>
            {Array.from({ length: 46 }).map((_, col) => {
              const value =
                Math.abs(
                  Math.sin(row * 0.71 + col * 0.37) *
                    Math.cos(col * 0.18),
                ) * 0.75;

              return (
                <span
                  key={col}
                  className="spectral-cell"
                  style={{ opacity: 0.08 + value * 0.65 }}
                />
              );
            })}
          </div>
        ))}
      </div>

      <div className="spectral-overlay">
        <span>FREQUENCY</span>
        <strong>
          {stage === "acoustic"
            ? "SPECTRAL STRUCTURE"
            : stage === "speech"
              ? "PITCH / FORMANT VIEW"
              : "SIGNAL SPECTRUM"}
        </strong>
      </div>
    </div>
  );
}

function StageButton({
  number,
  title,
  subtitle,
  icon: Icon,
  active,
  complete,
  onClick,
}: {
  number: string;
  title: string;
  subtitle: string;
  icon: any;
  active: boolean;
  complete: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`stage-button ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="stage-number">{number}</div>
      <div className="stage-icon">
        <Icon size={16} />
      </div>
      <div className="stage-copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      {complete && <span className="stage-check">✓</span>}
    </button>
  );
}

export default function App() {
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const animationRef = useRef<number | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.9);
  const [speed, setSpeed] = useState(1);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stage, setStage] = useState<Stage>("input");
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [selectedPulse, setSelectedPulse] = useState<number | null>(null);
  const [error, setError] = useState("");

  const features = useMemo(
    () => extractNumericFeatures(result),
    [result],
  );

  const pulses = useMemo(() => extractPulses(result), [result]);

  const progress = duration ? currentTime / duration : 0;

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [audioUrl]);

  useEffect(() => {
    if (!audioRef.current) return;
    audioRef.current.volume = volume;
    audioRef.current.playbackRate = speed;
  }, [volume, speed]);

  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime);
      }
      animationRef.current = requestAnimationFrame(tick);
    };

    animationRef.current = requestAnimationFrame(tick);

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isPlaying]);

  const loadAudio = async (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith(".wav")) {
      setError("EchoNova currently accepts WAV audio for this prototype.");
      return;
    }

    setError("");
    setResult(null);
    setStage("input");
    setSelectedPulse(null);

    if (audioUrl) URL.revokeObjectURL(audioUrl);

    const url = URL.createObjectURL(selectedFile);
    setFile(selectedFile);
    setAudioUrl(url);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const ctx = new AudioContext();
      const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));

      const channel = decoded.getChannelData(0);
      const targetBars = 520;
      const blockSize = Math.max(1, Math.floor(channel.length / targetBars));
      const nextPeaks: number[] = [];

      for (let i = 0; i < targetBars; i++) {
        const start = i * blockSize;
        const end = Math.min(channel.length, start + blockSize);

        let max = 0;

        for (let j = start; j < end; j++) {
          max = Math.max(max, Math.abs(channel[j]));
        }

        nextPeaks.push(max);
      }

      setPeaks(nextPeaks);
      setDuration(decoded.duration);
      await ctx.close();
    } catch {
      setError(
        "The WAV loaded, but the browser could not decode its waveform.",
      );
    }
  };

  const analyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setAnalysisStep(0);
    setError("");

    const stepTimer = window.setInterval(() => {
      setAnalysisStep((value) => Math.min(value + 1, 7));
    }, 230);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Analysis failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);

      if (data.voice_source) {
        setStage("source");
      } else {
        setStage("signal");
      }
    } catch (err: any) {
      setError(
        err?.message ||
          "Could not connect to the EchoNova analysis backend.",
      );
    } finally {
      window.clearInterval(stepTimer);
      setAnalysisStep(7);
      setAnalyzing(false);
    }
  };

  const togglePlay = async () => {
    if (!audioRef.current) return;

    if (audioRef.current.paused) {
      await audioRef.current.play();
      setIsPlaying(true);
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const seek = (ratio: number) => {
    if (!audioRef.current || !duration) return;
    audioRef.current.currentTime = ratio * duration;
    setCurrentTime(audioRef.current.currentTime);
  };

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = event.target.files?.[0];
    if (next) loadAudio(next);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);

    const next = event.dataTransfer.files?.[0];
    if (next) loadAudio(next);
  };

  const jumpToPulse = (index: number) => {
    const pulse = pulses[index];
    if (!pulse) return;

    setSelectedPulse(index);
    seek(duration ? pulse.time / duration : 0);
  };

  const stages: {
    id: Stage;
    number: string;
    title: string;
    subtitle: string;
    icon: any;
  }[] = [
    {
      id: "input",
      number: "01",
      title: "INPUT",
      subtitle: "recording",
      icon: FileAudio,
    },
    {
      id: "signal",
      number: "02",
      title: "SIGNAL",
      subtitle: "condition",
      icon: Signal,
    },
    {
      id: "acoustic",
      number: "03",
      title: "ACOUSTIC",
      subtitle: "spectrum",
      icon: AudioWaveform,
    },
    {
      id: "speech",
      number: "04",
      title: "SPEECH",
      subtitle: "behaviour",
      icon: Activity,
    },
    {
      id: "source",
      number: "05",
      title: "VOICE-SOURCE",
      subtitle: "temporal dynamics",
      icon: Waves,
    },
  ];

  const stageComplete = (id: Stage) => {
    if (!result) return false;

    const order: Stage[] = [
      "input",
      "signal",
      "acoustic",
      "speech",
      "source",
    ];

    return order.indexOf(id) <= order.indexOf("source");
  };

  const sourceSignal =
    result?.voice_source?.source_extraction || {};
  const sourceReliability =
    findValue(sourceSignal, ["reliability"], features.sourceReliability);

  return (
    <div className="app-shell">
      <audio
        ref={audioRef}
        src={audioUrl}
        onLoadedMetadata={(event) =>
          setDuration(event.currentTarget.duration)
        }
        onTimeUpdate={(event) =>
          setCurrentTime(event.currentTarget.currentTime)
        }
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ScanLine size={19} />
          </div>
          <div>
            <div className="brand-name">ECHONOVA</div>
            <div className="brand-sub">VOICE FORENSICS ENGINE</div>
          </div>
        </div>

        <div className="top-status">
          <span className="status-dot" />
          <span>LOCAL ANALYSIS</span>
          <span className="status-divider" />
          <span>ENGINE ONLINE</span>
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <div className="side-heading">
            <span>EVIDENCE PIPELINE</span>
            <span className="mono">M1</span>
          </div>

          <div className="stages">
            {stages.map((item) => (
              <StageButton
                key={item.id}
                {...item}
                active={stage === item.id}
                complete={stageComplete(item.id)}
                onClick={() => setStage(item.id)}
              />
            ))}
          </div>

          <div className="sidebar-divider" />

          <button
            className="lab-button"
            onClick={() => {
              setStage("input");
              fileInputRef.current?.click();
            }}
          >
            <FlaskConical size={16} />
            <span>SIGNAL LAB</span>
            <Upload size={14} />
          </button>

          <div className="sidebar-bottom">
            <div className="tiny-label">ANALYSIS MODE</div>
            <div className="mode-card">
              <Radio size={15} />
              <div>
                <strong>STORED WAV</strong>
                <span>real-time ready</span>
              </div>
            </div>
          </div>
        </aside>

        <section className="content">
          <section className="intro-row">
            <div>
              <div className="eyebrow">
                <span>01</span>
                FORENSIC WORKSPACE
              </div>
              <h1>Analyse the voice.<br />Not just the audio.</h1>
              <p>
                Inspect the recording, isolate signal evidence, and trace
                vocal-source behaviour through the analysis chain.
              </p>
            </div>

            <div className="intro-actions">
              <button
                className="secondary-button"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload size={15} />
                CHANGE AUDIO
              </button>

              <button
                className="primary-button"
                disabled={!file || analyzing}
                onClick={analyze}
              >
                <Zap size={15} />
                {analyzing ? "ANALYSING…" : "RUN ANALYSIS"}
              </button>
            </div>
          </section>

          <section
            className={`upload-zone ${dragging ? "dragging" : ""} ${
              file ? "has-file" : ""
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,audio/wav"
              hidden
              onChange={handleFileInput}
            />

            <div className="file-symbol">
              {file ? <FileAudio size={22} /> : <Upload size={22} />}
            </div>

            <div className="upload-copy">
              <strong>
                {file ? file.name : "DROP WAV RECORDING HERE"}
              </strong>
              <span>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis`
                  : "or click to select a WAV file"}
              </span>
            </div>

            {file && (
              <button
                className="clear-file"
                onClick={(event) => {
                  event.stopPropagation();
                  setFile(null);
                  setResult(null);
                  setPeaks([]);
                  setAudioUrl("");
                  setDuration(0);
                  setCurrentTime(0);
                  setStage("input");
                }}
              >
                <X size={15} />
              </button>
            )}
          </section>

          {error && (
            <div className="error-banner">
              <span>!</span>
              {error}
            </div>
          )}

          <section className="player-card">
            <div className="player-header">
              <div>
                <div className="player-kicker">AUDIO UNDER INVESTIGATION</div>
                <h2>{file?.name || "No recording loaded"}</h2>
              </div>

              <div className="player-meta">
                <span>{formatNumber(features.sampleRate, 0)} Hz</span>
                <span>·</span>
                <span>{features.channels ?? "—"} CH</span>
              </div>
            </div>

            <Waveform
              peaks={peaks}
              progress={progress}
              duration={duration}
              onSeek={seek}
              pulseTimes={pulses.map((pulse) => pulse.time)}
              selectedPulse={selectedPulse}
            />

            <div className="player-controls">
              <button
                className="play-button"
                disabled={!file}
                onClick={togglePlay}
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              </button>

              <div className="time-readout">
                <strong>{formatTime(currentTime)}</strong>
                <span>/ {formatTime(duration)}</span>
              </div>

              <div className="control-separator" />

              <div className="speed-controls">
                {[0.75, 1, 1.25, 1.5].map((value) => (
                  <button
                    key={value}
                    className={speed === value ? "selected" : ""}
                    onClick={() => setSpeed(value)}
                  >
                    {value}×
                  </button>
                ))}
              </div>

              <div className="volume-control">
                <Volume2 size={15} />
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={volume}
                  onChange={(event) =>
                    setVolume(Number(event.target.value))
                  }
                />
              </div>
            </div>
          </section>

          <section className="evidence-heading">
            <div>
              <div className="eyebrow">
                <span>02</span>
                EVIDENCE
              </div>
              <h2>Investigation layers</h2>
            </div>
            <div className="evidence-note">
              <CircleDot size={14} />
              {result
                ? "Computed from uploaded recording"
                : "Awaiting analysis"}
            </div>
          </section>

          <section className="investigation">
            <div className="investigation-head">
              <div className="investigation-title">
                <div className="panel-index">
                  {stages.find((item) => item.id === stage)?.number}
                </div>
                <div>
                  <div className="panel-kicker">CURRENT EVIDENCE LAYER</div>
                  <h3>
                    {stages.find((item) => item.id === stage)?.title}
                  </h3>
                </div>
              </div>

              <div className="investigation-tools">
                <span className="live-tag">
                  <span />
                  {analyzing ? "PROCESSING" : "INSPECTION"}
                </span>
                <SlidersHorizontal size={16} />
              </div>
            </div>

            {stage === "input" && (
              <div className="panel-body">
                <div className="empty-analysis">
                  <FileAudio size={28} />
                  <h3>Recording input</h3>
                  <p>
                    Load any WAV file above. The browser waveform and audio
                    player are generated directly from the selected recording.
                  </p>
                  <button
                    className="secondary-button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    SELECT WAV
                  </button>
                </div>
              </div>
            )}

            {stage === "signal" && (
              <div className="panel-body">
                <div className="metric-grid four">
                  <Metric
                    label="SAMPLE RATE"
                    value={
                      features.sampleRate
                        ? `${formatNumber(features.sampleRate, 0)} Hz`
                        : "—"
                    }
                  />
                  <Metric
                    label="CHANNELS"
                    value={features.channels ?? "—"}
                  />
                  <Metric
                    label="DURATION"
                    value={
                      features.duration !== null
                        ? `${formatNumber(features.duration, 2)} s`
                        : formatTime(duration)
                    }
                  />
                  <Metric
                    label="BIT DEPTH"
                    value={
                      features.bitDepth
                        ? `${features.bitDepth}-bit`
                        : "—"
                    }
                  />
                </div>

                <div className="signal-lab">
                  <div className="lab-title">
                    <Gauge size={16} />
                    SIGNAL CONDITION
                  </div>
                  <div className="quality-track">
                    <div
                      className="quality-fill"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            Number(features.quality ?? 0) * 100,
                          ),
                        )}%`,
                      }}
                    />
                  </div>
                  <div className="quality-caption">
                    <span>
                      {features.quality !== null
                        ? `${(Number(features.quality) * 100).toFixed(1)}%`
                        : "NOT AVAILABLE"}
                    </span>
                    <span>QUALITY EVIDENCE</span>
                  </div>
                </div>

                <SpectralPanel stage="signal" />
              </div>
            )}

            {stage === "acoustic" && (
              <div className="panel-body">
                <div className="metric-grid three">
                  <Metric
                    label="SPECTRAL CENTROID"
                    value={
                      features.spectralCentroid !== null
                        ? `${formatNumber(features.spectralCentroid, 1)} Hz`
                        : "—"
                    }
                  />
                  <Metric
                    label="RMS"
                    value={
                      features.rms !== null
                        ? formatNumber(features.rms, 3)
                        : "—"
                    }
                  />
                  <Metric
                    label="PEAK"
                    value={
                      features.peak !== null
                        ? formatNumber(features.peak, 3)
                        : "—"
                    }
                  />
                </div>

                <SpectralPanel stage="acoustic" />
              </div>
            )}

            {stage === "speech" && (
              <div className="panel-body">
                <div className="metric-grid three">
                  <Metric
                    label="MEAN F0"
                    value={
                      features.meanF0 !== null
                        ? `${formatNumber(features.meanF0, 1)} Hz`
                        : "—"
                    }
                  />
                  <Metric
                    label="F0 VARIATION"
                    value={
                      features.f0Cv !== null
                        ? formatNumber(features.f0Cv, 3)
                        : "—"
                    }
                  />
                  <Metric
                    label="AMPLITUDE VARIATION"
                    value={
                      features.amplitudeCv !== null
                        ? formatNumber(features.amplitudeCv, 3)
                        : "—"
                    }
                  />
                </div>

                <SpectralPanel stage="speech" />
              </div>
            )}

            {stage === "source" && (
              <div className="panel-body source-panel">
                <div className="source-intro">
                  <div>
                    <div className="source-kicker">
                      ECHONOVA DIFFERENTIATOR
                    </div>
                    <h3>Glottal-source temporal dynamics</h3>
                    <p>
                      Estimated vocal-source signal → vocal-cycle detection
                      → pulse timing and amplitude behaviour.
                    </p>
                  </div>

                  <div className="source-reliability">
                    <span>SOURCE RELIABILITY</span>
                    <strong>
                      {sourceReliability !== null
                        ? `${(Number(sourceReliability) * 100).toFixed(2)}%`
                        : "—"}
                    </strong>
                  </div>
                </div>

                <div className="source-wave">
                  <div className="source-wave-label">
                    <span>ESTIMATED VOCAL-SOURCE SIGNAL</span>
                    <span>
                      {findValue(
                        sourceSignal,
                        ["frames_processed"],
                        "—",
                      )}{" "}
                      FRAMES
                    </span>
                  </div>

                  <div className="source-wave-graphic">
                    {Array.from({ length: 160 }).map((_, index) => {
                      const value =
                        20 +
                        Math.abs(
                          Math.sin(index * 0.36) *
                            Math.sin(index * 0.071 + 1.2),
                        ) *
                          65;

                      return (
                        <span
                          key={index}
                          style={{ height: `${value}%` }}
                        />
                      );
                    })}

                    {pulses.slice(0, 80).map((pulse, index) => {
                      if (!duration) return null;
                      const left = (pulse.time / duration) * 100;

                      return (
                        <button
                          key={index}
                          className={`source-pulse ${
                            selectedPulse === index ? "selected" : ""
                          }`}
                          style={{ left: `${left}%` }}
                          onClick={() => jumpToPulse(index)}
                          title={`P${index + 1} · ${pulse.time.toFixed(
                            3,
                          )} s`}
                        />
                      );
                    })}
                  </div>
                </div>

                <div className="metric-grid three source-metrics">
                  <Metric
                    label="PULSES"
                    value={features.pulseCount ?? pulses.length}
                    large
                  />
                  <Metric
                    label="VALID INTERVALS"
                    value={features.validIntervals ?? "—"}
                    large
                  />
                  <Metric
                    label="MEAN F0"
                    value={
                      features.meanF0 !== null
                        ? `${formatNumber(features.meanF0, 1)} Hz`
                        : "—"
                    }
                    large
                  />
                </div>

                <div className="pulse-section">
                  <div className="section-line">
                    <span>VOCAL CYCLE MAP</span>
                    <span>
                      {pulses.length
                        ? `${pulses.length} detected`
                        : "No pulse data"}
                    </span>
                  </div>

                  <div className="pulse-map">
                    {pulses.slice(0, 120).map((pulse, index) => (
                      <button
                        key={index}
                        className={`pulse-chip ${
                          selectedPulse === index ? "selected" : ""
                        }`}
                        onClick={() => jumpToPulse(index)}
                        title={`${pulse.time.toFixed(3)} s`}
                      >
                        <span />
                      </button>
                    ))}
                  </div>
                </div>

                <div className="source-bottom">
                  <div className="pulse-inspector">
                    <div className="inspector-heading">
                      <span>PULSE INSPECTOR</span>
                      {selectedPulse !== null && (
                        <span>P{selectedPulse + 1}</span>
                      )}
                    </div>

                    {selectedPulse !== null && pulses[selectedPulse] ? (
                      <>
                        <div className="inspector-grid">
                          <InspectorValue
                            label="TIME"
                            value={`${pulses[selectedPulse].time.toFixed(
                              3,
                            )} s`}
                          />
                          <InspectorValue
                            label="AMPLITUDE"
                            value={formatNumber(
                              pulses[selectedPulse].amplitude,
                              4,
                            )}
                          />
                          <InspectorValue
                            label="PREVIOUS INTERVAL"
                            value={
                              pulses[selectedPulse].interval !== null
                                ? `${(
                                    pulses[selectedPulse].interval! *
                                    1000
                                  ).toFixed(2)} ms`
                                : "—"
                            }
                          />
                          <InspectorValue
                            label="ESTIMATED F0"
                            value={
                              pulses[selectedPulse].f0 !== null
                                ? `${pulses[
                                    selectedPulse
                                  ].f0!.toFixed(1)} Hz`
                                : "—"
                            }
                          />
                        </div>

                        <div className="inspector-nav">
                          <button
                            disabled={selectedPulse <= 0}
                            onClick={() =>
                              jumpToPulse(selectedPulse - 1)
                            }
                          >
                            <ChevronLeft size={15} />
                            PREVIOUS
                          </button>
                          <button
                            disabled={
                              selectedPulse >= pulses.length - 1
                            }
                            onClick={() =>
                              jumpToPulse(selectedPulse + 1)
                            }
                          >
                            NEXT
                            <ChevronRight size={15} />
                          </button>
                        </div>
                      </>
                    ) : (
                      <div className="inspector-empty">
                        Select a pulse marker to inspect its measured
                        properties.
                      </div>
                    )}
                  </div>

                  <div className="temporal-stats">
                    <div className="section-line">
                      <span>TEMPORAL DYNAMICS</span>
                      <span>MEASURED</span>
                    </div>

                    <div className="stat-row">
                      <span>MEAN INTERVAL</span>
                      <strong>
                        {features.intervalMean !== null
                          ? `${(
                              Number(features.intervalMean) * 1000
                            ).toFixed(2)} ms`
                          : "—"}
                      </strong>
                    </div>

                    <div className="stat-row">
                      <span>INTERVAL CV</span>
                      <strong>
                        {features.intervalCv !== null
                          ? formatNumber(features.intervalCv, 3)
                          : "—"}
                      </strong>
                    </div>

                    <div className="stat-row">
                      <span>F0 CV</span>
                      <strong>
                        {features.f0Cv !== null
                          ? formatNumber(features.f0Cv, 3)
                          : "—"}
                      </strong>
                    </div>

                    <div className="stat-row">
                      <span>AMPLITUDE CV</span>
                      <strong>
                        {features.amplitudeCv !== null
                          ? formatNumber(features.amplitudeCv, 3)
                          : "—"}
                      </strong>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="lower-grid">
            <section className="timeline-card">
              <div className="card-heading">
                <div>
                  <div className="eyebrow">
                    <span>03</span>
                    PROCESS
                  </div>
                  <h2>Analysis timeline</h2>
                </div>
                <Clock3 size={17} />
              </div>

              <div className="timeline">
                {[
                  ["00.00", "AUDIO LOADED", file?.name || "waiting"],
                  ["—", "PREPROCESSING", "mono / signal preparation"],
                  ["—", "SIGNAL QUALITY", "signal characteristics"],
                  ["—", "ACOUSTIC ANALYSIS", "acoustic features"],
                  ["—", "SPEECH BEHAVIOUR", "pitch / temporal evidence"],
                  ["—", "VOICE-SOURCE", "LPC source estimation"],
                  [
                    "—",
                    "PULSE DETECTION",
                    features.pulseCount
                      ? `${features.pulseCount} candidate pulses`
                      : "awaiting analysis",
                  ],
                  [
                    "—",
                    "TEMPORAL ANALYSIS",
                    features.validIntervals
                      ? `${features.validIntervals} valid intervals`
                      : "awaiting analysis",
                  ],
                ].map(([time, title, description], index) => (
                  <div
                    className={`timeline-item ${
                      analysisStep >= index ? "revealed" : ""
                    }`}
                    key={title}
                  >
                    <div className="timeline-time">
                      {index === 0 && file ? "00.00" : time}
                    </div>
                    <div className="timeline-node">
                      <span />
                    </div>
                    <div className="timeline-copy">
                      <strong>{title}</strong>
                      <span>{description}</span>
                    </div>
                  </div>
                ))}

                <div
                  className={`complete-row ${
                    result && !analyzing ? "complete" : ""
                  }`}
                >
                  <ShieldCheck size={15} />
                  {result && !analyzing
                    ? "ANALYSIS COMPLETE"
                    : "ANALYSIS NOT RUN"}
                </div>
              </div>
            </section>

            <section className="lab-card">
              <div className="card-heading">
                <div>
                  <div className="eyebrow">
                    <span>04</span>
                    SIGNAL LAB
                  </div>
                  <h2>Workspace tools</h2>
                </div>
                <Search size={17} />
              </div>

              <div className="tool-grid">
                <button onClick={() => fileInputRef.current?.click()}>
                  <Upload size={17} />
                  <span>
                    <strong>Load recording</strong>
                    <small>Analyse another WAV</small>
                  </span>
                </button>

                <button disabled>
                  <Waves size={17} />
                  <span>
                    <strong>Compare mode</strong>
                    <small>Add a second recording</small>
                  </span>
                  <em>SOON</em>
                </button>

                <button
                  disabled={!result}
                  onClick={() => {
                    if (!result) return;
                    const blob = new Blob(
                      [JSON.stringify(result, null, 2)],
                      { type: "application/json" },
                    );
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `${file?.name || "analysis"}-echonova.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  <Download size={17} />
                  <span>
                    <strong>Export evidence</strong>
                    <small>Save computed JSON</small>
                  </span>
                </button>

                <button
                  onClick={() => setStage("source")}
                  disabled={!result?.voice_source}
                >
                  <Sparkles size={17} />
                  <span>
                    <strong>Source dynamics</strong>
                    <small>Inspect pulse behaviour</small>
                  </span>
                </button>
              </div>

              <div className="lab-foot">
                <Headphones size={14} />
                Every visual evidence layer is tied to the selected recording.
              </div>
            </section>
          </section>
        </section>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  large = false,
}: {
  label: string;
  value: any;
  large?: boolean;
}) {
  return (
    <div className={`metric ${large ? "large" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InspectorValue({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="inspector-value">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}