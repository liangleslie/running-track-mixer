# Running Tracks Mixer (`rtmix`)

`rtmix` turns a folder of audio files (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`) into a single continuous, tempo-locked `.mp3` mashup overlaid on a sample-accurate metronome track for running cadence (default `180.0 BPM`).

---

## How It Works (7-Step Pipeline)

1. **Strip Non-Music Intro/Outro**: Detects continuous musical body (`0.8s` smoothed RMS `>= 5%` of track median with `2.2s` gap closing) to strip leading/trailing silence and detached MV skits/credits while preserving delicate piano/guitar/vocal intros. Snaps trim boundaries to the studio drum beat grid.
2. **High-Precision Tempo & Beat Grid Detection**: Separates percussive transients via HPSS (`librosa.effects.hpss`) and computes native tempo to `< 0.05 BPM` precision via 48-beat sliding-window Theil-Sen regression (`scipy.stats.theilslopes`), avoiding `librosa.beat.beat_track`'s coarse quantized BPM bins and drumless-bridge artifacts.
3. **Pre-Stretch Metadata JSON**: Writes `work/<track_id>/metadata.json` (`status: "analyzed"`) with native BPM, beat timestamps, trim points, and target harmonic factor (`0.5x`, `1.0x`, or `2.0x`).
4. **Pitch-Preserving Stretch & Loudness Normalization**: Runs a single-pass `ffmpeg` filtergraph (`atrim` -> `rubberband=tempo=R:transients=crisp:detector=compound:phase=laminar` -> EBU R128 `loudnorm=I=-14.0:TP=-1.5:LRA=11.0`) to uncompressed 44.1 kHz 32-bit float PCM `work/<track_id>/stretched.wav`.
5. **Post-Stretch Verification & Phase Lock**: Re-analyzes `stretched.wav` to verify achieved BPM (`|err| <= 0.80 BPM`, retrying once with a micro-corrected ratio if needed) and computes `first_beat_offset_sec` + beat phase error diagnostics (`status: "verified"`).
6. **Zero-Drift Sample-Accurate Metronome**: Synthesizes an 850 Hz woodblock click track (`-6.0 dBFS` default) stamped at exact sample coordinates `round(m * (60.0 / target_bpm) * 44100)` across the entire mashup duration (`0.0 ms` cumulative drift over 60+ minutes).
7. **Beat-Synced Crossfade Assembly**: Aligns each track's Beat 0 to its scheduled global metronome click (`sample_offset = grid_click_sample - round(first_beat_offset_sec * 44100)`), applies equal-power quarter-cycle sine/cosine crossfades over `crossfade_beats`, mixes onto the metronome backing, and encodes `output/mashup.mp3` + `output/mashup_manifest.json`.

---

## Requirements

- **Linux / macOS** with `ffmpeg` and `ffprobe` compiled with `librubberband` and `libmp3lame`
- **Python 3.12+** managed with [`uv`](https://docs.astral.sh/uv/)

### Install Dependencies

```bash
uv sync
```

---

## Quickstart

### Build a mashup from a folder (default `180.0 BPM`)

```bash
uv run rtmix build -i tracks -o output/mashup.mp3
```

### Randomly pick `N` valid tracks from a large library

Automatically draws replacements if any randomly picked song falls in the tempo dead zone:

```bash
uv run rtmix build -i tracks2 -r 10 -o output/mashup_10.mp3

# Reproducible random pick with a seed
uv run rtmix build -i tracks2 --random 5 --seed 7 -o output/mashup_5.mp3
```

### Force inclusion of dead-zone BPM tracks (`~110–132 BPM`)

By default, tracks requiring a stretch ratio outside `[0.85x, 1.35x]` are skipped with a `[WARN] [DEAD-ZONE SKIP]` message. Pass `-f` / `-force` / `--force` to include them anyway:

```bash
uv run rtmix build -i tracks -f -o output/mashup_forced.mp3
```

### Analyze tracks only (Steps 1–3)

```bash
uv run rtmix analyze tracks2 -r 20
```

### Verify cached tracks (Step 5 diagnostics)

```bash
uv run rtmix verify work
```

---

## Harmonic Factor & Dead-Zone Filtering

Stretching a `90 BPM` ballad to `180 BPM` (`2.0x`) destroys musicality. `rtmix` selects the best binary harmonic factor $k \in \{0.5, 1.0, 2.0\}$ so every beat lands on an integer multiple of your footsteps:

| Native BPM (at `180 BPM` target) | Harmonic Factor `k` | Track Target BPM | Footsteps per Beat | Status |
| :--- | :--- | :--- | :--- | :--- |
| `76.5 – 115.0 BPM` | `0.5x` (Half-time) | `90.0 BPM` | `2` steps / beat | Kept (`0.85x – 1.28x`) |
| `115.0 – 133.3 BPM` | — | — | — | **Dead Zone** (Dropped unless `-force`) |
| `133.3 – 211.8 BPM` | `1.0x` (1:1 Lock) | `180.0 BPM` | `1` step / beat | Kept (`0.85x – 1.35x`) |

---

## CLI Reference (`rtmix build`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-i`, `--input-dir <PATH>` | *(required)* | Input directory or audio file (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`) |
| `-o`, `--output <PATH>` | `output/mashup.mp3` | Output mashup MP3 path (manifest written alongside) |
| `-b`, `--bpm <FLOAT>` | `180.0` | Target running cadence BPM |
| `-r`, `-random`, `--random <N>` | `None` | Randomly pick `N` valid tracks (auto-replaces dropped dead-zone tracks) |
| `--seed <INT>` | `None` | Random seed for reproducible `--random` selection |
| `-n`, `--limit <N>` | `None` | Process only the first `N` tracks in sorted order |
| `-f`, `-force`, `--force` | `False` | Force inclusion of dead-zone BPM tracks outside `[--min-stretch, --max-stretch]` |
| `--min-stretch <FLOAT>` | `0.85` | Minimum allowed stretch ratio before dropping a track (`-15%` speed) |
| `--max-stretch <FLOAT>` | `1.35` | Maximum allowed stretch ratio before dropping a track (`+35%` speed) |
| `--harmonic-factors <LIST>` | `0.5,1.0,2.0` | Allowed harmonic multipliers relative to target BPM |
| `--crossfade-beats <INT>` | `8` | Beat-synced equal-power crossfade length in metronome beats |
| `--lead-in-beats <INT>` | `8` | Solo metronome clicks before Track 1 starts |
| `--lead-out-beats <INT>` | `8` | Solo metronome clicks after the final track ends |
| `--metronome-gain-db <FLOAT>` | `-6.0` | Metronome click peak level in dBFS |
| `--no-metronome` | `False` | Export beat-locked mashup without the metronome click overlay |
| `-j`, `--workers <INT>` | `min(16, cpu // 2)` | Number of parallel worker processes for per-track analysis & stretching |
| `--work-dir <PATH>` | `work` | Cache directory for per-track `metadata.json` and `stretched.wav` |
| `--rebuild` | `False` | Ignore cached `stretched.wav` in `--work-dir` and re-process from scratch |
| `--strict` | `False` | Abort pipeline immediately if any track fails verification |

---

## Project Layout

```text
running-tracks-mixer/
├── pyproject.toml
├── README.md
├── specs/
│   ├── 01-overview-and-architecture.md
│   ├── 02-audio-dsp-pipeline.md
│   └── 03-data-schemas-and-cli.md
├── src/rtmix/
│   ├── cli.py          # CLI argument parser & subcommands (build, analyze, verify)
│   ├── models.py       # TrackMetadata & MashupManifest dataclasses + JSON I/O
│   ├── analyzer.py     # Steps 1-3: Musical bounds, HPSS beat tracking, Theil-Sen BPM
│   ├── processor.py    # Steps 4-5: FFmpeg rubberband + loudnorm + phase verification
│   ├── metronome.py    # Step 6: Zero-drift sample-accurate woodblock click synthesizer
│   ├── mixer.py        # Step 7: Beat-grid scheduler, equal-power crossfades, MP3 export
│   └── pipeline.py     # Multi-process worker pool, caching, and --random sampling
└── tests/
    └── test_self_check.py
```

### Run Self-Checks

```bash
uv run python tests/test_self_check.py
```
