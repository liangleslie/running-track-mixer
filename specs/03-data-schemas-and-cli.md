# 03 — Data Schemas & CLI Specification

## 1. Per-Track Metadata Schema (`work/<track_id>/metadata.json`)

Each track produces a persistent JSON file capturing:
- **Step 3 State (`pre_stretch`)**: Raw file properties, detected active beat window, trimmed non-music margins, native BPM, harmonic factor, and planned stretch ratio.
- **Step 5 State (`post_stretch`)**: Verified BPM after `ffmpeg` `rubberband` time-stretching, Beat 0 phase anchor offset, phase alignment accuracy metrics, and post-stretch beat timestamps.

### JSON Example & Schema Structure

```json
{
  "schema_version": "1.0",
  "track_id": "01_daft_punk_harder_better_a1b2c3d4",
  "source": {
    "file_path": "/path/to/input/01_daft_punk_harder_better.mp3",
    "file_name": "01_daft_punk_harder_better.mp3",
    "sha256_prefix": "a1b2c3d4",
    "original_duration_sec": 224.62,
    "original_sample_rate": 44100,
    "original_channels": 2
  },
  "target_config": {
    "global_target_bpm": 170.0,
    "harmonic_factor": 0.5,
    "track_target_bpm": 85.0,
    "metronome_clicks_per_song_beat": 2.0
  },
  "pre_stretch": {
    "librosa_coarse_bpm": 123.046,
    "regression_native_bpm": 123.008,
    "native_beat_period_sec": 0.487773,
    "total_detected_beats_raw": 448,
    "active_window": {
      "start_beat_index": 8,
      "end_beat_index": 436,
      "active_beat_count": 428,
      "first_active_beat_sec": 4.115,
      "last_active_beat_sec": 212.882,
      "trim_start_sec": 4.100,
      "trim_end_sec": 212.882,
      "stripped_intro_sec": 4.100,
      "stripped_outro_sec": 11.738
    },
    "planned_stretch_ratio": 0.691012,
    "beat_timestamps_active_sec": [4.115, 4.603, 5.091, "..."]
  },
  "post_stretch": {
    "stretched_wav_path": "work/01_daft_punk_harder_better_a1b2c3d4/stretched.wav",
    "stretched_duration_sec": 302.133,
    "applied_stretch_ratio": 0.691012,
    "correction_passes": 0,
    "verified_regression_bpm": 85.004,
    "bpm_error": 0.004,
    "first_beat_offset_sec": 0.0152,
    "sync_diagnostics": {
      "mean_abs_phase_error_ms": 2.41,
      "max_abs_phase_error_ms": 6.85,
      "is_within_tolerance": true
    },
    "beat_timestamps_sec": [0.0152, 0.7211, 1.4270, "..."]
  },
  "status": "verified"
}
```

### Field Rules
- `status`: Enum of `"analyzed"` (after Step 3), `"verified"` (after Step 5 passes tolerance), or `"failed"`.
- `harmonic_factor`: $k \in \{0.5, 1.0, 2.0\}$. When `0.5`, `track_target_bpm = 0.5 * global_target_bpm` and `metronome_clicks_per_song_beat = 2.0`.
- `beat_timestamps_active_sec`: Full array of active beat timestamps (in seconds relative to the original file start) prior to trimming and stretching.
- `post_stretch.beat_timestamps_sec`: Full array of verified beat timestamps (in seconds relative to `stretched.wav` start).

---

## 2. Global Mashup Manifest Schema (`output/mashup_manifest.json`)

Written alongside `output/mashup.mp3` at the end of Step 7 to provide a complete cue sheet and audit trail of the mashup.

```json
{
  "schema_version": "1.0",
  "created_at": "2026-10-29T07:00:00+08:00",
  "output_mp3_path": "output/mashup.mp3",
  "sample_rate": 44100,
  "target_bpm": 170.0,
  "metronome_beat_period_sec": 0.352941,
  "crossfade_beats": 8,
  "lead_in_beats": 8,
  "lead_out_beats": 8,
  "metronome_gain_db": -6.0,
  "total_metronome_beats": 4820,
  "total_duration_sec": 1701.176,
  "tracks": [
    {
      "order": 1,
      "track_id": "01_daft_punk_harder_better_a1b2c3d4",
      "file_name": "01_daft_punk_harder_better.mp3",
      "native_bpm": 123.008,
      "harmonic_factor": 0.5,
      "target_track_bpm": 85.0,
      "verified_bpm": 85.004,
      "stretch_ratio": 0.691012,
      "song_beats": 428,
      "metronome_beats_spanned": 856,
      "start_metronome_beat": 8,
      "end_metronome_beat": 864,
      "start_time_sec": 2.824,
      "end_time_sec": 304.941,
      "crossfade_in_beats": 0,
      "crossfade_out_beats": 8,
      "mean_abs_phase_error_ms": 2.41
    },
    {
      "order": 2,
      "track_id": "02_pendulum_watercolour_9f8e7d6c",
      "file_name": "02_pendulum_watercolour.m4a",
      "native_bpm": 174.002,
      "harmonic_factor": 1.0,
      "target_track_bpm": 170.0,
      "verified_bpm": 169.998,
      "stretch_ratio": 0.976989,
      "song_beats": 640,
      "metronome_beats_spanned": 640,
      "start_metronome_beat": 856,
      "end_metronome_beat": 1496,
      "start_time_sec": 302.118,
      "end_time_sec": 528.000,
      "crossfade_in_beats": 8,
      "crossfade_out_beats": 8,
      "mean_abs_phase_error_ms": 1.85
    }
  ]
}
```

---

## 3. CLI Interface Specification

All commands are executed via `uv run rtmix`.

### Primary Command: `uv run rtmix build`

```bash
uv run rtmix build \
  --input-dir ./tracks \
  --output ./output/mashup.mp3 \
  --bpm 170 \
  --crossfade-beats 8 \
  --lead-in-beats 8 \
  --lead-out-beats 8 \
  --metronome-gain-db -6.0 \
  --work-dir ./work
```

#### CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-i, --input-dir` | `Path` | *Required* | Directory containing source audio tracks (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`), sorted lexicographically by filename. |
| `-o, --output` | `Path` | `output/mashup.mp3` | Output path for the final `.mp3` mashup. Writes `mashup_manifest.json` in the same directory. |
| `-b, --bpm` | `float` | `180.0` | Target running cadence in beats (steps) per minute (`60.0` to `240.0`). |
| `--crossfade-beats` | `int` | `8` | Beat-synced overlap between adjacent tracks in global metronome beats (must be an even integer $\ge 0$; `0` = beat-matched cut). |
| `--lead-in-beats` | `int` | `8` | Number of solo metronome clicks before Track 1 Beat 0 starts. |
| `--lead-out-beats` | `int` | `8` | Number of solo metronome clicks after the final track's last beat. |
| `--metronome-gain-db` | `float` | `-6.0` | Peak amplitude of metronome clicks in dBFS (`-30.0` to `0.0`; use `--no-metronome` to mute). |
| `--harmonic-factors` | `str` | `"0.5,1.0,2.0"` | Comma-separated binary tempo multipliers/factors relative to `--bpm`. |
| `--min-stretch` | `float` | `0.85` | Minimum allowed stretch ratio before skipping dead-zone tracks. |
| `--max-stretch` | `float` | `1.35` | Maximum allowed stretch ratio before skipping dead-zone tracks. |
| `--work-dir` | `Path` | `./work` | Directory for per-track cache (`metadata.json` and intermediate `stretched.wav`). |
| `-j, --workers` | `int` | `os.cpu_count() // 2` | Number of parallel worker processes for per-track analysis and time-stretching. |
| `-n, --limit` | `int` | `None` | Process first `N` tracks from `--input-dir`. |
| `-r, -random, --random` | `int` | `None` | Randomly pick `N` valid tracks for the mashup (automatically draws replacements if any pick is dropped as dead-zone). |
| `--seed` | `int` | `None` | Optional random seed for reproducible `--random` selection. |
| `-f, -force, --force` | `bool` | `False` | Force inclusion of dead-zone BPM tracks outside `[--min-stretch, --max-stretch]`. |
| `--rebuild` | `bool` | `False` | Ignore cached `stretched.wav` / `metadata.json` and re-process all tracks from scratch. |

### Utility Subcommands
- `uv run rtmix analyze <file_or_dir> --bpm 170`: Runs only Steps 1–3 (active window detection + BPM regression + harmonic target calculation) and prints a summary table + writes `metadata.json` without running `ffmpeg` time-stretching.
- `uv run rtmix verify <work_dir>`: Inspects existing `metadata.json` files and reports any tracks with high phase error or extreme stretch ratios ($> \pm 25\%$).

---

## 4. Edge Cases & Error Handling

1. **Unstable / Beatless Tracks (Ambient, Classical, Live Unquantized Drums)**:
   - If fewer than `16` active beats are found or Theil-Sen IBI standard deviation exceeds `8%` of the median beat period, log a clear warning and skip the track (or abort if `--strict` is passed) rather than producing an out-of-sync trainwreck.
2. **Short Tracks vs. Crossfade Length**:
   - If a track's `metronome_beats_spanned` is smaller than $2 \times C_{\text{beats}}$, clamp `crossfade_beats` for that transition to $\lfloor M_i / 4 \rfloor$ (rounded down to an even number).
3. **Corrupt / Unsupported Files**:
   - Non-audio files in `--input-dir` (e.g. `.DS_Store`, `cover.jpg`, `.json`) are silently ignored; only extensions in `{.mp3, .m4a, .aac, .wav, .flac, .ogg, .opus}` are ingested.
