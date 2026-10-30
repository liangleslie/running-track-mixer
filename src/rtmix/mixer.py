"""Global beat-grid scheduler, beat-synced equal-power crossfader, and MP3 master exporter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from rtmix.analyzer import SAMPLE_RATE
from rtmix.metronome import beat_to_sample_index, render_metronome_track
from rtmix.models import MashupManifest, TrackCueEntry, TrackMetadata


def assemble_mashup(
    tracks: list[TrackMetadata],
    output_mp3_path: Path,
    target_bpm: float,
    crossfade_beats: int = 8,
    lead_in_beats: int = 8,
    lead_out_beats: int = 8,
    metronome_gain_db: float = -6.0,
    enable_metronome: bool = True,
) -> MashupManifest:
    """Run Steps 6 and 7: schedule tracks on global beat grid, crossfade, overlay metronome, export MP3."""
    if not tracks:
        raise ValueError("Cannot assemble mashup: no verified tracks provided.")

    # Ensure crossfade_beats is even so half-time (0.5x) tracks align on whole musical beats
    if crossfade_beats % 2 != 0:
        crossfade_beats += 1

    # 1. Calculate global beat schedule
    cue_entries: list[TrackCueEntry] = []
    m_cursor = lead_in_beats
    n_tracks = len(tracks)

    for idx, tm in enumerate(tracks):
        if tm.post_stretch is None:
            raise ValueError(f"Track {tm.track_id} is missing post_stretch verification data.")

        song_beats = tm.pre_stretch.active_window.active_beat_count
        k = tm.target_config.harmonic_factor
        metro_beats_spanned = int(round(song_beats / k))

        # Clamp crossfade on short tracks
        xfade_in = 0 if idx == 0 else min(crossfade_beats, (metro_beats_spanned // 4) * 2)
        xfade_out = (
            0 if idx == n_tracks - 1 else min(crossfade_beats, (metro_beats_spanned // 4) * 2)
        )

        m_start = m_cursor
        m_end = m_start + metro_beats_spanned
        t_start = m_start * (60.0 / target_bpm)
        t_end = m_end * (60.0 / target_bpm)

        cue_entries.append(
            TrackCueEntry(
                order=idx + 1,
                track_id=tm.track_id,
                file_name=tm.source.file_name,
                native_bpm=tm.pre_stretch.regression_native_bpm,
                harmonic_factor=k,
                target_track_bpm=tm.target_config.track_target_bpm,
                verified_bpm=tm.post_stretch.verified_regression_bpm,
                stretch_ratio=tm.post_stretch.applied_stretch_ratio,
                song_beats=song_beats,
                metronome_beats_spanned=metro_beats_spanned,
                start_metronome_beat=m_start,
                end_metronome_beat=m_end,
                start_time_sec=round(t_start, 3),
                end_time_sec=round(t_end, 3),
                crossfade_in_beats=xfade_in,
                crossfade_out_beats=xfade_out,
                mean_abs_phase_error_ms=tm.post_stretch.sync_diagnostics.mean_abs_phase_error_ms,
            )
        )
        m_cursor = m_end - xfade_out

    total_metro_beats = cue_entries[-1].end_metronome_beat + lead_out_beats
    total_samples = beat_to_sample_index(total_metro_beats, target_bpm, sr=SAMPLE_RATE)

    # 2. Allocate master buffer (initialize with metronome or silence)
    if enable_metronome:
        master_buf = render_metronome_track(
            total_samples=total_samples,
            total_metronome_beats=total_metro_beats,
            target_bpm=target_bpm,
            gain_db=metronome_gain_db,
            sr=SAMPLE_RATE,
        )
    else:
        master_buf = np.zeros((total_samples, 2), dtype=np.float32)

    # 3. Place each stretched track onto master_buf with beat-synced equal-power crossfades
    for tm, cue in zip(tracks, cue_entries, strict=True):
        assert tm.post_stretch is not None
        wav_path = Path(tm.post_stretch.stretched_wav_path)
        audio, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if file_sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz in {wav_path}, got {file_sr}Hz")
        if audio.shape[1] == 1:
            audio = np.column_stack((audio[:, 0], audio[:, 0]))
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        # Sample coordinate of Beat 0 on the global grid vs inside `audio`
        grid_beat0_sample = beat_to_sample_index(cue.start_metronome_beat, target_bpm, sr=SAMPLE_RATE)
        grid_end_sample = beat_to_sample_index(cue.end_metronome_beat, target_bpm, sr=SAMPLE_RATE)
        first_beat_sample_offset = int(round(tm.post_stretch.first_beat_offset_sec * SAMPLE_RATE))

        # Slice audio from its Beat 0 up to the exact span of the track on the beat grid
        span_samples = grid_end_sample - grid_beat0_sample
        preroll_samples = min(first_beat_sample_offset, grid_beat0_sample)

        src_start = max(0, first_beat_sample_offset - preroll_samples)
        src_end = min(len(audio), first_beat_sample_offset + span_samples)
        segment = audio[src_start:src_end].copy()

        dst_start = grid_beat0_sample - (first_beat_sample_offset - src_start)
        dst_end = min(total_samples, dst_start + len(segment))
        seg_len = dst_end - dst_start
        segment = segment[:seg_len]

        # Apply equal-power fade-in over [grid_beat0_sample, grid_beat0_sample + xfade_in_samples)
        if cue.crossfade_in_beats > 0:
            xfade_in_end_sample = beat_to_sample_index(
                cue.start_metronome_beat + cue.crossfade_in_beats, target_bpm, sr=SAMPLE_RATE
            )
            fade_start_rel = max(0, grid_beat0_sample - dst_start)
            fade_end_rel = min(seg_len, xfade_in_end_sample - dst_start)
            if fade_start_rel > 0:
                segment[:fade_start_rel] = 0.0
            fade_len = fade_end_rel - fade_start_rel
            if fade_len > 0:
                theta = np.linspace(0.0, 0.5 * np.pi, fade_len, endpoint=False, dtype=np.float32)
                gain_in = np.sin(theta)[:, np.newaxis]
                segment[fade_start_rel:fade_end_rel] *= gain_in

        # Apply equal-power fade-out over [grid_end_sample - xfade_out_samples, dst_end)
        if cue.crossfade_out_beats > 0:
            xfade_out_start_sample = beat_to_sample_index(
                cue.end_metronome_beat - cue.crossfade_out_beats, target_bpm, sr=SAMPLE_RATE
            )
            fade_start_rel = max(0, xfade_out_start_sample - dst_start)
            fade_end_rel = min(seg_len, grid_end_sample - dst_start)
            fade_len = fade_end_rel - fade_start_rel
            if fade_len > 0:
                theta = np.linspace(0.0, 0.5 * np.pi, fade_len, endpoint=False, dtype=np.float32)
                gain_out = np.cos(theta)[:, np.newaxis]
                segment[fade_start_rel:fade_end_rel] *= gain_out
            if fade_end_rel < seg_len:
                segment[fade_end_rel:] = 0.0
        else:
            # Short 10ms anti-click release on final track beat boundary
            rel_samples = min(int(0.010 * SAMPLE_RATE), seg_len)
            if rel_samples > 0:
                segment[-rel_samples:] *= np.linspace(1.0, 0.0, rel_samples, dtype=np.float32)[
                    :, np.newaxis
                ]

        master_buf[dst_start:dst_end] += segment

    # 4. Transparent soft peak limiting above -1.0 dBFS (0.891) so sum of click + kick never clips
    threshold = 0.89125
    headroom = 0.99 - threshold
    abs_buf = np.abs(master_buf)
    over = abs_buf > threshold
    if np.any(over):
        master_buf[over] = np.sign(master_buf[over]) * (
            threshold + headroom * np.tanh((abs_buf[over] - threshold) / headroom)
        )

    # 5. Stream master_buf to FFmpeg libmp3lame
    output_mp3_path.parent.mkdir(parents=True, exist_ok=True)
    encode_to_mp3(master_buf, output_mp3_path, sr=SAMPLE_RATE)

    manifest = MashupManifest(
        created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        output_mp3_path=str(output_mp3_path),
        sample_rate=SAMPLE_RATE,
        target_bpm=round(target_bpm, 4),
        metronome_beat_period_sec=round(60.0 / target_bpm, 6),
        crossfade_beats=crossfade_beats,
        lead_in_beats=lead_in_beats,
        lead_out_beats=lead_out_beats,
        metronome_gain_db=metronome_gain_db,
        total_metronome_beats=total_metro_beats,
        total_duration_sec=round(total_samples / SAMPLE_RATE, 3),
        tracks=cue_entries,
    )
    manifest_path = output_mp3_path.with_name(f"{output_mp3_path.stem}_manifest.json")
    manifest.save(manifest_path)
    return manifest


def encode_to_mp3(stereo_buf: np.ndarray, output_mp3_path: Path, sr: int = SAMPLE_RATE) -> None:
    """Pipe stereo float32 array directly into FFmpeg libmp3lame encoder."""
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "2",
        "-i",
        "-",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(output_mp3_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    # Stream in 1MB chunks to avoid large pipe buffer copy
    raw_bytes = np.ascontiguousarray(stereo_buf, dtype=np.float32).tobytes()
    _, stderr = proc.communicate(input=raw_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg MP3 encoding failed: {stderr.decode('utf-8', errors='replace')}")
