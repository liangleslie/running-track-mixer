"""Audio decoding, HPSS active-window trimming, and Theil-Sen BPM analysis."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy.stats import theilslopes
import soundfile as sf

from rtmix.models import (
    ActiveWindow,
    PostStretchVerification,
    PreStretchAnalysis,
    SourceInfo,
    SyncDiagnostics,
    TargetConfig,
    TrackMetadata,
)

SAMPLE_RATE = 44100
HOP_COARSE = 512
HOP_FINE = 128
PREROLL_SEC = 0.015
DEFAULT_HARMONIC_FACTORS = [0.5, 1.0, 2.0]


def compute_sha256_prefix(path: Path, length: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def make_track_id(path: Path, sha_prefix: str) -> str:
    stem = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", path.stem).strip("_")
    if len(stem) > 40:
        stem = stem[:40].rstrip("_")
    return f"{stem}_{sha_prefix}"


def probe_source_info(path: Path) -> SourceInfo:
    sha_prefix = compute_sha256_prefix(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    sr = int(stream.get("sample_rate") or SAMPLE_RATE)
    channels = int(stream.get("channels") or 2)
    return SourceInfo(
        file_path=str(path.resolve()),
        file_name=path.name,
        sha256_prefix=sha_prefix,
        original_duration_sec=round(duration, 4),
        original_sample_rate=sr,
        original_channels=channels,
    )


def decode_audio_mono(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio container (.m4a, .mp3, .wav, .flac) to mono float32 at `sr`."""
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y_mono = np.mean(y, axis=1)
        if file_sr != sr:
            y_mono = librosa.resample(y_mono, orig_sr=file_sr, target_sr=sr)
        return y_mono

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def _detect_beats_and_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    start_bpm: float = 120.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (coarse_bpm, refined_beat_times, beat_perc_rms, beat_onset_peaks).

    Uses spectrogram-domain HPSS so STFT is computed only once.
    """
    stft_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP_COARSE))
    _, stft_perc = librosa.decompose.hpss(stft_mag, margin=2.0)

    mel_perc = librosa.feature.melspectrogram(S=stft_perc**2, sr=sr)
    onset_env = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel_perc, ref=np.max),
        sr=sr,
        hop_length=HOP_COARSE,
    )
    tempo_arr, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_COARSE,
        start_bpm=start_bpm,
        tightness=100,
    )
    coarse_bpm = float(np.ndim(tempo_arr) and tempo_arr[0] or tempo_arr)

    if len(beat_frames) == 0:
        return coarse_bpm, np.array([], dtype=np.float64), np.array([]), np.array([])

    # Fine onset envelope (hop=128 -> 2.9ms resolution) to de-quantize 11.6ms coarse frame hop
    onset_fine = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_FINE)
    ratio = HOP_COARSE // HOP_FINE
    search_radius = 3  # +/- 3 fine frames (~8.7ms, half of coarse hop)
    refined_times = np.empty(len(beat_frames), dtype=np.float64)

    for idx, bf in enumerate(beat_frames):
        center = int(bf * ratio)
        lo = max(0, center - search_radius)
        hi = min(len(onset_fine), center + search_radius + 1)
        if hi > lo:
            best_fine = lo + int(np.argmax(onset_fine[lo:hi]))
            refined_times[idx] = best_fine * HOP_FINE / sr
        else:
            refined_times[idx] = bf * HOP_COARSE / sr

    # Compute per-beat percussive RMS and frame-level total RMS for musical windowing
    perc_rms_frames = librosa.feature.rms(S=stft_perc, frame_length=2048, hop_length=HOP_COARSE)[0]
    total_rms_frames = librosa.feature.rms(S=stft_mag, frame_length=2048, hop_length=HOP_COARSE)[0]
    n_beats = len(beat_frames)
    beat_perc_rms = np.zeros(n_beats, dtype=np.float64)

    for i in range(n_beats):
        f0 = int(beat_frames[i])
        f1 = int(beat_frames[i + 1]) if i + 1 < n_beats else min(len(perc_rms_frames), f0 + 20)
        f0 = max(0, min(f0, len(perc_rms_frames) - 1))
        f1 = max(f0 + 1, min(f1, len(perc_rms_frames)))
        beat_perc_rms[i] = float(np.sqrt(np.mean(perc_rms_frames[f0:f1] ** 2)))

    return coarse_bpm, refined_times, beat_perc_rms, total_rms_frames


def compute_regression_bpm(active_beat_times: np.ndarray) -> tuple[float, float]:
    """Compute (bpm, beat_period_sec) using windowed Theil-Sen regression immune to bridge slips."""
    n = len(active_beat_times)
    win_len = 48
    if n >= win_len:
        slopes: list[float] = []
        for i in range(0, n - win_len + 1, 16):
            seg = active_beat_times[i : i + win_len]
            ibis = np.diff(seg)
            mean_ibi = float(np.mean(ibis))
            if mean_ibi > 0 and (float(np.std(ibis)) / mean_ibi) < 0.10:
                s, _, _, _ = theilslopes(seg, np.arange(win_len, dtype=np.float64))
                slopes.append(float(s))
        if len(slopes) >= 3:
            beat_period = float(np.median(slopes))
            return 60.0 / beat_period, beat_period

    indices = np.arange(n, dtype=np.float64)
    slope, _, _, _ = theilslopes(active_beat_times, indices)
    beat_period = float(slope)
    bpm = 60.0 / beat_period
    return bpm, beat_period


def _find_continuous_musical_bounds(
    total_rms_frames: np.ndarray,
    total_duration_sec: float,
    sr: int = SAMPLE_RATE,
) -> tuple[float, float]:
    """Find [t_music_start, t_music_end] keeping acoustic/vocal intros while stripping silence & MV skits."""
    n_frames = len(total_rms_frames)
    if n_frames < 20:
        return 0.0, total_duration_sec

    mid_lo = max(0, int(0.20 * n_frames))
    mid_hi = min(n_frames, max(mid_lo + 10, int(0.80 * n_frames)))
    med_total_rms = float(np.median(total_rms_frames[mid_lo:mid_hi]))

    win = max(1, int(0.8 * sr / HOP_COARSE))
    rms_smooth = np.convolve(total_rms_frames, np.ones(win) / win, mode="same")
    active = rms_smooth >= 0.05 * med_total_rms

    # Bridge brief musical pauses up to 2.2s
    gap_frames = int(2.2 * sr / HOP_COARSE)
    closed = active.copy()
    idx_true = np.where(active)[0]
    if len(idx_true) == 0:
        return 0.0, total_duration_sec

    for a, b in zip(idx_true[:-1], idx_true[1:]):
        if 1 < (b - a) <= gap_frames:
            closed[a:b] = True

    padded = np.concatenate(([False], closed, [False]))
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]
    best = int(np.argmax(ends - starts))
    t_start = max(0.0, float(starts[best] * HOP_COARSE / sr))
    t_end = min(total_duration_sec, float(ends[best] * HOP_COARSE / sr))
    return t_start, t_end


def _select_active_beat_window(
    beat_times: np.ndarray,
    beat_perc_rms: np.ndarray,
    total_rms_frames: np.ndarray,
    total_duration_sec: float,
) -> tuple[ActiveWindow, float, float]:
    """Identify [i_start, i_end] preserving musical intros/outros while locking to core drum grid."""
    n = len(beat_times)
    if n < 16:
        raise ValueError(f"Insufficient beats detected ({n} < 16) for rhythmic windowing.")

    # 1. Find continuous musical bounds (including piano/guitar/vocal intros)
    t_music_start, t_music_end = _find_continuous_musical_bounds(
        total_rms_frames, total_duration_sec, sr=SAMPLE_RATE
    )

    # 2. Identify core percussive section for high-precision BPM & phase anchoring
    ibis = np.diff(beat_times)
    mid_lo = max(0, int(0.25 * len(ibis)))
    mid_hi = min(len(ibis), max(mid_lo + 4, int(0.75 * len(ibis))))
    median_ibi = float(np.median(ibis[mid_lo:mid_hi]))
    med_perc_rms = float(np.median(beat_perc_rms[mid_lo:mid_hi]))

    ibi_ok = np.abs(ibis - median_ibi) / max(median_ibi, 1e-6) <= 0.10
    ibi_ok = np.append(ibi_ok, True)
    perc_mask = ibi_ok & (beat_perc_rms >= 0.28 * med_perc_rms)

    window_size = 8
    min_active_in_window = 6
    p_start = 0
    for i in range(0, n - window_size):
        if int(np.sum(perc_mask[i : i + window_size])) >= min_active_in_window:
            p_start = i + int(np.argmax(perc_mask[i : i + window_size]))
            break

    p_end = n - 1
    for i in range(n - 1, p_start + window_size - 1, -1):
        win = perc_mask[i - window_size + 1 : i + 1]
        if int(np.sum(win)) >= min_active_in_window:
            p_end = i - int(np.argmax(win[::-1]))
            break

    if (p_end - p_start) < 12:
        p_start, p_end = 0, n - 1

    perc_slice = beat_times[p_start : p_end + 1]
    perc_rms_slice = beat_perc_rms[p_start : p_end + 1]
    native_bpm, native_period = compute_regression_bpm(perc_slice)

    # 3. Compute studio drum grid phase anchor from strong percussive beats
    raw_drum_anchor = float(beat_times[p_start])
    strong_beats = perc_slice[perc_rms_slice >= 0.40 * med_perc_rms]
    if len(strong_beats) < 8:
        strong_beats = perc_slice
    rel_phases = (
        (strong_beats - raw_drum_anchor + 0.5 * native_period) % native_period
    ) - 0.5 * native_period
    drum_anchor_sec = raw_drum_anchor + float(np.median(rel_phases))

    # 4. Project drum_anchor_sec backwards to t_music_start by exact integer beats
    k_back = int(np.floor((drum_anchor_sec - t_music_start) / native_period))
    first_beat_sec = drum_anchor_sec - k_back * native_period
    while first_beat_sec < max(PREROLL_SEC, t_music_start - 0.25 * native_period):
        first_beat_sec += native_period

    raw_span_sec = max(4.0 * native_period, t_music_end - first_beat_sec)
    true_beat_count = int(round(raw_span_sec / native_period))
    active_count = max(4, (true_beat_count // 4) * 4)
    while first_beat_sec + active_count * native_period > total_duration_sec and active_count > 4:
        active_count -= 4

    last_beat_sec = min(total_duration_sec, first_beat_sec + active_count * native_period)
    trim_start_sec = max(0.0, first_beat_sec - PREROLL_SEC)
    trim_end_sec = last_beat_sec

    i_start = int(np.searchsorted(beat_times, first_beat_sec - 0.25 * native_period))
    i_end = int(min(n - 1, np.searchsorted(beat_times, last_beat_sec + 0.25 * native_period)))

    win = ActiveWindow(
        start_beat_index=i_start,
        end_beat_index=i_end,
        active_beat_count=int(active_count),
        first_active_beat_sec=round(first_beat_sec, 4),
        last_active_beat_sec=round(last_beat_sec, 4),
        trim_start_sec=round(trim_start_sec, 4),
        trim_end_sec=round(trim_end_sec, 4),
        stripped_intro_sec=round(trim_start_sec, 4),
        stripped_outro_sec=round(max(0.0, total_duration_sec - trim_end_sec), 4),
    )
    return win, native_bpm, native_period


def select_harmonic_target(
    native_bpm: float,
    global_target_bpm: float = 180.0,
    harmonic_factors: list[float] | None = None,
    slowdown_penalty: float = 2.0,
) -> TargetConfig:
    """Pick binary harmonic factor k in harmonic_factors with upward-bias (penalizing slowdown vs speedup)."""
    if harmonic_factors is None:
        harmonic_factors = DEFAULT_HARMONIC_FACTORS

    def _stretch_cost(k: float) -> float:
        r = (k * global_target_bpm) / native_bpm
        log_dist = float(np.log2(r))
        return log_dist if log_dist >= 0.0 else slowdown_penalty * abs(log_dist)

    best_k = min(harmonic_factors, key=_stretch_cost)
    track_target_bpm = best_k * global_target_bpm
    return TargetConfig(
        global_target_bpm=round(global_target_bpm, 4),
        harmonic_factor=round(float(best_k), 6),
        track_target_bpm=round(track_target_bpm, 4),
        metronome_clicks_per_song_beat=round(1.0 / best_k, 4),
    )


def analyze_track(
    path: Path,
    global_target_bpm: float,
    harmonic_factors: list[float] | None = None,
) -> TrackMetadata:
    """Run Steps 1-3 on a source audio track and return TrackMetadata."""
    if harmonic_factors is None:
        harmonic_factors = DEFAULT_HARMONIC_FACTORS

    source = probe_source_info(path)
    y = decode_audio_mono(path, sr=SAMPLE_RATE)
    actual_duration = len(y) / SAMPLE_RATE
    if source.original_duration_sec <= 0.0:
        source.original_duration_sec = round(actual_duration, 4)

    if actual_duration < 15.0:
        raise ValueError(f"Track '{path.name}' is too short ({actual_duration:.1f}s < 15.0s).")

    coarse_bpm, beat_times, beat_perc_rms, total_rms_frames = _detect_beats_and_features(
        y, sr=SAMPLE_RATE
    )
    active_win, native_bpm, native_period = _select_active_beat_window(
        beat_times, beat_perc_rms, total_rms_frames, actual_duration
    )

    active_beats = beat_times[active_win.start_beat_index : active_win.end_beat_index + 1]
    target_cfg = select_harmonic_target(native_bpm, global_target_bpm, harmonic_factors)
    stretch_ratio = target_cfg.track_target_bpm / native_bpm

    pre = PreStretchAnalysis(
        librosa_coarse_bpm=round(coarse_bpm, 4),
        regression_native_bpm=round(native_bpm, 4),
        native_beat_period_sec=round(native_period, 6),
        total_detected_beats_raw=len(beat_times),
        active_window=active_win,
        planned_stretch_ratio=round(stretch_ratio, 6),
        beat_timestamps_active_sec=[round(float(t), 4) for t in active_beats],
    )

    track_id = make_track_id(path, source.sha256_prefix)
    return TrackMetadata(
        track_id=track_id,
        source=source,
        target_config=target_cfg,
        pre_stretch=pre,
        status="analyzed",
    )


def verify_stretched_track(
    stretched_wav_path: Path,
    track_target_bpm: float,
    applied_stretch_ratio: float,
    correction_passes: int = 0,
    bpm_tolerance: float = 0.15,
) -> PostStretchVerification:
    """Run Step 5 verification on `stretched.wav` and return PostStretchVerification."""
    y_post = decode_audio_mono(stretched_wav_path, sr=SAMPLE_RATE)
    stretched_duration = len(y_post) / SAMPLE_RATE

    _, post_beats, post_rms, _ = _detect_beats_and_features(
        y_post, sr=SAMPLE_RATE, start_bpm=track_target_bpm
    )
    if len(post_beats) < 8:
        raise ValueError(
            f"Post-stretch beat verification found too few beats ({len(post_beats)}) in {stretched_wav_path}"
        )

    # Trim outer 2 beats when computing regression slope to avoid boundary filter edge effects
    core_beats = post_beats[2:-2] if len(post_beats) > 12 else post_beats
    verified_bpm, _ = compute_regression_bpm(core_beats)
    bpm_error = abs(verified_bpm - track_target_bpm)

    expected_period = 60.0 / track_target_bpm
    theo_first_beat = min(PREROLL_SEC / max(applied_stretch_ratio, 0.1), 0.5 * expected_period)

    # Filter to rhythmically active beats (excluding drumless vocal bridges) for phase anchoring
    med_post_rms = float(np.median(post_rms))
    perc_beats = post_beats[post_rms >= 0.35 * med_post_rms]
    if len(perc_beats) < 8:
        perc_beats = post_beats

    # Circular modular residual centered around theo_first_beat to find optimal Beat 0 phase anchor
    rel_phases = (
        (perc_beats - theo_first_beat + 0.5 * expected_period) % expected_period
    ) - 0.5 * expected_period
    first_beat_offset = float((theo_first_beat + float(np.median(rel_phases))) % expected_period)

    # Phase error of each percussive beat against the ideal grid (first_beat_offset + k * expected_period)
    phase_errors_sec = (
        (perc_beats - first_beat_offset + 0.5 * expected_period) % expected_period
    ) - 0.5 * expected_period
    abs_errors_ms = np.abs(phase_errors_sec) * 1000.0

    # Use IQR-filtered mean/max (middle 85% of percussive beats)
    p85_errors = np.sort(abs_errors_ms)[: max(1, int(0.85 * len(abs_errors_ms)))]
    mean_err_ms = float(np.mean(p85_errors))
    max_err_ms = float(np.max(p85_errors))

    return PostStretchVerification(
        stretched_wav_path=str(stretched_wav_path),
        stretched_duration_sec=round(stretched_duration, 4),
        applied_stretch_ratio=round(applied_stretch_ratio, 6),
        correction_passes=correction_passes,
        verified_regression_bpm=round(verified_bpm, 4),
        bpm_error=round(bpm_error, 4),
        first_beat_offset_sec=round(first_beat_offset, 5),
        sync_diagnostics=SyncDiagnostics(
            mean_abs_phase_error_ms=round(mean_err_ms, 2),
            max_abs_phase_error_ms=round(max_err_ms, 2),
            is_within_tolerance=bool(bpm_error <= bpm_tolerance),
        ),
        beat_timestamps_sec=[round(float(t), 4) for t in post_beats],
    )
