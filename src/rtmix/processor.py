"""FFmpeg trimming, rubberband time-stretching, loudness normalization, and verification."""

from __future__ import annotations

from pathlib import Path
import subprocess

from rtmix.analyzer import SAMPLE_RATE, verify_stretched_track
from rtmix.models import TrackMetadata

BPM_TOLERANCE = 0.80


def run_ffmpeg_stretch_and_normalize(
    input_path: Path,
    output_wav_path: Path,
    trim_start_sec: float,
    trim_end_sec: float,
    stretch_ratio: float,
    loudness_lufs: float = -14.0,
) -> None:
    """Trim to active window, pitch-preserving time-stretch via rubberband, and normalize loudness."""
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = (
        f"atrim=start={trim_start_sec:.6f}:end={trim_end_sec:.6f},"
        f"asetpts=PTS-STARTPTS,"
        f"rubberband=tempo={stretch_ratio:.6f}:transients=crisp:detector=compound:phase=laminar,"
        f"loudnorm=I={loudness_lufs:.1f}:TP=-1.5:LRA=11.0"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(input_path),
        "-af",
        filter_chain,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_f32le",
        str(output_wav_path),
    ]
    subprocess.run(cmd, check=True)


def process_and_verify_track(
    metadata: TrackMetadata,
    work_dir: Path,
    bpm_tolerance: float = BPM_TOLERANCE,
) -> TrackMetadata:
    """Run Steps 4 and 5: FFmpeg rubberband stretch + post-stretch verification (with 1 micro-correction retry)."""
    track_dir = work_dir / metadata.track_id
    track_dir.mkdir(parents=True, exist_ok=True)
    meta_path = track_dir / "metadata.json"
    stretched_wav = track_dir / "stretched.wav"

    # Persist Step 3 pre-stretch metadata first
    metadata.save(meta_path)

    input_path = Path(metadata.source.file_path)
    win = metadata.pre_stretch.active_window
    applied_ratio = metadata.pre_stretch.planned_stretch_ratio
    target_bpm = metadata.target_config.track_target_bpm

    run_ffmpeg_stretch_and_normalize(
        input_path=input_path,
        output_wav_path=stretched_wav,
        trim_start_sec=win.trim_start_sec,
        trim_end_sec=win.trim_end_sec,
        stretch_ratio=applied_ratio,
    )

    post = verify_stretched_track(
        stretched_wav_path=stretched_wav,
        track_target_bpm=target_bpm,
        applied_stretch_ratio=applied_ratio,
        correction_passes=0,
        bpm_tolerance=bpm_tolerance,
    )

    # Step 5 micro-correction retry if residual BPM drift exceeds tolerance (and isn't an octave jump)
    if (
        not post.sync_diagnostics.is_within_tolerance
        and 0.85 <= (target_bpm / max(post.verified_regression_bpm, 1.0)) <= 1.15
    ):
        refined_ratio = applied_ratio * (target_bpm / post.verified_regression_bpm)
        run_ffmpeg_stretch_and_normalize(
            input_path=input_path,
            output_wav_path=stretched_wav,
            trim_start_sec=win.trim_start_sec,
            trim_end_sec=win.trim_end_sec,
            stretch_ratio=refined_ratio,
        )
        post = verify_stretched_track(
            stretched_wav_path=stretched_wav,
            track_target_bpm=target_bpm,
            applied_stretch_ratio=refined_ratio,
            correction_passes=1,
            bpm_tolerance=bpm_tolerance,
        )

    metadata.post_stretch = post
    metadata.status = "verified"
    metadata.save(meta_path)
    return metadata
