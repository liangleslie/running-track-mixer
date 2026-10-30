"""Assert-based self-check for rtmix DSP math, active windowing, and mashup output."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from rtmix.analyzer import (
    SAMPLE_RATE,
    analyze_track,
    compute_regression_bpm,
    select_harmonic_target,
)
from rtmix.metronome import beat_to_sample_index, synthesize_woodblock_click
from rtmix.mixer import assemble_mashup
from rtmix.pipeline import run_pipeline
from rtmix.processor import process_and_verify_track


def _make_synthetic_song(path: Path, bpm: float = 90.0, n_beats: int = 40, intro_sec: float = 3.0, outro_sec: float = 3.0) -> None:
    """Create a stereo WAV with quiet room floor intro/outro and musical body + drum beats at `bpm`."""
    period = 60.0 / bpm
    total_sec = intro_sec + n_beats * period + outro_sec
    n_samples = int(round(total_sec * SAMPLE_RATE))
    buf = np.zeros((n_samples, 2), dtype=np.float32)

    # Musical chord active during the song body [intro_sec, intro_sec + n_beats * period]
    s_body0 = int(round(intro_sec * SAMPLE_RATE))
    s_body1 = int(round((intro_sec + n_beats * period) * SAMPLE_RATE))
    t_body = np.arange(s_body1 - s_body0, dtype=np.float32) / SAMPLE_RATE
    chord = 0.05 * np.sin(2.0 * np.pi * 220.0 * t_body)
    buf[s_body0:s_body1, 0] = chord
    buf[s_body0:s_body1, 1] = chord

    click = synthesize_woodblock_click(sr=SAMPLE_RATE, gain_db=-3.0, f_start=180.0, f_end=60.0)
    for b in range(n_beats):
        s0 = int(round((intro_sec + b * period) * SAMPLE_RATE))
        s1 = min(n_samples, s0 + len(click))
        buf[s0:s1] += click[: s1 - s0]

    sf.write(str(path), buf, SAMPLE_RATE, subtype="FLOAT")


def main() -> None:
    # 1. Harmonic factor selection check:
    # - 85 BPM song vs 180 BPM target -> 0.5x = 90 BPM (2 steps per 1 beat)
    cfg_half = select_harmonic_target(85.0, 180.0)
    assert cfg_half.harmonic_factor == 0.5, f"Expected 0.5x factor for 85 BPM, got {cfg_half.harmonic_factor}"
    assert cfg_half.track_target_bpm == 90.0
    assert cfg_half.metronome_clicks_per_song_beat == 2.0

    # - 150 BPM song vs 180 BPM target -> 1.0x = 180 BPM
    cfg_full = select_harmonic_target(150.0, 180.0)
    assert cfg_full.harmonic_factor == 1.0
    assert cfg_full.track_target_bpm == 180.0

    # 2. Zero cumulative drift check over 10,000 beats at 170 BPM
    expected_sample_10k = round(10000 * (60.0 / 170.0) * SAMPLE_RATE)
    assert beat_to_sample_index(10000, 170.0) == expected_sample_10k

    # 3. Regression BPM accuracy check
    true_period = 60.0 / 123.456
    beats = 2.0 + np.arange(100) * true_period
    est_bpm, _ = compute_regression_bpm(beats)
    assert abs(est_bpm - 123.456) < 1e-4, f"Regression BPM error: {est_bpm}"

    # 4. End-to-end synthetic pipeline check (intro/outro strip + stretch + verify + mix)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        s1 = tmp_dir / "song1_80bpm.wav"
        s2 = tmp_dir / "song2_165bpm.wav"
        _make_synthetic_song(s1, bpm=80.0, n_beats=40, intro_sec=3.0, outro_sec=3.0)
        _make_synthetic_song(s2, bpm=165.0, n_beats=64, intro_sec=2.5, outro_sec=2.5)

        work_dir = tmp_dir / "work"
        tm1 = process_and_verify_track(analyze_track(s1, 170.0), work_dir=work_dir)
        tm2 = process_and_verify_track(analyze_track(s2, 170.0), work_dir=work_dir)

        assert tm1.pre_stretch.active_window.stripped_intro_sec >= 2.5
        assert tm1.pre_stretch.active_window.stripped_outro_sec >= 2.5
        assert tm1.post_stretch is not None
        assert abs(tm1.post_stretch.verified_regression_bpm - 85.0) < 0.25

        assert tm2.post_stretch is not None
        assert abs(tm2.post_stretch.verified_regression_bpm - 170.0) < 0.25

        out_mp3 = tmp_dir / "out" / "mashup.mp3"
        manifest = assemble_mashup([tm1, tm2], out_mp3, target_bpm=170.0, crossfade_beats=8)
        assert out_mp3.is_file() and out_mp3.stat().st_size > 10000
        manifest_json = json.loads((tmp_dir / "out" / "mashup_manifest.json").read_text())
        assert len(manifest_json["tracks"]) == 2
        assert manifest.tracks[1].start_metronome_beat == manifest.tracks[0].end_metronome_beat - 8

        # 5. Parallel worker pool + dead-zone drop vs -force check
        s3_deadzone = tmp_dir / "song3_120bpm_deadzone.wav"
        _make_synthetic_song(s3_deadzone, bpm=120.0, n_beats=48, intro_sec=2.0, outro_sec=2.0)

        # Default (force=False) drops the 120 BPM dead-zone track at 180 BPM target
        par_mp3 = tmp_dir / "out_par" / "mashup.mp3"
        par_manifest = run_pipeline(
            input_dir=tmp_dir,
            output_mp3_path=par_mp3,
            target_bpm=180.0,
            crossfade_beats=8,
            work_dir=tmp_dir / "work_par",
            workers=2,
            force=False,
        )
        assert [t.file_name for t in par_manifest.tracks] == ["song1_80bpm.wav", "song2_165bpm.wav"]

        # With force=True (-force / --force), dead-zone track is included
        force_mp3 = tmp_dir / "out_force" / "mashup.mp3"
        force_manifest = run_pipeline(
            input_dir=tmp_dir,
            output_mp3_path=force_mp3,
            target_bpm=180.0,
            crossfade_beats=8,
            work_dir=tmp_dir / "work_par",
            workers=2,
            force=True,
        )
        assert [t.file_name for t in force_manifest.tracks] == [
            "song1_80bpm.wav",
            "song2_165bpm.wav",
            "song3_120bpm_deadzone.wav",
        ]

        # 6. Random N selection check (automatically skips dead-zone track and fills N=2 valid tracks)
        rand_mp3 = tmp_dir / "out_rand" / "mashup.mp3"
        rand_manifest = run_pipeline(
            input_dir=tmp_dir,
            output_mp3_path=rand_mp3,
            target_bpm=180.0,
            crossfade_beats=8,
            work_dir=tmp_dir / "work_par",
            random_sample=2,
            seed=42,
            workers=2,
            force=False,
        )
        assert len(rand_manifest.tracks) == 2
        assert "song3_120bpm_deadzone.wav" not in [t.file_name for t in rand_manifest.tracks]

    print("ALL SELF-CHECKS PASSED.")


if __name__ == "__main__":
    main()
