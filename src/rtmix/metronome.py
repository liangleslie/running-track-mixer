"""Sample-accurate woodblock metronome synthesis and zero-drift grid stamping."""

from __future__ import annotations

import numpy as np

from rtmix.analyzer import SAMPLE_RATE


def synthesize_woodblock_click(
    sr: int = SAMPLE_RATE,
    duration_sec: float = 0.025,
    gain_db: float = -6.0,
    f_start: float = 1400.0,
    f_end: float = 850.0,
) -> np.ndarray:
    """Synthesize a crisp stereo woodblock click kernel (N, 2) float32."""
    n_samples = int(round(duration_sec * sr))
    t = np.arange(n_samples, dtype=np.float64) / sr

    tau_f = 0.004
    phase = 2.0 * np.pi * (f_end * t + (f_start - f_end) * tau_f * (1.0 - np.exp(-t / tau_f)))

    attack = np.minimum(1.0, t / 0.001)
    decay = np.exp(-t / 0.0045)
    amplitude = 10.0 ** (gain_db / 20.0)

    mono_click = (amplitude * attack * decay * np.sin(phase)).astype(np.float32)
    return np.column_stack((mono_click, mono_click))


def beat_to_sample_index(beat_idx: int | float, target_bpm: float, sr: int = SAMPLE_RATE) -> int:
    """Convert global metronome beat index to exact sample coordinate (zero cumulative drift)."""
    beat_period_sec = 60.0 / target_bpm
    return int(round(beat_idx * beat_period_sec * sr))


def render_metronome_track(
    total_samples: int,
    total_metronome_beats: int,
    target_bpm: float,
    gain_db: float = -6.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Create stereo float32 metronome buffer of shape (total_samples, 2) with clicks at exact beats."""
    buf = np.zeros((total_samples, 2), dtype=np.float32)
    click = synthesize_woodblock_click(sr=sr, gain_db=gain_db)
    click_len = len(click)

    for m in range(total_metronome_beats):
        s0 = beat_to_sample_index(m, target_bpm, sr=sr)
        if s0 >= total_samples:
            break
        s1 = min(total_samples, s0 + click_len)
        buf[s0:s1] += click[: s1 - s0]

    return buf
