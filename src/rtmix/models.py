"""Data models and JSON schemas for per-track metadata and global mashup manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class SourceInfo:
    file_path: str
    file_name: str
    sha256_prefix: str
    original_duration_sec: float
    original_sample_rate: int
    original_channels: int


@dataclass
class TargetConfig:
    global_target_bpm: float
    harmonic_factor: float
    track_target_bpm: float
    metronome_clicks_per_song_beat: float


@dataclass
class ActiveWindow:
    start_beat_index: int
    end_beat_index: int
    active_beat_count: int
    first_active_beat_sec: float
    last_active_beat_sec: float
    trim_start_sec: float
    trim_end_sec: float
    stripped_intro_sec: float
    stripped_outro_sec: float


@dataclass
class PreStretchAnalysis:
    librosa_coarse_bpm: float
    regression_native_bpm: float
    native_beat_period_sec: float
    total_detected_beats_raw: int
    active_window: ActiveWindow
    planned_stretch_ratio: float
    beat_timestamps_active_sec: list[float] = field(default_factory=list)


@dataclass
class SyncDiagnostics:
    mean_abs_phase_error_ms: float
    max_abs_phase_error_ms: float
    is_within_tolerance: bool


@dataclass
class PostStretchVerification:
    stretched_wav_path: str
    stretched_duration_sec: float
    applied_stretch_ratio: float
    correction_passes: int
    verified_regression_bpm: float
    bpm_error: float
    first_beat_offset_sec: float
    sync_diagnostics: SyncDiagnostics
    beat_timestamps_sec: list[float] = field(default_factory=list)


@dataclass
class TrackMetadata:
    track_id: str
    source: SourceInfo
    target_config: TargetConfig
    pre_stretch: PreStretchAnalysis
    post_stretch: PostStretchVerification | None = None
    status: str = "analyzed"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrackMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        post = None
        if data.get("post_stretch") is not None:
            ps = data["post_stretch"]
            post = PostStretchVerification(
                stretched_wav_path=ps["stretched_wav_path"],
                stretched_duration_sec=ps["stretched_duration_sec"],
                applied_stretch_ratio=ps["applied_stretch_ratio"],
                correction_passes=ps["correction_passes"],
                verified_regression_bpm=ps["verified_regression_bpm"],
                bpm_error=ps["bpm_error"],
                first_beat_offset_sec=ps["first_beat_offset_sec"],
                sync_diagnostics=SyncDiagnostics(**ps["sync_diagnostics"]),
                beat_timestamps_sec=ps["beat_timestamps_sec"],
            )
        pre = data["pre_stretch"]
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            track_id=data["track_id"],
            source=SourceInfo(**data["source"]),
            target_config=TargetConfig(**data["target_config"]),
            pre_stretch=PreStretchAnalysis(
                librosa_coarse_bpm=pre["librosa_coarse_bpm"],
                regression_native_bpm=pre["regression_native_bpm"],
                native_beat_period_sec=pre["native_beat_period_sec"],
                total_detected_beats_raw=pre["total_detected_beats_raw"],
                active_window=ActiveWindow(**pre["active_window"]),
                planned_stretch_ratio=pre["planned_stretch_ratio"],
                beat_timestamps_active_sec=pre["beat_timestamps_active_sec"],
            ),
            post_stretch=post,
            status=data.get("status", "analyzed"),
        )


@dataclass
class TrackCueEntry:
    order: int
    track_id: str
    file_name: str
    native_bpm: float
    harmonic_factor: float
    target_track_bpm: float
    verified_bpm: float
    stretch_ratio: float
    song_beats: int
    metronome_beats_spanned: int
    start_metronome_beat: int
    end_metronome_beat: int
    start_time_sec: float
    end_time_sec: float
    crossfade_in_beats: int
    crossfade_out_beats: int
    mean_abs_phase_error_ms: float


@dataclass
class MashupManifest:
    created_at: str
    output_mp3_path: str
    sample_rate: int
    target_bpm: float
    metronome_beat_period_sec: float
    crossfade_beats: int
    lead_in_beats: int
    lead_out_beats: int
    metronome_gain_db: float
    total_metronome_beats: int
    total_duration_sec: float
    tracks: list[TrackCueEntry]
    schema_version: str = "1.0"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
