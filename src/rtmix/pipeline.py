"""Pipeline orchestration, parallel worker pool, and caching for Running Tracks Mixer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import random
import threading

from rtmix.analyzer import (
    DEFAULT_HARMONIC_FACTORS,
    analyze_track,
    compute_sha256_prefix,
    make_track_id,
    select_harmonic_target,
)
from rtmix.mixer import assemble_mashup
from rtmix.models import MashupManifest, TrackMetadata
from rtmix.processor import process_and_verify_track

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


def default_workers() -> int:
    """Return sensible default worker process count (half of logical CPUs, at least 1)."""
    return max(1, (os.cpu_count() or 4) // 2)


def _init_worker() -> None:
    """Cap internal BLAS/OpenMP/Numba thread pools per worker to prevent CPU thrashing."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "2")


def _emit_log(msg: str, log_fn: LogCallback | None = None) -> None:
    print(msg, flush=True)
    if log_fn is not None:
        log_fn(msg)


def discover_audio_files(
    input_path: Path | Sequence[Path],
    limit: int | None = None,
    random_sample: int | None = None,
    seed: int | None = None,
) -> list[Path]:
    """Return list of supported audio files from a file, directory, or sequence of paths."""
    paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    files: list[Path] = []
    seen: set[Path] = set()

    for p in paths:
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved = p.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(p)
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(child)
        else:
            raise FileNotFoundError(f"Input path does not exist: {p}")

    if random_sample is not None and random_sample > 0:
        rng = random.Random(seed)
        files = rng.sample(files, min(random_sample, len(files)))
    elif limit is not None and limit > 0:
        files = files[:limit]
    return files


def load_cached_track(
    path: Path,
    work_dir: Path,
    target_bpm: float,
    harmonic_factors: list[float],
    min_stretch: float = 0.85,
    max_stretch: float = 1.35,
    force: bool = False,
) -> tuple[str, TrackMetadata | None]:
    """Return ('verified', tm), ('skipped_deadzone', tm), or ('miss', None)."""
    sha_prefix = compute_sha256_prefix(path)
    track_id = make_track_id(path, sha_prefix)
    meta_path = work_dir / track_id / "metadata.json"
    wav_path = work_dir / track_id / "stretched.wav"
    if not meta_path.is_file():
        return "miss", None
    try:
        tm = TrackMetadata.load(meta_path)
    except Exception:
        return "miss", None

    expected_cfg = select_harmonic_target(
        tm.pre_stretch.regression_native_bpm, target_bpm, harmonic_factors
    )
    if (
        abs(tm.target_config.global_target_bpm - target_bpm) >= 1e-3
        or abs(tm.target_config.harmonic_factor - expected_cfg.harmonic_factor) >= 1e-3
    ):
        return "miss", None

    r = tm.pre_stretch.planned_stretch_ratio
    in_bounds = min_stretch <= r <= max_stretch

    if not in_bounds and not force:
        return "skipped_deadzone", tm

    if tm.status == "verified" and tm.post_stretch is not None and wav_path.is_file():
        return "verified", tm

    return "miss", None


def _process_track_job(
    idx: int,
    total: int,
    file_path: Path,
    target_bpm: float,
    harmonic_factors: list[float],
    work_dir: Path,
    min_stretch: float = 0.85,
    max_stretch: float = 1.35,
    force: bool = False,
) -> tuple[int, TrackMetadata | None, str]:
    """Worker entrypoint for Steps 1-5 on a single track."""
    tm = analyze_track(
        path=file_path,
        global_target_bpm=target_bpm,
        harmonic_factors=harmonic_factors,
    )
    win = tm.pre_stretch.active_window
    r = tm.pre_stretch.planned_stretch_ratio
    in_bounds = min_stretch <= r <= max_stretch

    if not in_bounds and not force:
        meta_path = work_dir / tm.track_id / "metadata.json"
        tm.status = "skipped_deadzone"
        tm.save(meta_path)
        msg = (
            f"[{idx}/{total}] [WARN] Dropping dead-zone track '{file_path.name}': "
            f"native={tm.pre_stretch.regression_native_bpm:.2f} BPM -> "
            f"target={tm.target_config.track_target_bpm:.1f} BPM ({tm.target_config.harmonic_factor}x) "
            f"requires {r:.3f}x stretch (outside [{min_stretch:.2f}x, {max_stretch:.2f}x]; "
            f"pass -force to include)"
        )
        return idx, None, msg

    force_note = ""
    if not in_bounds and force:
        force_note = (
            f"    [WARN] Forcing dead-zone stretch {r:.3f}x "
            f"(outside [{min_stretch:.2f}x, {max_stretch:.2f}x])\n"
        )

    tm = process_and_verify_track(tm, work_dir=work_dir)
    assert tm.post_stretch is not None
    ps = tm.post_stretch
    msg = (
        f"[{idx}/{total}] DONE {file_path.name}\n"
        f"{force_note}"
        f"    -> Native BPM: {tm.pre_stretch.regression_native_bpm:.2f} "
        f"(coarse: {tm.pre_stretch.librosa_coarse_bpm:.2f}) | "
        f"Target: {tm.target_config.track_target_bpm:.1f} BPM ({tm.target_config.harmonic_factor}x) | "
        f"Stretch: {tm.pre_stretch.planned_stretch_ratio:.4f}x | "
        f"Trim: -{win.stripped_intro_sec:.1f}s head, -{win.stripped_outro_sec:.1f}s tail "
        f"({win.active_beat_count} beats)\n"
        f"    -> Verified BPM: {ps.verified_regression_bpm:.2f} "
        f"(retries={ps.correction_passes}) | "
        f"Beat0 offset: {ps.first_beat_offset_sec * 1000:.1f}ms | "
        f"Phase error: mean={ps.sync_diagnostics.mean_abs_phase_error_ms:.2f}ms, "
        f"max={ps.sync_diagnostics.max_abs_phase_error_ms:.2f}ms"
    )
    return idx, tm, msg


def _analyze_track_job(
    idx: int,
    total: int,
    file_path: Path,
    target_bpm: float,
    harmonic_factors: list[float],
    work_dir: Path,
) -> tuple[int, TrackMetadata, str]:
    """Worker entrypoint for Steps 1-3 pre-stretch analysis only."""
    tm = analyze_track(file_path, global_target_bpm=target_bpm, harmonic_factors=harmonic_factors)
    meta_path = work_dir / tm.track_id / "metadata.json"
    tm.save(meta_path)
    win = tm.pre_stretch.active_window
    msg = (
        f"[{idx}/{total}] {file_path.name}\n"
        f"  native_bpm={tm.pre_stretch.regression_native_bpm:.2f} "
        f"(coarse={tm.pre_stretch.librosa_coarse_bpm:.2f}) | "
        f"target={tm.target_config.track_target_bpm:.1f} ({tm.target_config.harmonic_factor}x) | "
        f"stretch={tm.pre_stretch.planned_stretch_ratio:.4f}x | "
        f"active_beats={win.active_beat_count} "
        f"[{win.trim_start_sec:.2f}s .. {win.trim_end_sec:.2f}s] "
        f"(-{win.stripped_intro_sec:.1f}s head, -{win.stripped_outro_sec:.1f}s tail)\n"
        f"  saved -> {meta_path}"
    )
    return idx, tm, msg


def run_analyze_parallel(
    files: list[Path],
    target_bpm: float,
    harmonic_factors: list[float],
    work_dir: Path,
    workers: int | None = None,
) -> list[TrackMetadata]:
    """Run Steps 1-3 across `files` in parallel and return TrackMetadata list in input order."""
    total = len(files)
    max_workers = min(total, workers if workers and workers > 0 else default_workers())
    ordered: dict[int, TrackMetadata] = {}

    if max_workers <= 1:
        for idx, fpath in enumerate(files, start=1):
            _, tm, msg = _analyze_track_job(idx, total, fpath, target_bpm, harmonic_factors, work_dir)
            print(msg, flush=True)
            ordered[idx] = tm
    else:
        print(f"Analyzing {total} tracks across {max_workers} parallel workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
            futures = {
                pool.submit(
                    _analyze_track_job, idx, total, fpath, target_bpm, harmonic_factors, work_dir
                ): (idx, fpath)
                for idx, fpath in enumerate(files, start=1)
            }
            for fut in as_completed(futures):
                idx, fpath = futures[fut]
                try:
                    _, tm, msg = fut.result()
                    print(msg, flush=True)
                    ordered[idx] = tm
                except Exception as exc:
                    print(f"[{idx}/{total}] [WARN] Failed '{fpath.name}': {exc}", flush=True)

    return [ordered[i] for i in sorted(ordered)]


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Pipeline cancelled by user.")


def _run_batch(
    batch_files: list[tuple[int, Path]],
    total_label: int,
    target_bpm: float,
    harmonic_factors: list[float],
    work_dir: Path,
    min_stretch: float,
    max_stretch: float,
    workers: int | None,
    force: bool,
    rebuild: bool,
    strict: bool,
    log_fn: LogCallback | None = None,
    on_track_done: Callable[[bool], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[TrackMetadata]:
    """Process a batch of (idx, Path) items using cache + parallel worker pool in deterministic order."""
    ordered_tracks: dict[int, TrackMetadata] = {}
    pending_jobs: list[tuple[int, Path]] = []

    for idx, file_path in batch_files:
        _check_cancel(cancel_event)
        status, cached = (
            ("miss", None)
            if rebuild
            else load_cached_track(
                file_path,
                work_dir,
                target_bpm,
                harmonic_factors,
                min_stretch,
                max_stretch,
                force=force,
            )
        )
        if status == "verified" and cached is not None:
            ps = cached.post_stretch
            assert ps is not None
            _emit_log(
                f"[{idx}/{total_label}] CACHED {file_path.name} | "
                f"native={cached.pre_stretch.regression_native_bpm:.2f} -> "
                f"target={cached.target_config.track_target_bpm:.1f} ({cached.target_config.harmonic_factor}x) | "
                f"verified={ps.verified_regression_bpm:.2f} BPM | "
                f"err={ps.sync_diagnostics.mean_abs_phase_error_ms:.1f}ms",
                log_fn,
            )
            ordered_tracks[idx] = cached
            if on_track_done is not None:
                on_track_done(True)
        elif status == "skipped_deadzone" and cached is not None:
            r = cached.pre_stretch.planned_stretch_ratio
            _emit_log(
                f"[{idx}/{total_label}] [WARN] Dropping dead-zone track '{file_path.name}': "
                f"native={cached.pre_stretch.regression_native_bpm:.2f} BPM -> "
                f"target={cached.target_config.track_target_bpm:.1f} BPM ({cached.target_config.harmonic_factor}x) "
                f"requires {r:.3f}x stretch (outside [{min_stretch:.2f}x, {max_stretch:.2f}x]; "
                f"pass -force to include)",
                log_fn,
            )
            if on_track_done is not None:
                on_track_done(False)
        else:
            pending_jobs.append((idx, file_path))

    if pending_jobs:
        max_workers = min(
            len(pending_jobs),
            workers if workers and workers > 0 else default_workers(),
        )
        if max_workers <= 1:
            for idx, file_path in pending_jobs:
                _check_cancel(cancel_event)
                try:
                    _emit_log(f"[{idx}/{total_label}] Processing: {file_path.name} ...", log_fn)
                    _, tm, msg = _process_track_job(
                        idx,
                        total_label,
                        file_path,
                        target_bpm,
                        harmonic_factors,
                        work_dir,
                        min_stretch,
                        max_stretch,
                        force=force,
                    )
                    _emit_log(msg, log_fn)
                    if tm is not None:
                        ordered_tracks[idx] = tm
                    if on_track_done is not None:
                        on_track_done(tm is not None)
                except Exception as exc:
                    if strict:
                        raise
                    _emit_log(
                        f"[{idx}/{total_label}] [WARN] Skipping '{file_path.name}': {exc}",
                        log_fn,
                    )
                    if on_track_done is not None:
                        on_track_done(False)
        else:
            _emit_log(
                f"Processing {len(pending_jobs)} uncached tracks across {max_workers} parallel workers ...",
                log_fn,
            )
            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
                futures = {
                    pool.submit(
                        _process_track_job,
                        idx,
                        total_label,
                        file_path,
                        target_bpm,
                        harmonic_factors,
                        work_dir,
                        min_stretch,
                        max_stretch,
                        force,
                    ): (idx, file_path)
                    for idx, file_path in pending_jobs
                }
                for fut in as_completed(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        for f in futures:
                            f.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise InterruptedError("Pipeline cancelled by user.")
                    idx, file_path = futures[fut]
                    try:
                        _, tm, msg = fut.result()
                        _emit_log(msg, log_fn)
                        if tm is not None:
                            ordered_tracks[idx] = tm
                        if on_track_done is not None:
                            on_track_done(tm is not None)
                    except Exception as exc:
                        if strict:
                            for f in futures:
                                f.cancel()
                            raise
                        _emit_log(
                            f"[{idx}/{total_label}] [WARN] Skipping '{file_path.name}': {exc}",
                            log_fn,
                        )
                        if on_track_done is not None:
                            on_track_done(False)

    return [ordered_tracks[i] for i in sorted(ordered_tracks)]


def run_pipeline(
    input_dir: Path | Sequence[Path],
    output_mp3_path: Path,
    target_bpm: float = 180.0,
    crossfade_beats: int = 8,
    lead_in_beats: int = 8,
    lead_out_beats: int = 8,
    metronome_gain_db: float = -6.0,
    enable_metronome: bool = True,
    harmonic_factors: list[float] | None = None,
    min_stretch: float = 0.85,
    max_stretch: float = 1.35,
    work_dir: Path = Path("work"),
    limit: int | None = None,
    random_sample: int | None = None,
    seed: int | None = None,
    workers: int | None = None,
    force: bool = False,
    rebuild: bool = False,
    strict: bool = False,
    log_fn: LogCallback | None = None,
    progress_fn: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> MashupManifest:
    """Execute full 7-step mashup pipeline across input_dir with parallel track processing."""
    if harmonic_factors is None:
        harmonic_factors = list(DEFAULT_HARMONIC_FACTORS)

    _check_cancel(cancel_event)

    if random_sample is not None and random_sample > 0:
        all_files = discover_audio_files(input_dir)
        if not all_files:
            raise ValueError(f"No supported audio files found in {input_dir}")
        rng = random.Random(seed)
        rng.shuffle(all_files)
        verified_tracks: list[TrackMetadata] = []
        cursor = 0
        attempt_idx = 0
        target_goal = min(random_sample, len(all_files))
        if progress_fn is not None:
            progress_fn(0, target_goal)

        def _on_random_track_done(kept: bool) -> None:
            if kept and progress_fn is not None:
                progress_fn(min(len(verified_tracks) + 1, target_goal), target_goal)

        while len(verified_tracks) < random_sample and cursor < len(all_files):
            _check_cancel(cancel_event)
            needed = random_sample - len(verified_tracks)
            chunk = all_files[cursor : cursor + needed]
            cursor += len(chunk)
            batch_items = [(attempt_idx + i + 1, p) for i, p in enumerate(chunk)]
            attempt_idx += len(chunk)
            batch_verified = _run_batch(
                batch_files=batch_items,
                total_label=random_sample,
                target_bpm=target_bpm,
                harmonic_factors=harmonic_factors,
                work_dir=work_dir,
                min_stretch=min_stretch,
                max_stretch=max_stretch,
                workers=workers,
                force=force,
                rebuild=rebuild,
                strict=strict,
                log_fn=log_fn,
                on_track_done=_on_random_track_done,
                cancel_event=cancel_event,
            )
            verified_tracks.extend(batch_verified)
            if progress_fn is not None:
                progress_fn(min(len(verified_tracks), target_goal), target_goal)
        verified_tracks = verified_tracks[:random_sample]
    else:
        files = discover_audio_files(input_dir, limit=limit)
        if not files:
            raise ValueError(f"No supported audio files found in {input_dir}")
        total_files = len(files)
        done_counter = 0
        if progress_fn is not None:
            progress_fn(0, total_files)

        def _on_seq_track_done(_kept: bool) -> None:
            nonlocal done_counter
            done_counter += 1
            if progress_fn is not None:
                progress_fn(done_counter, total_files)

        batch_items = list(enumerate(files, start=1))
        verified_tracks = _run_batch(
            batch_files=batch_items,
            total_label=total_files,
            target_bpm=target_bpm,
            harmonic_factors=harmonic_factors,
            work_dir=work_dir,
            min_stretch=min_stretch,
            max_stretch=max_stretch,
            workers=workers,
            force=force,
            rebuild=rebuild,
            strict=strict,
            log_fn=log_fn,
            on_track_done=_on_seq_track_done,
            cancel_event=cancel_event,
        )

    _check_cancel(cancel_event)
    _emit_log(
        f"\nAssembling {len(verified_tracks)} tracks @ {target_bpm:.1f} BPM "
        f"(crossfade={crossfade_beats} beats) -> {output_mp3_path} ...",
        log_fn,
    )
    manifest = assemble_mashup(
        tracks=verified_tracks,
        output_mp3_path=output_mp3_path,
        target_bpm=target_bpm,
        crossfade_beats=crossfade_beats,
        lead_in_beats=lead_in_beats,
        lead_out_beats=lead_out_beats,
        metronome_gain_db=metronome_gain_db,
        enable_metronome=enable_metronome,
    )
    _emit_log(
        f"Done! Output: {manifest.output_mp3_path} "
        f"({manifest.total_duration_sec / 60.0:.2f} min, {manifest.total_metronome_beats} beats)",
        log_fn,
    )
    return manifest
