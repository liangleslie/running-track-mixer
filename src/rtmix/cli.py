"""CLI entrypoint for Running Tracks Mixer (`uv run rtmix`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rtmix.models import TrackMetadata
from rtmix.pipeline import (
    default_workers,
    discover_audio_files,
    run_analyze_parallel,
    run_pipeline,
)


def _parse_single_factor(token: str) -> float:
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num.strip()) / float(den.strip())
    val = float(token)
    if abs(val - 2.0 / 3.0) < 1e-3:
        return 2.0 / 3.0
    return val


def _parse_harmonic_factors(raw: str) -> list[float]:
    factors = [_parse_single_factor(x.strip()) for x in raw.split(",") if x.strip()]
    if not factors or any(f <= 0 for f in factors):
        raise ValueError(f"Invalid --harmonic-factors: {raw}")
    return factors


def cmd_build(args: argparse.Namespace) -> int:
    factors = _parse_harmonic_factors(args.harmonic_factors)
    run_pipeline(
        input_dir=Path(args.input_dir),
        output_mp3_path=Path(args.output),
        target_bpm=args.bpm,
        crossfade_beats=args.crossfade_beats,
        lead_in_beats=args.lead_in_beats,
        lead_out_beats=args.lead_out_beats,
        metronome_gain_db=args.metronome_gain_db,
        enable_metronome=not args.no_metronome,
        harmonic_factors=factors,
        min_stretch=args.min_stretch,
        max_stretch=args.max_stretch,
        work_dir=Path(args.work_dir),
        limit=args.limit,
        random_sample=args.random,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
        rebuild=args.rebuild,
        strict=args.strict,
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    factors = _parse_harmonic_factors(args.harmonic_factors)
    files = discover_audio_files(
        Path(args.input_path),
        limit=args.limit,
        random_sample=args.random,
        seed=args.seed,
    )
    if not files:
        print(f"No audio files found at {args.input_path}", file=sys.stderr)
        return 1

    run_analyze_parallel(
        files=files,
        target_bpm=args.bpm,
        harmonic_factors=factors,
        work_dir=Path(args.work_dir),
        workers=args.workers,
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    meta_files = sorted(work_dir.glob("*/metadata.json"))
    if not meta_files:
        print(f"No metadata.json files found in {work_dir}", file=sys.stderr)
        return 1

    failures = 0
    for mpath in meta_files:
        tm = TrackMetadata.load(mpath)
        ps = tm.post_stretch
        if ps is None:
            print(f"[INCOMPLETE] {tm.source.file_name}: status={tm.status} (no post_stretch data)")
            failures += 1
            continue
        ok = ps.sync_diagnostics.is_within_tolerance
        tag = "OK" if ok else "DRIFT"
        if not ok:
            failures += 1
        print(
            f"[{tag}] {tm.source.file_name} | "
            f"target={tm.target_config.track_target_bpm:.2f} | "
            f"verified={ps.verified_regression_bpm:.2f} (err={ps.bpm_error:.3f} BPM) | "
            f"phase_err_mean={ps.sync_diagnostics.mean_abs_phase_error_ms:.2f}ms"
        )
    return 1 if failures > 0 else 0


def build_parser() -> argparse.ArgumentParser:
    def_workers = default_workers()
    parser = argparse.ArgumentParser(
        prog="rtmix",
        description="Running Tracks Mixer: tempo-locked audio mashup with sample-accurate metronome",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rtmix build
    p_build = sub.add_parser("build", help="Run full pipeline to build tempo-locked mashup MP3")
    p_build.add_argument("-i", "--input-dir", required=True, help="Input directory or audio file")
    p_build.add_argument("-o", "--output", default="output/mashup.mp3", help="Output MP3 path")
    p_build.add_argument("-b", "--bpm", type=float, default=180.0, help="Target running BPM (default: 180.0)")
    p_build.add_argument(
        "--crossfade-beats",
        type=int,
        default=8,
        help="Beat-synced crossfade overlap in metronome beats (default: 8)",
    )
    p_build.add_argument(
        "--lead-in-beats",
        type=int,
        default=8,
        help="Solo metronome beats before first track (default: 8)",
    )
    p_build.add_argument(
        "--lead-out-beats",
        type=int,
        default=8,
        help="Solo metronome beats after final track (default: 8)",
    )
    p_build.add_argument(
        "--metronome-gain-db",
        type=float,
        default=-6.0,
        help="Metronome click peak level in dBFS (default: -6.0)",
    )
    p_build.add_argument(
        "--no-metronome",
        action="store_true",
        help="Disable metronome click overlay",
    )
    p_build.add_argument(
        "--harmonic-factors",
        default="0.5,1.0,2.0",
        help="Comma-separated binary harmonic multipliers relative to target BPM (default: 0.5,1.0,2.0)",
    )
    p_build.add_argument(
        "--min-stretch",
        type=float,
        default=0.85,
        help="Minimum allowed stretch ratio before skipping dead-zone track (default: 0.85)",
    )
    p_build.add_argument(
        "--max-stretch",
        type=float,
        default=1.35,
        help="Maximum allowed stretch ratio before skipping dead-zone track (default: 1.35)",
    )
    p_build.add_argument("--work-dir", default="work", help="Working cache directory")
    p_build.add_argument(
        "-n", "--limit", type=int, default=None, help="Process first N tracks from input-dir"
    )
    p_build.add_argument(
        "-r",
        "-random",
        "--random",
        type=int,
        default=None,
        help="Randomly pick N valid tracks for the mashup (replaces dropped dead-zone tracks automatically)",
    )
    p_build.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible --random track selection",
    )
    p_build.add_argument(
        "-j",
        "--workers",
        type=int,
        default=def_workers,
        help=f"Number of parallel worker processes (default: {def_workers})",
    )
    p_build.add_argument(
        "-f",
        "-force",
        "--force",
        action="store_true",
        help="Force inclusion of dead-zone BPM tracks outside [min-stretch, max-stretch]",
    )
    p_build.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cached stretched tracks in work-dir and re-process from scratch",
    )
    p_build.add_argument("--strict", action="store_true", help="Abort on first track failure")
    p_build.set_defaults(func=cmd_build)

    # rtmix analyze
    p_analyze = sub.add_parser("analyze", help="Run Steps 1-3 pre-stretch analysis only")
    p_analyze.add_argument("input_path", help="Input audio file or directory")
    p_analyze.add_argument("-b", "--bpm", type=float, default=180.0, help="Target running BPM (default: 180.0)")
    p_analyze.add_argument(
        "--harmonic-factors",
        default="0.5,1.0,2.0",
        help="Comma-separated harmonic multipliers (default: 0.5,1.0,2.0)",
    )
    p_analyze.add_argument("--work-dir", default="work", help="Working cache directory")
    p_analyze.add_argument("-n", "--limit", type=int, default=None, help="Limit number of tracks")
    p_analyze.add_argument(
        "-r",
        "-random",
        "--random",
        type=int,
        default=None,
        help="Randomly pick N tracks to analyze",
    )
    p_analyze.add_argument("--seed", type=int, default=None, help="Optional random seed")
    p_analyze.add_argument(
        "-j",
        "--workers",
        type=int,
        default=def_workers,
        help=f"Number of parallel worker processes (default: {def_workers})",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # rtmix verify
    p_verify = sub.add_parser("verify", help="Inspect cached track metadata and sync diagnostics")
    p_verify.add_argument("work_dir", nargs="?", default="work", help="Working cache directory")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
