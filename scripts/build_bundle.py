"""Cross-platform PyInstaller build, smoke-test, and portable archive script."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44100


def _locate_tool(name: str) -> str:
    exe = f"{name}.exe" if sys.platform.startswith("win") else name
    staged = ROOT / "packaging" / "bin" / exe
    if staged.is_file():
        return str(staged)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Could not locate required tool '{exe}' in packaging/bin or PATH.")


def verify_ffmpeg_capabilities() -> None:
    """Confirm staged/system ffmpeg & ffprobe exist and support rubberband + libmp3lame."""
    ffmpeg = _locate_tool("ffmpeg")
    _locate_tool("ffprobe")

    filters = subprocess.run([ffmpeg, "-v", "error", "-filters"], capture_output=True, text=True, check=True)
    if "rubberband" not in filters.stdout:
        raise RuntimeError(f"FFmpeg at '{ffmpeg}' lacks required 'rubberband' audio filter.")

    encoders = subprocess.run([ffmpeg, "-v", "error", "-encoders"], capture_output=True, text=True, check=True)
    if "libmp3lame" not in encoders.stdout:
        raise RuntimeError(f"FFmpeg at '{ffmpeg}' lacks required 'libmp3lame' MP3 encoder.")

    print(f"Verified FFmpeg ({ffmpeg}): rubberband + libmp3lame present.", flush=True)


def _frozen_executable() -> Path:
    dist_dir = ROOT / "dist"
    if sys.platform.startswith("win"):
        return dist_dir / "RunningTracksMixer" / "RunningTracksMixer.exe"
    if sys.platform == "darwin":
        return dist_dir / "RunningTracksMixer.app" / "Contents" / "MacOS" / "RunningTracksMixer"
    return dist_dir / "RunningTracksMixer" / "RunningTracksMixer"


def _write_synthetic_song(path: Path, bpm: float, n_beats: int = 40) -> None:
    period = 60.0 / bpm
    intro_sec = 2.0
    outro_sec = 2.0
    total_samples = int(round((intro_sec + n_beats * period + outro_sec) * SAMPLE_RATE))
    buf = np.zeros((total_samples, 2), dtype=np.float32)

    t_body = np.arange(int(round(n_beats * period * SAMPLE_RATE)), dtype=np.float32) / SAMPLE_RATE
    pad = 0.08 * np.sin(2.0 * np.pi * 220.0 * t_body)
    s0 = int(round(intro_sec * SAMPLE_RATE))
    buf[s0 : s0 + len(pad), 0] += pad
    buf[s0 : s0 + len(pad), 1] += pad

    click_len = int(0.035 * SAMPLE_RATE)
    t_clk = np.arange(click_len, dtype=np.float32) / SAMPLE_RATE
    transient = 0.70 * np.sin(2.0 * np.pi * 140.0 * t_clk) * np.exp(-t_clk * 65.0)
    for b in range(n_beats):
        pos = s0 + int(round(b * period * SAMPLE_RATE))
        end = min(total_samples, pos + click_len)
        seg = transient[: end - pos]
        buf[pos:end, 0] += seg
        buf[pos:end, 1] += seg

    sf.write(str(path), buf, SAMPLE_RATE, subtype="PCM_16")


def run_smoke_tests(exe_path: Path) -> None:
    """Verify frozen GUI startup and full multi-worker DSP mashup without system PATH ffmpeg."""
    if not exe_path.is_file():
        raise FileNotFoundError(f"Frozen executable not found: {exe_path}")

    # Check A: Frozen PySide6 GUI offscreen startup
    gui_env = os.environ.copy()
    gui_env["QT_QPA_PLATFORM"] = "offscreen"
    gui_env["PYTHONFAULTHANDLER"] = "1"
    gui_res = subprocess.run(
        [str(exe_path), "--self-test-gui"],
        env=gui_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if gui_res.returncode != 0 or "SELF_TEST_GUI_OK" not in gui_res.stdout:
        raise RuntimeError(
            f"Frozen GUI self-test failed (rc={gui_res.returncode}):\n"
            f"STDOUT:\n{gui_res.stdout}\nSTDERR:\n{gui_res.stderr}"
        )
    print("Smoke Test A passed: Frozen PySide6 GUI initialized.", flush=True)

    # Check B: Frozen 2-worker DSP + bundled FFmpeg build with stripped PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_dir = tmp_dir / "tracks"
        in_dir.mkdir(parents=True, exist_ok=True)
        work_dir = tmp_dir / "work"
        out_mp3 = tmp_dir / "mashup.mp3"

        _write_synthetic_song(in_dir / "s1_80bpm.wav", bpm=80.0, n_beats=40)
        _write_synthetic_song(in_dir / "s2_165bpm.wav", bpm=165.0, n_beats=64)

        dsp_env = os.environ.copy()
        dsp_env["PATH"] = "C:\\Windows\\System32" if sys.platform.startswith("win") else ""
        dsp_env["PYTHONFAULTHANDLER"] = "1"
        dsp_env.pop("RTMIX_BIN_DIR", None)

        cmd = [
            str(exe_path),
            "build",
            "-i",
            str(in_dir),
            "-o",
            str(out_mp3),
            "--work-dir",
            str(work_dir),
            "-j",
            "2",
        ]
        res = subprocess.run(
            cmd,
            env=dsp_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if res.returncode != 0 or not out_mp3.is_file() or out_mp3.stat().st_size <= 10_000:
            diag_cmd = [
                str(exe_path),
                "build",
                "-i",
                str(in_dir),
                "-o",
                str(out_mp3),
                "--work-dir",
                str(work_dir),
                "-j",
                "1",
                "--strict",
            ]
            diag_res = subprocess.run(
                diag_cmd,
                env=dsp_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=180,
            )
            raise RuntimeError(
                f"Frozen DSP build failed (rc={res.returncode}):\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n"
                f"--- Single-worker diagnostic (-j 1 --strict, rc={diag_res.returncode}) ---\n"
                f"STDOUT:\n{diag_res.stdout}\nSTDERR:\n{diag_res.stderr}"
            )
        print(
            f"Smoke Test B passed: Frozen 2-worker mashup built ({out_mp3.stat().st_size} bytes).",
            flush=True,
        )


def build_archive() -> Path:
    """Package `dist/RunningTracksMixer` into a portable OS-specific release archive."""
    dist_dir = ROOT / "dist"
    if sys.platform.startswith("win"):
        out = Path(
            shutil.make_archive(
                str(dist_dir / "RunningTracksMixer-windows-x64"),
                "zip",
                root_dir=str(dist_dir),
                base_dir="RunningTracksMixer",
            )
        )
    elif sys.platform == "darwin":
        out = dist_dir / "RunningTracksMixer-macos-arm64.dmg"
        if out.exists():
            out.unlink()
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "Running Tracks Mixer",
                "-srcfolder",
                str(dist_dir / "RunningTracksMixer.app"),
                "-ov",
                "-format",
                "UDZO",
                str(out),
            ],
            check=True,
        )
    else:
        out = Path(
            shutil.make_archive(
                str(dist_dir / "RunningTracksMixer-linux-x64"),
                "gztar",
                root_dir=str(dist_dir),
                base_dir="RunningTracksMixer",
            )
        )
    print(f"Created portable archive: {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portable RunningTracksMixer bundle.")
    parser.add_argument("--smoke-test", action="store_true", help="Run frozen GUI + DSP smoke tests")
    parser.add_argument("--archive", action="store_true", help="Create portable OS release archive")
    args = parser.parse_args()

    verify_ffmpeg_capabilities()

    spec_path = ROOT / "packaging" / "rtmix.spec"
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)],
        cwd=str(ROOT),
        check=True,
    )

    if sys.platform == "darwin":
        app_path = ROOT / "dist" / "RunningTracksMixer.app"
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app_path)], check=True)

    exe_path = _frozen_executable()
    if args.smoke_test:
        run_smoke_tests(exe_path)

    if args.archive:
        build_archive()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
