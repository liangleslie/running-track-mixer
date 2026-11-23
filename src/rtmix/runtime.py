"""Runtime environment helpers for frozen (PyInstaller) and development execution."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def is_frozen() -> bool:
    """Return True when executing inside a frozen PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def resolve_binary(name: str) -> str:
    """Locate `ffmpeg` or `ffprobe`, preferring bundled binaries before system PATH."""
    exe = f"{name}.exe" if sys.platform.startswith("win") else name
    candidates: list[Path] = []

    env_dir = os.environ.get("RTMIX_BIN_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / exe)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "bin" / exe)

    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "bin" / exe)
    candidates.append(exe_dir / exe)

    # Local staged binary when running from source checkout
    candidates.append(Path(__file__).resolve().parents[2] / "packaging" / "bin" / exe)

    for cand in candidates:
        if cand.is_file():
            return str(cand)

    found = shutil.which(name)
    if found:
        return found

    # macOS Finder-launched .app bundles often omit Homebrew paths from PATH
    if sys.platform == "darwin":
        for mac_dir in (
            "/opt/homebrew/opt/ffmpeg-full/bin",
            "/usr/local/opt/ffmpeg-full/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ):
            cand = Path(mac_dir) / name
            if cand.is_file():
                return str(cand)

    return name


def subprocess_kwargs() -> dict[str, Any]:
    """Return OS-specific subprocess keyword arguments (suppresses console flash on Windows)."""
    if sys.platform.startswith("win"):
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def run_checked(cmd: list[str], *, text: bool = False) -> subprocess.CompletedProcess[Any]:
    """Run external binary (`ffmpeg`/`ffprobe`) and include stderr in exception on failure."""
    res = subprocess.run(cmd, capture_output=True, text=text, **subprocess_kwargs())
    if res.returncode != 0:
        raw_err = res.stderr if isinstance(res.stderr, str) else res.stderr.decode("utf-8", errors="replace")
        detail = raw_err.strip() or f"signal/exit code {res.returncode}"
        raise RuntimeError(f"{Path(cmd[0]).name} failed (rc={res.returncode}): {detail}")
    return res


def default_work_dir() -> Path:
    """Return default cache directory (CWD `work` in dev, OS user cache dir when frozen)."""
    if not is_frozen():
        return Path("work")
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "rtmix" / "work"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "rtmix" / "work"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "rtmix" / "work"


def default_output_mp3_path() -> Path:
    """Return default output MP3 path (CWD `output/mashup.mp3` in dev, Music folder when frozen)."""
    if not is_frozen():
        return Path("output/mashup.mp3")
    return Path.home() / "Music" / "RunningTracksMixer" / "mashup.mp3"
