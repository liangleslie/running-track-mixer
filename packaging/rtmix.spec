# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller --onedir / macOS .app spec for Running Tracks Mixer."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821
IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"


def _find_tool(name: str) -> str:
    exe = f"{name}.exe" if IS_WIN else name
    staged = ROOT / "packaging" / "bin" / exe
    if staged.is_file():
        return str(staged)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Missing required binary '{exe}'. Place it in packaging/bin/ or install it on PATH."
    )


ffmpeg_bin = _find_tool("ffmpeg")
ffprobe_bin = _find_tool("ffprobe")

datas = []
binaries = [
    (ffmpeg_bin, "bin"),
    (ffprobe_bin, "bin"),
]
hiddenimports = []

# librosa and scipy use lazy_loader stub (.pyi) inspection at import time
datas += collect_data_files("librosa", excludes=["**/__pycache__"])
hiddenimports += collect_submodules("librosa")
datas += collect_data_files("scipy", includes=["**/*.pyi"])

try:
    datas += collect_data_files("_soundfile_data", excludes=["**/__pycache__"])
    binaries += collect_dynamic_libs("_soundfile_data")
except Exception:
    pass

excludes = [
    "tkinter",
    "_tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.Qt3DCore",
    "PySide6.QtMultimedia",
    "PySide6.QtPdf",
    "PySide6.QtSql",
    "PySide6.QtTest",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry_gui.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)
mac_entitlements = str(ROOT / "packaging" / "entitlements.plist") if IS_MAC else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RunningTracksMixer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    entitlements_file=mac_entitlements,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RunningTracksMixer",
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="RunningTracksMixer.app",
        bundle_identifier="com.rtmix.runningtracksmixer",
        entitlements_file=mac_entitlements,
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleDisplayName": "Running Tracks Mixer",
            "CFBundleShortVersionString": "0.1.0",
            "LSMinimumSystemVersion": "12.0",
        },
    )
