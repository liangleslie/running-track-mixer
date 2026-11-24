"""Frozen PyInstaller entrypoint for Running Tracks Mixer (GUI + CLI + worker pools)."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import shutil
import sys
import tempfile

# Avoid OpenMP/workqueue collision and read-only bundle / parallel Numba cache races
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
_NUMBA_TMP_DIR = os.path.join(tempfile.gettempdir(), f"rtmix_numba_{os.getpid()}")
os.environ["NUMBA_CACHE_DIR"] = _NUMBA_TMP_DIR
atexit.register(shutil.rmtree, _NUMBA_TMP_DIR, True)

# Must execute before importing PySide6, librosa, scipy, or rtmix so frozen
# ProcessPoolExecutor workers terminate without re-launching the Qt UI.
if __name__ == "__main__":
    multiprocessing.freeze_support()


def _ensure_std_streams() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _run_gui_self_test() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from rtmix.gui import RtmixMainWindow, apply_dark_fusion_theme

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication([])
    apply_dark_fusion_theme(app)
    win = RtmixMainWindow()
    app.processEvents()
    win.close()
    print("SELF_TEST_GUI_OK", flush=True)
    return 0


def main() -> int:
    _ensure_std_streams()
    user_args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]

    if user_args == ["--self-test-gui"]:
        return _run_gui_self_test()

    if user_args:
        from rtmix.cli import main as cli_main

        return int(cli_main(user_args))

    from rtmix.gui import main as gui_main

    return int(gui_main())


if __name__ == "__main__":
    raise SystemExit(main())
