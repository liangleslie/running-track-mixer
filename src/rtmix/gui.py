"""PySide6 (Qt 6) desktop GUI for Running Tracks Mixer (`rtmix`)."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import threading
from typing import Any
import sys

from PySide6.QtCore import QObject, QSignalBlocker, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFontDatabase,
    QIntValidator,
    QPalette,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from rtmix.analyzer import DEFAULT_HARMONIC_FACTORS
from rtmix.pipeline import (
    SUPPORTED_EXTENSIONS,
    default_workers,
    discover_audio_files,
    run_pipeline,
)
from rtmix.runtime import default_output_mp3_path


def open_folder_in_explorer(folder: Path) -> None:
    """Open `folder` in the OS native file explorer (Windows/macOS/Linux)."""
    target = folder.resolve()
    target.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def _extract_dropped_local_paths(mime_data: Any) -> list[Path]:
    """Extract existing local filesystem paths from Qt drag-and-drop MIME data."""
    if mime_data is None or not mime_data.hasUrls():
        return []
    paths: list[Path] = []
    for url in mime_data.urls():
        if url.isLocalFile():
            p = Path(url.toLocalFile())
            if p.exists():
                paths.append(p)
    return paths


class DropQueueListWidget(QListWidget):
    """Queue list widget accepting local file and folder drops from OS file managers."""

    paths_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(10)
        self.setFont(fixed_font)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = _extract_dropped_local_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.paths_dropped.emit(paths)
        else:
            event.ignore()


class PipelineSignals(QObject):
    """Thread-safe signals bridging background worker execution to the Qt main thread."""

    log_emitted = Signal(str)
    progress_changed = Signal(int, int)
    finished_ok = Signal(object)
    cancelled = Signal()
    failed = Signal(str)


class RtmixMainWindow(QMainWindow):
    """Main Qt 6 window exposing the Running Tracks Mixer pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Running Tracks Mixer (rtmix)")
        self.resize(920, 820)
        self.setMinimumSize(760, 660)
        self.setAcceptDrops(True)

        self.queued_paths: list[Path] = []
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self.signals = PipelineSignals(self)
        self.signals.log_emitted.connect(self._on_worker_log)
        self.signals.progress_changed.connect(self._on_worker_progress)
        self.signals.finished_ok.connect(self._on_worker_done)
        self.signals.cancelled.connect(self._on_worker_cancelled)
        self.signals.failed.connect(self._on_worker_error)

        self._build_ui()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = _extract_dropped_local_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self._add_paths_to_queue(paths)
        else:
            event.ignore()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # 1. Input Queue Section
        input_group = QGroupBox("1. Audio Tracks / Folders", central)
        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(12, 12, 12, 12)
        input_layout.setSpacing(8)

        queue_toolbar = QHBoxLayout()
        queue_toolbar.setSpacing(8)

        add_files_btn = QPushButton("+ Add Files")
        add_files_btn.setObjectName("primaryButton")
        add_files_btn.clicked.connect(self._on_add_files)
        queue_toolbar.addWidget(add_files_btn)

        add_folder_btn = QPushButton("+ Add Folder")
        add_folder_btn.setObjectName("primaryButton")
        add_folder_btn.clicked.connect(self._on_add_folder)
        queue_toolbar.addWidget(add_folder_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_selected)
        queue_toolbar.addWidget(remove_btn)

        clear_btn = QPushButton("Clear Queue")
        clear_btn.clicked.connect(self._on_clear_queue)
        queue_toolbar.addWidget(clear_btn)

        queue_toolbar.addStretch(1)

        self.queue_count_label = QLabel("0 tracks queued — Drag & drop files/folders, or use buttons")
        self.queue_count_label.setObjectName("mutedLabel")
        queue_toolbar.addWidget(self.queue_count_label)

        input_layout.addLayout(queue_toolbar)

        self.queue_list = DropQueueListWidget(input_group)
        self.queue_list.setMinimumHeight(110)
        self.queue_list.setMaximumHeight(160)
        self.queue_list.paths_dropped.connect(self._add_paths_to_queue)
        input_layout.addWidget(self.queue_list)

        root_layout.addWidget(input_group)

        # 2. Core Controls Section
        controls_group = QGroupBox("2. Mix && Tempo Settings", central)
        grid = QGridLayout(controls_group)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        # Row 0: Output MP3
        grid.addWidget(QLabel("Output MP3:"), 0, 0)
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.output_edit = QLineEdit(str(default_output_mp3_path()))
        out_row.addWidget(self.output_edit, 1)

        browse_out_btn = QPushButton("Browse...")
        browse_out_btn.setObjectName("primaryButton")
        browse_out_btn.clicked.connect(self._on_browse_output)
        out_row.addWidget(browse_out_btn)

        open_out_btn = QPushButton("Open Folder")
        open_out_btn.clicked.connect(self._on_open_output_folder)
        out_row.addWidget(open_out_btn)
        grid.addLayout(out_row, 0, 1, 1, 2)

        # Row 1: Target Tempo slider + spinbox
        grid.addWidget(QLabel("Target Tempo:"), 1, 0)
        self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpm_slider.setRange(120, 220)
        self.bpm_slider.setValue(180)
        self.bpm_slider.valueChanged.connect(self._on_bpm_slider_changed)
        grid.addWidget(self.bpm_slider, 1, 1)

        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(120.0, 220.0)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setSingleStep(1.0)
        self.bpm_spin.setSuffix(" BPM")
        self.bpm_spin.setValue(180.0)
        self.bpm_spin.setFixedWidth(120)
        self.bpm_spin.valueChanged.connect(self._on_bpm_spin_changed)
        grid.addWidget(self.bpm_spin, 1, 2)

        # Row 2: CPU Workers slider + spinbox
        max_cpus = max(1, os.cpu_count() or 4)
        def_cpus = min(max_cpus, default_workers())
        grid.addWidget(QLabel("CPU Workers:"), 2, 0)
        self.cores_slider = QSlider(Qt.Orientation.Horizontal)
        self.cores_slider.setRange(1, max_cpus)
        self.cores_slider.setValue(def_cpus)
        self.cores_slider.valueChanged.connect(self._on_cores_slider_changed)
        grid.addWidget(self.cores_slider, 2, 1)

        self.cores_spin = QSpinBox()
        self.cores_spin.setRange(1, max_cpus)
        self.cores_spin.setSuffix(" cores")
        self.cores_spin.setValue(def_cpus)
        self.cores_spin.setFixedWidth(120)
        self.cores_spin.valueChanged.connect(self._on_cores_spin_changed)
        grid.addWidget(self.cores_spin, 2, 2)

        # Row 3: Metronome Click toggle + Gain slider + label
        self.metronome_cb = QCheckBox("Metronome Click")
        self.metronome_cb.setChecked(True)
        self.metronome_cb.toggled.connect(self._on_metronome_toggled)
        grid.addWidget(self.metronome_cb, 3, 0)

        self.metro_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.metro_gain_slider.setRange(-36, 0)  # 0.5 dB increments: -18.0 dB .. 0.0 dB
        self.metro_gain_slider.setSingleStep(1)
        self.metro_gain_slider.setPageStep(2)
        self.metro_gain_slider.setValue(-12)  # -6.0 dB
        self.metro_gain_slider.valueChanged.connect(self._on_metro_gain_changed)
        grid.addWidget(self.metro_gain_slider, 3, 1)

        self.metro_gain_label = QLabel("-6.0 dB")
        self.metro_gain_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.metro_gain_label.setFixedWidth(120)
        grid.addWidget(self.metro_gain_label, 3, 2)

        # Row 4: Pick Random N + Seed
        self.random_cb = QCheckBox("Pick Random N:")
        self.random_cb.setChecked(False)
        self.random_cb.toggled.connect(self._on_random_toggled)
        grid.addWidget(self.random_cb, 4, 0)

        rand_row = QHBoxLayout()
        rand_row.setSpacing(10)
        self.random_count_spin = QSpinBox()
        self.random_count_spin.setRange(1, 99999)
        self.random_count_spin.setValue(10)
        self.random_count_spin.setSuffix(" tracks")
        self.random_count_spin.setFixedWidth(120)
        self.random_count_spin.setEnabled(False)
        rand_row.addWidget(self.random_count_spin)

        rand_row.addWidget(QLabel("Optional Seed:"))
        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("auto")
        self.seed_edit.setValidator(QIntValidator(0, 2147483647, self))
        self.seed_edit.setFixedWidth(100)
        self.seed_edit.setEnabled(False)
        rand_row.addWidget(self.seed_edit)
        rand_row.addStretch(1)
        grid.addLayout(rand_row, 4, 1, 1, 2)

        root_layout.addWidget(controls_group)

        # 3. Collapsible Advanced Settings Section
        adv_wrapper = QFrame(central)
        adv_wrapper.setObjectName("cardFrame")
        adv_wrapper_layout = QVBoxLayout(adv_wrapper)
        adv_wrapper_layout.setContentsMargins(10, 6, 10, 8)
        adv_wrapper_layout.setSpacing(6)

        self.adv_toggle_btn = QToolButton(adv_wrapper)
        self.adv_toggle_btn.setText(
            "Advanced Settings (Min/Max Stretch, Force Dead-Zone, Crossfade, Cache)"
        )
        self.adv_toggle_btn.setCheckable(True)
        self.adv_toggle_btn.setChecked(False)
        self.adv_toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.adv_toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_toggle_btn.setAutoRaise(True)
        self.adv_toggle_btn.toggled.connect(self._toggle_advanced)
        adv_wrapper_layout.addWidget(self.adv_toggle_btn)

        self.adv_frame = QFrame(adv_wrapper)
        self.adv_frame.setObjectName("subCardFrame")
        adv_grid = QGridLayout(self.adv_frame)
        adv_grid.setContentsMargins(12, 10, 12, 10)
        adv_grid.setHorizontalSpacing(14)
        adv_grid.setVerticalSpacing(8)
        adv_grid.setColumnStretch(5, 1)

        adv_grid.addWidget(QLabel("Min Stretch:"), 0, 0)
        self.min_stretch_spin = QDoubleSpinBox()
        self.min_stretch_spin.setRange(0.20, 2.00)
        self.min_stretch_spin.setSingleStep(0.05)
        self.min_stretch_spin.setDecimals(2)
        self.min_stretch_spin.setSuffix("x")
        self.min_stretch_spin.setValue(0.85)
        adv_grid.addWidget(self.min_stretch_spin, 0, 1)

        adv_grid.addWidget(QLabel("Max Stretch:"), 0, 2)
        self.max_stretch_spin = QDoubleSpinBox()
        self.max_stretch_spin.setRange(0.50, 3.00)
        self.max_stretch_spin.setSingleStep(0.05)
        self.max_stretch_spin.setDecimals(2)
        self.max_stretch_spin.setSuffix("x")
        self.max_stretch_spin.setValue(1.35)
        adv_grid.addWidget(self.max_stretch_spin, 0, 3)

        self.force_cb = QCheckBox("Force Dead-Zone Tracks (-force)")
        adv_grid.addWidget(self.force_cb, 0, 4, 1, 2)

        adv_grid.addWidget(QLabel("Crossfade Beats:"), 1, 0)
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(0, 64)
        self.crossfade_spin.setSuffix(" beats")
        self.crossfade_spin.setValue(8)
        adv_grid.addWidget(self.crossfade_spin, 1, 1)

        adv_grid.addWidget(QLabel("Lead-in / Out Beats:"), 1, 2)
        self.lead_beats_spin = QSpinBox()
        self.lead_beats_spin.setRange(0, 64)
        self.lead_beats_spin.setSuffix(" beats")
        self.lead_beats_spin.setValue(8)
        adv_grid.addWidget(self.lead_beats_spin, 1, 3)

        self.rebuild_cb = QCheckBox("Rebuild Cache (--rebuild)")
        adv_grid.addWidget(self.rebuild_cb, 1, 4, 1, 2)

        self.adv_frame.setVisible(False)
        adv_wrapper_layout.addWidget(self.adv_frame)
        root_layout.addWidget(adv_wrapper)

        # 4. Action & Progress Bar Section
        action_frame = QFrame(central)
        action_frame.setObjectName("cardFrame")
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(10)

        self.build_btn = QPushButton("Build Mashup")
        self.build_btn.setObjectName("primaryButton")
        self.build_btn.setMinimumSize(150, 36)
        self.build_btn.clicked.connect(self._on_start_build)
        action_layout.addWidget(self.build_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setMinimumSize(96, 36)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_build)
        action_layout.addWidget(self.cancel_btn)

        prog_col = QVBoxLayout()
        prog_col.setSpacing(4)
        self.status_label = QLabel("Ready")
        prog_col.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        prog_col.addWidget(self.progress_bar)
        action_layout.addLayout(prog_col, 1)

        root_layout.addWidget(action_frame)

        # 5. Live Log Console Section
        log_group = QGroupBox("Pipeline Log", central)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 12, 12, 12)

        self.log_edit = QPlainTextEdit(log_group)
        self.log_edit.setReadOnly(True)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(10)
        self.log_edit.setFont(fixed_font)
        log_layout.addWidget(self.log_edit)

        root_layout.addWidget(log_group, 1)

    # --- UI callbacks ---

    def _toggle_advanced(self, checked: bool | None = None) -> None:
        is_open = not self.adv_frame.isVisible() if checked is None else bool(checked)
        with QSignalBlocker(self.adv_toggle_btn):
            self.adv_toggle_btn.setChecked(is_open)
        self.adv_toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if is_open else Qt.ArrowType.RightArrow
        )
        self.adv_frame.setVisible(is_open)

    def _on_bpm_slider_changed(self, val: int) -> None:
        with QSignalBlocker(self.bpm_spin):
            self.bpm_spin.setValue(float(val))

    def _on_bpm_spin_changed(self, val: float) -> None:
        with QSignalBlocker(self.bpm_slider):
            self.bpm_slider.setValue(round(val))

    def _on_cores_slider_changed(self, val: int) -> None:
        with QSignalBlocker(self.cores_spin):
            self.cores_spin.setValue(val)

    def _on_cores_spin_changed(self, val: int) -> None:
        with QSignalBlocker(self.cores_slider):
            self.cores_slider.setValue(val)

    def _on_metronome_toggled(self, checked: bool) -> None:
        self.metro_gain_slider.setEnabled(checked)
        self._update_metro_gain_label()

    def _on_metro_gain_changed(self, _val: int) -> None:
        self._update_metro_gain_label()

    def _update_metro_gain_label(self) -> None:
        if not self.metronome_cb.isChecked():
            self.metro_gain_label.setText("Muted")
        else:
            db = self.metro_gain_slider.value() / 2.0
            self.metro_gain_label.setText(f"{db:.1f} dB")

    def _on_random_toggled(self, checked: bool) -> None:
        self.random_count_spin.setEnabled(checked)
        self.seed_edit.setEnabled(checked)

    def _add_paths_to_queue(self, paths: list[Path]) -> None:
        existing_resolved = {p.resolve() for p in self.queued_paths}
        added = 0
        for p in paths:
            if not p.exists():
                continue
            if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            res = p.resolve()
            if res not in existing_resolved:
                existing_resolved.add(res)
                self.queued_paths.append(p)
                added += 1
        if added > 0:
            self._refresh_queue_display()

    def _refresh_queue_display(self) -> None:
        self.queue_list.clear()
        for p in self.queued_paths:
            tag = "[DIR] " if p.is_dir() else "[FILE]"
            self.queue_list.addItem(f"{tag} {p}")
        if not self.queued_paths:
            self.queue_count_label.setText(
                "0 tracks queued — Drag & drop files/folders, or use buttons"
            )
            return
        try:
            discovered = discover_audio_files(self.queued_paths)
            count = len(discovered)
        except Exception:
            count = 0
        self.queue_count_label.setText(
            f"{len(self.queued_paths)} item(s) queued — {count} audio track(s) discovered"
        )

    def _on_add_files(self) -> None:
        exts = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Tracks",
            "",
            f"Audio Files ({exts});;All Files (*)",
        )
        if selected:
            self._add_paths_to_queue([Path(s) for s in selected])

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Audio Tracks")
        if folder:
            self._add_paths_to_queue([Path(folder)])

    def _on_remove_selected(self) -> None:
        rows = sorted({item.row() for item in self.queue_list.selectedIndexes()}, reverse=True)
        for idx in rows:
            if 0 <= idx < len(self.queued_paths):
                self.queued_paths.pop(idx)
        if rows:
            self._refresh_queue_display()

    def _on_clear_queue(self) -> None:
        self.queued_paths.clear()
        self._refresh_queue_display()

    def _on_browse_output(self) -> None:
        initial = Path(self.output_edit.text().strip() or str(default_output_mp3_path()))
        start_dir = initial.parent if initial.parent.exists() else Path.cwd()
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Mashup MP3",
            str(start_dir / initial.name),
            "MP3 Audio (*.mp3);;All Files (*)",
        )
        if chosen:
            self.output_edit.setText(chosen)

    def _on_open_output_folder(self) -> None:
        out_path = Path(self.output_edit.text().strip() or str(default_output_mp3_path()))
        try:
            open_folder_in_explorer(out_path.parent)
        except Exception as exc:
            QMessageBox.critical(self, "Open Folder Error", str(exc))

    def _on_worker_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def _on_worker_progress(self, done: int, total: int) -> None:
        value = round(1000.0 * done / total) if total > 0 else 0
        self.progress_bar.setValue(min(1000, max(0, value)))
        self.status_label.setText(f"Processing tracks: {done} / {total}")

    def _on_worker_done(self, manifest: Any) -> None:
        self.progress_bar.setValue(1000)
        mins = manifest.total_duration_sec / 60.0
        self.status_label.setText(f"Done! {manifest.output_mp3_path} ({mins:.2f} min)")
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_worker_cancelled(self) -> None:
        self.status_label.setText("Cancelled by user.")
        self.log_edit.appendPlainText("[CANCELLED] Pipeline stopped by user.")
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_worker_error(self, error_msg: str) -> None:
        self.status_label.setText(f"Error: {error_msg}")
        self.log_edit.appendPlainText(f"[ERROR] {error_msg}")
        self.build_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.critical(self, "Build Failed", error_msg)

    def _on_start_build(self) -> None:
        if not self.queued_paths:
            QMessageBox.warning(
                self,
                "No Tracks Queued",
                "Please add at least one audio file or folder (via buttons or drag & drop).",
            )
            return

        min_stretch = float(self.min_stretch_spin.value())
        max_stretch = float(self.max_stretch_spin.value())
        if min_stretch <= 0 or max_stretch <= min_stretch:
            QMessageBox.critical(
                self,
                "Invalid Settings",
                "Min stretch must be > 0 and less than Max stretch.",
            )
            return

        random_sample: int | None = None
        seed: int | None = None
        if self.random_cb.isChecked():
            random_sample = int(self.random_count_spin.value())
            raw_seed = self.seed_edit.text().strip()
            if raw_seed:
                seed = int(raw_seed)

        target_bpm = float(self.bpm_spin.value())
        workers = int(self.cores_spin.value())
        enable_metronome = bool(self.metronome_cb.isChecked())
        metronome_gain_db = self.metro_gain_slider.value() / 2.0
        crossfade_beats = int(self.crossfade_spin.value())
        lead_beats = int(self.lead_beats_spin.value())
        output_mp3 = Path(self.output_edit.text().strip() or str(default_output_mp3_path()))
        force = bool(self.force_cb.isChecked())
        rebuild = bool(self.rebuild_cb.isChecked())
        inputs = list(self.queued_paths)

        self._cancel_event.clear()
        self.build_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting pipeline...")
        self.log_edit.clear()
        self.log_edit.appendPlainText(
            f"Starting build: target={target_bpm:.1f} BPM | workers={workers} | "
            f"stretch=[{min_stretch:.2f}x, {max_stretch:.2f}x] | force={force} | "
            f"random={random_sample or 'all'}"
        )

        def _worker() -> None:
            try:
                manifest = run_pipeline(
                    input_dir=inputs,
                    output_mp3_path=output_mp3,
                    target_bpm=target_bpm,
                    crossfade_beats=crossfade_beats,
                    lead_in_beats=lead_beats,
                    lead_out_beats=lead_beats,
                    metronome_gain_db=metronome_gain_db,
                    enable_metronome=enable_metronome,
                    harmonic_factors=list(DEFAULT_HARMONIC_FACTORS),
                    min_stretch=min_stretch,
                    max_stretch=max_stretch,
                    workers=workers,
                    random_sample=random_sample,
                    seed=seed,
                    force=force,
                    rebuild=rebuild,
                    log_fn=self.signals.log_emitted.emit,
                    progress_fn=self.signals.progress_changed.emit,
                    cancel_event=self._cancel_event,
                )
                self.signals.finished_ok.emit(manifest)
            except InterruptedError:
                self.signals.cancelled.emit()
            except Exception as exc:
                self.signals.failed.emit(str(exc))

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _on_cancel_build(self) -> None:
        self._cancel_event.set()
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)


def apply_dark_fusion_theme(app: QApplication) -> None:
    """Apply Qt Fusion dark palette and clean crisp control styling."""
    app.setStyle("Fusion")

    palette = QPalette()
    bg_window = QColor(30, 31, 34)
    bg_panel = QColor(39, 41, 46)
    bg_base = QColor(24, 25, 28)
    text_main = QColor(235, 237, 240)
    text_muted = QColor(148, 154, 165)
    accent = QColor(45, 114, 200)

    palette.setColor(QPalette.ColorRole.Window, bg_window)
    palette.setColor(QPalette.ColorRole.WindowText, text_main)
    palette.setColor(QPalette.ColorRole.Base, bg_base)
    palette.setColor(QPalette.ColorRole.AlternateBase, bg_panel)
    palette.setColor(QPalette.ColorRole.ToolTipBase, bg_panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, text_main)
    palette.setColor(QPalette.ColorRole.Text, text_main)
    palette.setColor(QPalette.ColorRole.Button, QColor(54, 58, 66))
    palette.setColor(QPalette.ColorRole.ButtonText, text_main)
    palette.setColor(QPalette.ColorRole.Light, QColor(110, 116, 128))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(84, 89, 99))
    palette.setColor(QPalette.ColorRole.Mid, QColor(72, 76, 86))
    palette.setColor(QPalette.ColorRole.Dark, QColor(20, 21, 24))
    palette.setColor(QPalette.ColorRole.Shadow, QColor(12, 13, 15))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, text_muted)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, text_muted)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, text_muted)
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QGroupBox, QFrame#cardFrame {
            background-color: #27292e;
            border: 1px solid #3a3d45;
            border-radius: 6px;
            margin-top: 8px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 6px;
            color: #d8dce3;
        }
        QFrame#subCardFrame {
            background-color: #212327;
            border: 1px solid #34373e;
            border-radius: 5px;
        }
        QLabel#mutedLabel {
            color: #9ba3b0;
            font-weight: normal;
        }
        QLineEdit, QListWidget, QPlainTextEdit {
            background-color: #18191c;
            border: 1px solid #3c4049;
            border-radius: 4px;
            padding: 4px 6px;
        }
        QPushButton {
            background-color: #363941;
            border: 1px solid #484c57;
            border-radius: 5px;
            padding: 5px 12px;
            font-weight: 600;
        }
        QPushButton:hover:!disabled {
            background-color: #424650;
        }
        QPushButton#primaryButton {
            background-color: #2667b8;
            border-color: #3579ce;
            color: #ffffff;
        }
        QPushButton#primaryButton:hover:!disabled {
            background-color: #2f77d1;
        }
        QPushButton#dangerButton {
            background-color: #8f2d2d;
            border-color: #a83a3a;
            color: #ffffff;
        }
        QPushButton#dangerButton:hover:!disabled {
            background-color: #a83737;
        }
        QPushButton:disabled,
        QPushButton#primaryButton:disabled,
        QPushButton#dangerButton:disabled {
            background-color: #2a2c31;
            border-color: #35373e;
            color: #777d88;
        }
        QProgressBar {
            background-color: #18191c;
            border: 1px solid #3c4049;
            border-radius: 4px;
            text-align: center;
            min-height: 16px;
        }
        QProgressBar::chunk {
            background-color: #2667b8;
            border-radius: 3px;
        }
        """
    )


def main() -> int:
    """Entrypoint for the PySide6 desktop application."""
    multiprocessing.freeze_support()
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    apply_dark_fusion_theme(app)
    window = RtmixMainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
