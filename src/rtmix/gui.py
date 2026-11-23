"""CustomTkinter desktop GUI for Running Tracks Mixer (`rtmix`)."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from rtmix.analyzer import DEFAULT_HARMONIC_FACTORS
from rtmix.pipeline import (
    SUPPORTED_EXTENSIONS,
    default_workers,
    discover_audio_files,
    run_pipeline,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _HAS_TKDND = True
except Exception:
    DND_FILES = "DND_Files"
    TkinterDnD = None  # type: ignore[assignment]
    _HAS_TKDND = False


def parse_dropped_paths(raw_data: str, tk_interp: tk.Misc) -> list[Path]:
    """Parse Tcl list string from tkdnd <<Drop>> event into existing Paths."""
    if not raw_data:
        return []
    try:
        tokens = list(tk_interp.tk.splitlist(raw_data))
    except Exception:
        tokens = raw_data.strip().split()
    out: list[Path] = []
    for tok in tokens:
        cleaned = tok.strip()
        if cleaned.startswith("file://"):
            cleaned = cleaned[7:]
        p = Path(cleaned)
        if p.exists():
            out.append(p)
    return out


def open_folder_in_explorer(folder: Path) -> None:
    """Open `folder` in the OS native file explorer (Windows/macOS/Linux)."""
    target = folder.resolve()
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


class RtmixApp(ctk.CTk):
    """Main CustomTkinter application window with optional drag-and-drop support."""

    def __init__(self) -> None:
        super().__init__()
        self._dnd_enabled = False
        if _HAS_TKDND and TkinterDnD is not None:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
                self._dnd_enabled = True
            except Exception:
                self._dnd_enabled = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Running Tracks Mixer (rtmix)")
        self.geometry("920x820")
        self.minsize(780, 680)

        self.queued_paths: list[Path] = []
        self._ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._advanced_visible = False

        self._build_ui()
        self._enable_drag_and_drop()
        self.after(80, self._poll_ui_queue)

    def _enable_drag_and_drop(self) -> None:
        if not self._dnd_enabled:
            return
        for widget in (self, self.queue_listbox):
            try:
                widget.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                widget.dnd_bind("<<Drop>>", self._on_drop_files)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # 1. Input Queue Section
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        header_row.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        header_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header_row,
            text="1. Audio Tracks / Folders",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        dnd_note = (
            "Drag & drop files/folders from Explorer, or use buttons"
            if self._dnd_enabled
            else "Add audio files or folders below"
        )
        self.queue_count_label = ctk.CTkLabel(
            header_row,
            text=f"0 tracks queued ({dnd_note})",
            text_color=("gray30", "gray70"),
        )
        self.queue_count_label.grid(row=0, column=1, padx=12, sticky="e")

        btn_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_row.grid(row=1, column=0, padx=10, pady=4, sticky="ew")

        ctk.CTkButton(btn_row, text="+ Add Files", width=120, command=self._on_add_files).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(btn_row, text="+ Add Folder", width=120, command=self._on_add_folder).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(
            btn_row,
            text="Remove Selected",
            width=130,
            fg_color="gray35",
            hover_color="gray25",
            command=self._on_remove_selected,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="Clear Queue",
            width=110,
            fg_color="gray35",
            hover_color="gray25",
            command=self._on_clear_queue,
        ).pack(side="left")

        list_container = ctk.CTkFrame(input_frame)
        list_container.grid(row=2, column=0, padx=10, pady=(4, 10), sticky="ew")
        list_container.grid_columnconfigure(0, weight=1)

        self.queue_listbox = tk.Listbox(
            list_container,
            height=5,
            selectmode=tk.EXTENDED,
            bg="#1d1e1e",
            fg="#e5e5e5",
            selectbackground="#1f538d",
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            font=("Monospace", 10),
        )
        self.queue_listbox.grid(row=0, column=0, sticky="ew", padx=(4, 0), pady=4)
        scrollbar = ctk.CTkScrollbar(list_container, command=self.queue_listbox.yview, height=95)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=4, pady=4)
        self.queue_listbox.configure(yscrollcommand=scrollbar.set)

        # 2. Core Controls Section
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=1, column=0, padx=14, pady=6, sticky="ew")
        controls_frame.grid_columnconfigure(1, weight=1)

        # Output file row
        ctk.CTkLabel(
            controls_frame,
            text="Output MP3:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=(12, 8), pady=(10, 6), sticky="w")

        out_row = ctk.CTkFrame(controls_frame, fg_color="transparent")
        out_row.grid(row=0, column=1, columnspan=2, padx=(0, 12), pady=(10, 6), sticky="ew")
        out_row.grid_columnconfigure(0, weight=1)

        self.output_var = tk.StringVar(value="output/mashup.mp3")
        self.output_entry = ctk.CTkEntry(out_row, textvariable=self.output_var)
        self.output_entry.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        ctk.CTkButton(out_row, text="Browse...", width=95, command=self._on_browse_output).grid(
            row=0, column=1, padx=(0, 8)
        )
        ctk.CTkButton(
            out_row,
            text="Open Folder",
            width=105,
            fg_color="gray35",
            hover_color="gray25",
            command=self._on_open_output_folder,
        ).grid(row=0, column=2)

        # Target Tempo (BPM) slider
        ctk.CTkLabel(
            controls_frame,
            text="Target Tempo:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=1, column=0, padx=(12, 8), pady=6, sticky="w")

        self.bpm_slider = ctk.CTkSlider(
            controls_frame,
            from_=120.0,
            to=220.0,
            number_of_steps=100,
            command=self._on_bpm_slider_change,
        )
        self.bpm_slider.set(180.0)
        self.bpm_slider.grid(row=1, column=1, padx=(0, 10), pady=6, sticky="ew")

        self.bpm_value_label = ctk.CTkLabel(controls_frame, text="180.0 BPM", width=95, anchor="e")
        self.bpm_value_label.grid(row=1, column=2, padx=(0, 12), pady=6, sticky="e")

        # CPU Cores slider
        max_cpus = max(1, os.cpu_count() or 4)
        def_cpus = default_workers()
        ctk.CTkLabel(
            controls_frame,
            text="CPU Workers:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=2, column=0, padx=(12, 8), pady=6, sticky="w")

        self.cores_slider = ctk.CTkSlider(
            controls_frame,
            from_=1,
            to=max_cpus,
            number_of_steps=max(1, max_cpus - 1),
            command=self._on_cores_slider_change,
        )
        self.cores_slider.set(def_cpus)
        self.cores_slider.grid(row=2, column=1, padx=(0, 10), pady=6, sticky="ew")

        self.cores_value_label = ctk.CTkLabel(
            controls_frame, text=f"{def_cpus} cores", width=95, anchor="e"
        )
        self.cores_value_label.grid(row=2, column=2, padx=(0, 12), pady=6, sticky="e")

        # Metronome row (Toggle + Gain slider)
        self.metronome_var = tk.BooleanVar(value=True)
        self.metronome_cb = ctk.CTkCheckBox(
            controls_frame,
            text="Metronome Click",
            variable=self.metronome_var,
            font=ctk.CTkFont(weight="bold"),
            command=self._on_metronome_toggle,
        )
        self.metronome_cb.grid(row=3, column=0, padx=(12, 8), pady=6, sticky="w")

        self.metro_gain_slider = ctk.CTkSlider(
            controls_frame,
            from_=-18.0,
            to=0.0,
            number_of_steps=36,
            command=self._on_metro_gain_change,
        )
        self.metro_gain_slider.set(-6.0)
        self.metro_gain_slider.grid(row=3, column=1, padx=(0, 10), pady=6, sticky="ew")

        self.metro_gain_label = ctk.CTkLabel(controls_frame, text="-6.0 dB", width=95, anchor="e")
        self.metro_gain_label.grid(row=3, column=2, padx=(0, 12), pady=6, sticky="e")

        # Random pick row
        self.random_enabled_var = tk.BooleanVar(value=False)
        self.random_cb = ctk.CTkCheckBox(
            controls_frame,
            text="Pick Random N:",
            variable=self.random_enabled_var,
            font=ctk.CTkFont(weight="bold"),
            command=self._on_random_toggle,
        )
        self.random_cb.grid(row=4, column=0, padx=(12, 8), pady=(6, 10), sticky="w")

        rand_row = ctk.CTkFrame(controls_frame, fg_color="transparent")
        rand_row.grid(row=4, column=1, columnspan=2, padx=(0, 12), pady=(6, 10), sticky="w")

        self.random_count_var = tk.StringVar(value="10")
        self.random_count_entry = ctk.CTkEntry(
            rand_row, textvariable=self.random_count_var, width=70, state="disabled"
        )
        self.random_count_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(rand_row, text="tracks   |   Optional Seed:").pack(side="left", padx=(0, 8))
        self.seed_var = tk.StringVar(value="")
        self.seed_entry = ctk.CTkEntry(
            rand_row, textvariable=self.seed_var, width=80, placeholder_text="auto", state="disabled"
        )
        self.seed_entry.pack(side="left")

        # 3. Collapsible Advanced Settings Section
        adv_wrapper = ctk.CTkFrame(self)
        adv_wrapper.grid(row=2, column=0, padx=14, pady=4, sticky="ew")
        adv_wrapper.grid_columnconfigure(0, weight=1)

        self.adv_toggle_btn = ctk.CTkButton(
            adv_wrapper,
            text="▸ Advanced Settings (Min/Max Stretch, Force Dead-Zone, Crossfade, Cache)",
            anchor="w",
            fg_color="transparent",
            hover_color=("gray80", "gray25"),
            text_color=("gray10", "gray85"),
            font=ctk.CTkFont(weight="bold"),
            command=self._toggle_advanced,
        )
        self.adv_toggle_btn.grid(row=0, column=0, padx=6, pady=4, sticky="ew")

        self.adv_frame = ctk.CTkFrame(adv_wrapper, fg_color=("gray90", "gray17"))
        self.adv_frame.grid_columnconfigure((1, 3, 5), weight=1)

        # Row 0 in Advanced: Min Stretch, Max Stretch, Force checkbox
        ctk.CTkLabel(self.adv_frame, text="Min Stretch:").grid(
            row=0, column=0, padx=(12, 6), pady=(10, 6), sticky="w"
        )
        self.min_stretch_var = tk.StringVar(value="0.85")
        ctk.CTkEntry(self.adv_frame, textvariable=self.min_stretch_var, width=75).grid(
            row=0, column=1, padx=(0, 16), pady=(10, 6), sticky="w"
        )

        ctk.CTkLabel(self.adv_frame, text="Max Stretch:").grid(
            row=0, column=2, padx=(6, 6), pady=(10, 6), sticky="w"
        )
        self.max_stretch_var = tk.StringVar(value="1.35")
        ctk.CTkEntry(self.adv_frame, textvariable=self.max_stretch_var, width=75).grid(
            row=0, column=3, padx=(0, 16), pady=(10, 6), sticky="w"
        )

        self.force_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.adv_frame,
            text="Force Dead-Zone Tracks (-force)",
            variable=self.force_var,
        ).grid(row=0, column=4, columnspan=2, padx=(6, 12), pady=(10, 6), sticky="w")

        # Row 1 in Advanced: Crossfade Beats, Lead-in Beats, Lead-out Beats, Rebuild Cache
        ctk.CTkLabel(self.adv_frame, text="Crossfade Beats:").grid(
            row=1, column=0, padx=(12, 6), pady=(6, 10), sticky="w"
        )
        self.crossfade_var = tk.StringVar(value="8")
        ctk.CTkEntry(self.adv_frame, textvariable=self.crossfade_var, width=75).grid(
            row=1, column=1, padx=(0, 16), pady=(6, 10), sticky="w"
        )

        ctk.CTkLabel(self.adv_frame, text="Lead-in / Out Beats:").grid(
            row=1, column=2, padx=(6, 6), pady=(6, 10), sticky="w"
        )
        self.lead_beats_var = tk.StringVar(value="8")
        ctk.CTkEntry(self.adv_frame, textvariable=self.lead_beats_var, width=75).grid(
            row=1, column=3, padx=(0, 16), pady=(6, 10), sticky="w"
        )

        self.rebuild_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            self.adv_frame,
            text="Rebuild Cache (--rebuild)",
            variable=self.rebuild_var,
        ).grid(row=1, column=4, columnspan=2, padx=(6, 12), pady=(6, 10), sticky="w")

        # 4. Action & Progress Bar Section
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=3, column=0, padx=14, pady=6, sticky="ew")
        action_frame.grid_columnconfigure(2, weight=1)

        self.build_btn = ctk.CTkButton(
            action_frame,
            text="Build Mashup",
            width=160,
            height=36,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_start_build,
        )
        self.build_btn.grid(row=0, column=0, padx=(10, 8), pady=10)

        self.cancel_btn = ctk.CTkButton(
            action_frame,
            text="Cancel",
            width=100,
            height=36,
            fg_color="#8b2626",
            hover_color="#6b1d1d",
            state="disabled",
            command=self._on_cancel_build,
        )
        self.cancel_btn.grid(row=0, column=1, padx=(0, 12), pady=10)

        prog_col = ctk.CTkFrame(action_frame, fg_color="transparent")
        prog_col.grid(row=0, column=2, padx=(0, 12), pady=8, sticky="ew")
        prog_col.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(prog_col, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(prog_col)
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=1, column=0, pady=(4, 0), sticky="ew")

        # 5. Live Log Console Section
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=4, column=0, padx=14, pady=(4, 14), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="Pipeline Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(6, 2), sticky="w")

        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Monospace", size=12))
        self.log_box.grid(row=1, column=0, padx=10, pady=(2, 10), sticky="nsew")
        self.log_box.configure(state="disabled")

    # --- UI callbacks ---

    def _toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self.adv_toggle_btn.configure(
                text="▾ Advanced Settings (Min/Max Stretch, Force Dead-Zone, Crossfade, Cache)"
            )
            self.adv_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        else:
            self.adv_toggle_btn.configure(
                text="▸ Advanced Settings (Min/Max Stretch, Force Dead-Zone, Crossfade, Cache)"
            )
            self.adv_frame.grid_forget()

    def _on_bpm_slider_change(self, val: float) -> None:
        rounded = round(float(val))
        self.bpm_value_label.configure(text=f"{rounded:.1f} BPM")

    def _on_cores_slider_change(self, val: float) -> None:
        cores = max(1, round(float(val)))
        self.cores_value_label.configure(text=f"{cores} core{'s' if cores != 1 else ''}")

    def _on_metronome_toggle(self) -> None:
        enabled = self.metronome_var.get()
        self.metro_gain_slider.configure(state="normal" if enabled else "disabled")
        self.metro_gain_label.configure(
            text=f"{round(self.metro_gain_slider.get() * 2) / 2:.1f} dB" if enabled else "Muted"
        )

    def _on_metro_gain_change(self, val: float) -> None:
        stepped = round(float(val) * 2.0) / 2.0
        self.metro_gain_label.configure(text=f"{stepped:.1f} dB")

    def _on_random_toggle(self) -> None:
        state = "normal" if self.random_enabled_var.get() else "disabled"
        self.random_count_entry.configure(state=state)
        self.seed_entry.configure(state=state)

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
        self.queue_listbox.delete(0, tk.END)
        for p in self.queued_paths:
            tag = "[DIR] " if p.is_dir() else "[FILE]"
            self.queue_listbox.insert(tk.END, f"{tag} {p}")
        try:
            discovered = discover_audio_files(self.queued_paths) if self.queued_paths else []
            count = len(discovered)
        except Exception:
            count = 0
        self.queue_count_label.configure(
            text=f"{len(self.queued_paths)} item(s) queued — {count} audio track(s) discovered"
        )

    def _on_add_files(self) -> None:
        exts = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        selected = filedialog.askopenfilenames(
            title="Select Audio Tracks",
            filetypes=[("Audio Files", exts), ("All Files", "*.*")],
        )
        if selected:
            self._add_paths_to_queue([Path(s) for s in selected])

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder of Audio Tracks")
        if folder:
            self._add_paths_to_queue([Path(folder)])

    def _on_drop_files(self, event: Any) -> None:
        paths = parse_dropped_paths( getattr(event, "data", ""), self)
        if paths:
            self._add_paths_to_queue(paths)

    def _on_remove_selected(self) -> None:
        sel = list(self.queue_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            if 0 <= idx < len(self.queued_paths):
                self.queued_paths.pop(idx)
        self._refresh_queue_display()

    def _on_clear_queue(self) -> None:
        self.queued_paths.clear()
        self._refresh_queue_display()

    def _on_browse_output(self) -> None:
        initial = Path(self.output_var.get() or "output/mashup.mp3")
        chosen = filedialog.asksaveasfilename(
            title="Save Output Mashup MP3",
            initialdir=str(initial.parent if initial.parent.exists() else Path.cwd()),
            initialfile=initial.name,
            defaultextension=".mp3",
            filetypes=[("MP3 Audio", "*.mp3")],
        )
        if chosen:
            self.output_var.set(chosen)

    def _on_open_output_folder(self) -> None:
        out_path = Path(self.output_var.get().strip() or "output/mashup.mp3")
        try:
            open_folder_in_explorer(out_path.parent)
        except Exception as exc:
            messagebox.showerror("Open Folder Error", str(exc))

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    done, total = payload
                    frac = float(done) / float(total) if total > 0 else 0.0
                    self.progress_bar.set(min(1.0, max(0.0, frac)))
                    self.status_label.configure(text=f"Processing tracks: {done} / {total}")
                elif kind == "done":
                    manifest = payload
                    self.progress_bar.set(1.0)
                    mins = manifest.total_duration_sec / 60.0
                    self.status_label.configure(
                        text=f"Done! {manifest.output_mp3_path} ({mins:.2f} min)"
                    )
                    self.build_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                elif kind == "cancelled":
                    self.status_label.configure(text="Cancelled by user.")
                    self._append_log("[CANCELLED] Pipeline stopped by user.")
                    self.build_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                elif kind == "error":
                    self.status_label.configure(text=f"Error: {payload}")
                    self._append_log(f"[ERROR] {payload}")
                    self.build_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    messagebox.showerror("Build Failed", str(payload))
        except queue.Empty:
            pass
        self.after(80, self._poll_ui_queue)

    def _on_start_build(self) -> None:
        if not self.queued_paths:
            messagebox.showwarning(
                "No Tracks Queued",
                "Please add at least one audio file or folder (via buttons or drag & drop).",
            )
            return

        try:
            min_stretch = float(self.min_stretch_var.get().strip())
            max_stretch = float(self.max_stretch_var.get().strip())
            if min_stretch <= 0 or max_stretch <= min_stretch:
                raise ValueError("Min stretch must be > 0 and less than Max stretch.")
            crossfade_beats = int(self.crossfade_var.get().strip())
            lead_beats = int(self.lead_beats_var.get().strip())
            if crossfade_beats < 0 or lead_beats < 0:
                raise ValueError("Beat counts must be >= 0.")

            random_sample: int | None = None
            seed: int | None = None
            if self.random_enabled_var.get():
                random_sample = int(self.random_count_var.get().strip())
                if random_sample <= 0:
                    raise ValueError("Random N must be >= 1.")
                raw_seed = self.seed_var.get().strip()
                if raw_seed:
                    seed = int(raw_seed)
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return

        target_bpm = float(round(self.bpm_slider.get()))
        workers = max(1, int(round(self.cores_slider.get())))
        enable_metronome = bool(self.metronome_var.get())
        metronome_gain_db = round(float(self.metro_gain_slider.get()) * 2.0) / 2.0
        output_mp3 = Path(self.output_var.get().strip() or "output/mashup.mp3")
        force = bool(self.force_var.get())
        rebuild = bool(self.rebuild_var.get())
        inputs = list(self.queued_paths)

        self._cancel_event.clear()
        self.build_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_bar.set(0.0)
        self.status_label.configure(text="Starting pipeline...")

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self._append_log(
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
                    log_fn=lambda msg: self._ui_queue.put(("log", msg)),
                    progress_fn=lambda d, t: self._ui_queue.put(("progress", (d, t))),
                    cancel_event=self._cancel_event,
                )
                self._ui_queue.put(("done", manifest))
            except InterruptedError:
                self._ui_queue.put(("cancelled", None))
            except Exception as exc:
                self._ui_queue.put(("error", str(exc)))

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def _on_cancel_build(self) -> None:
        self._cancel_event.set()
        self.status_label.configure(text="Cancelling...")
        self.cancel_btn.configure(state="disabled")


def main() -> int:
    multiprocessing.freeze_support()
    app = RtmixApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
