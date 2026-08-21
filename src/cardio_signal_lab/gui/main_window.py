"""Main application window with three-level view hierarchy and dynamic menus."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QDialog,
    QFormLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
    QLabel,
    QComboBox,
    QSpinBox,
    QProgressDialog,
    QStackedWidget,
)
from PySide6.QtCore import Qt
from loguru import logger

from cardio_signal_lab import __version__
from cardio_signal_lab.config import get_config, get_config_manager, get_help_text, get_keysequence
from cardio_signal_lab.core import (
    BadSegment,
    get_loader,
    SignalType,
    CsvLoader,
    PeakData,
    PeakClassification,
    save_session,
    load_session,
    export_csv,
    export_npy,
    export_annotations,
    export_intervals,
    save_processing_parameters,
    load_events_csv,
    load_peaks_binary_csv,
)
from cardio_signal_lab.core.session import verify_source_checksum
from cardio_signal_lab.gui.multi_signal_view import MultiSignalView
from cardio_signal_lab.gui.signal_type_view import SignalTypeView
from cardio_signal_lab.gui.single_channel_view import SingleChannelView
from cardio_signal_lab.gui.status_bar import AppStatusBar
from cardio_signal_lab.gui.event_editor_dialog import EventEditorDialog
from cardio_signal_lab.gui.log_panel import LogPanel
from cardio_signal_lab.gui.processing_panel import ProcessingPanel
from cardio_signal_lab.processing import ProcessingPipeline, ProcessingWorker
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import RecordingSession, SignalData


class MainWindow(QMainWindow):
    """Main application window with three-level view hierarchy.

    View levels:
    - multi: All signal types overview (one plot per type)
    - type: All channels of one signal type (stacked plots)
    - channel: Single channel for processing/correction

    Menu system adapts per view level:
    - multi: File, Edit(disabled), Select(types), View, Help
    - type: File, Edit(disabled), Select(channels), View, Help
    - channel: File, Edit, Process, View, Help
    """

    def __init__(self):
        super().__init__()

        self.config = get_config()
        self.signals = get_app_signals()

        # View state
        self.current_view_level = "multi"  # "multi" | "type" | "channel"
        self.current_signal_type: SignalType | None = None
        self.current_session: RecordingSession | None = None
        self.current_signal: SignalData | None = None
        # Track whether we came from type view (for ESC navigation)
        self._came_from_type_view = False
        self._last_open_dir: Path = Path.home()  # Remembered for open/import dialogs
        self._last_save_dir: Path = Path.home()  # Remembered for save/export dialogs

        # Processing state (active channel)
        self.pipeline = ProcessingPipeline()
        self._raw_samples: np.ndarray | None = None  # Baseline signal for pipeline replay
        self._current_peaks: PeakData | None = None
        self._current_bad_segments: list[BadSegment] = []
        self._current_gap_segments: list[BadSegment] = []  # Timestamp gaps, displayed as colored trace
        self._show_gap_segments: bool = False
        self._interpolated_bad_segments: list[BadSegment] = []  # Saved after interpolation for overlay toggle
        self._show_interpolated_regions: bool = False
        self._processing_worker: ProcessingWorker | None = None
        # Structural ops (crop / resample) that have been applied directly to the
        # raw signal and cannot be replayed or reverted by the pipeline.
        self._structural_ops: list = []
        # Write-once snapshot of the file-loaded signal, captured on first channel visit.
        # Never overwritten by crop/resample, enabling full Reset to original state.
        self._original_samples: np.ndarray | None = None
        self._original_timestamps: np.ndarray | None = None
        self._original_sampling_rate: float | None = None

        # Derived visualization state (active channel)
        self._eda_tonic: np.ndarray | None = None   # Tonic (SCL) component from eda_process()
        self._eda_phasic: np.ndarray | None = None  # Phasic (SCR) component from eda_process()

        # Per-channel state store: (SignalType, channel_name) -> state dict
        # Allows peaks and processing to survive navigation between channels.
        self._channel_state: dict[tuple, dict] = {}

        # Derived channel specs for session persistence.
        # Each entry: {"type": "l2_norm", "signal_type": str, "source_channels": [str, ...]}
        self._derived_channel_specs: list[dict] = []

        # Session-level fields (not per-channel)
        self._session_notes: str = ""
        self._current_session_path: Path | None = None  # Path of last save/load .csl.json
        self._is_dirty: bool = False  # Unsaved changes since last save/load

        # Window setup
        self.setWindowTitle("CardioSignalLab")
        self.resize(self.config.gui.window_width, self.config.gui.window_height)

        # Create stacked widget (container persists; views inside are rebuilt on file close)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create the three views and wire their signals
        self._create_views()

        # Status bar
        self.status_bar = AppStatusBar(self)
        self.setStatusBar(self.status_bar)

        # Log panel (dockable)
        self.log_panel = LogPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_panel)
        self.log_panel.hide()  # Hidden by default

        # Register log panel with loguru
        logger.add(
            self.log_panel.get_loguru_sink(),
            format="<lvl>{level}</lvl>|{message}",
            level="INFO",
            colorize=False,
        )

        # Processing panel (dockable, right side)
        self.processing_panel = ProcessingPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.processing_panel)
        self.processing_panel.hide()  # Hidden by default

        # Build initial menus
        self._build_menus()

        # Connect app signals (these survive view rebuilds)
        self.signals.mode_changed.connect(self._on_mode_changed)
        self.signals.file_loaded.connect(self._on_file_loaded_signal)

        logger.info("MainWindow initialized with 3-level view hierarchy")

    # ---- Menu Building ----

    def _create_views(self):
        """Create the three view widgets and add them to the stacked widget."""
        self.multi_signal_view = MultiSignalView()
        self.signal_type_view = SignalTypeView()
        self.single_channel_view = SingleChannelView()

        self.stacked_widget.addWidget(self.multi_signal_view)   # Index 0
        self.stacked_widget.addWidget(self.signal_type_view)    # Index 1
        self.stacked_widget.addWidget(self.single_channel_view) # Index 2
        self.stacked_widget.setCurrentWidget(self.multi_signal_view)

        # Connect view signals
        self.multi_signal_view.signal_type_selected.connect(self._on_signal_type_selected)
        self.signal_type_view.channel_selected.connect(self._on_channel_selected)
        self.single_channel_view.return_to_multi_requested.connect(self._on_return_to_multi)
        self.single_channel_view.peaks_changed.connect(self._on_peaks_changed)

        logger.debug("Views created")

    def _destroy_views(self):
        """Remove and destroy all three view widgets from the stacked widget."""
        for view in (self.single_channel_view, self.signal_type_view, self.multi_signal_view):
            self.stacked_widget.removeWidget(view)
            view.deleteLater()
        logger.debug("Views destroyed")

    def _rebuild_views(self):
        """Destroy and recreate all views. Used on file close to guarantee clean state."""
        self._destroy_views()
        self._create_views()

    def _build_menus(self):
        """Build menu bar based on current view level."""
        self.menuBar().clear()

        if self.current_view_level == "multi":
            self._build_multi_menus()
        elif self.current_view_level == "type":
            self._build_type_menus()
        else:  # channel
            self._build_channel_menus()

        logger.debug(f"Menus rebuilt for {self.current_view_level} view level")

    def _build_multi_menus(self):
        """Menus for multi-signal view: File, Edit(disabled), Select(types), View, Help."""
        file_menu = self.menuBar().addMenu("&File")
        self._add_file_menu_actions(file_menu)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.setEnabled(False)

        select_menu = self.menuBar().addMenu("&Select")
        self._add_select_menu_actions(select_menu)

        view_menu = self.menuBar().addMenu("&View")
        self._add_view_menu_actions(view_menu)

        help_menu = self.menuBar().addMenu("&Help")
        self._add_help_menu_actions(help_menu)

    def _build_type_menus(self):
        """Menus for signal-type view: File, Edit(disabled), Process(derive), Select, View, Help."""
        file_menu = self.menuBar().addMenu("&File")
        self._add_file_menu_actions(file_menu)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.setEnabled(False)

        # Process menu (always shown in type view)
        process_menu = self.menuBar().addMenu("&Process")

        crop_all_action = QAction("Crop &All Channels...", self)
        crop_all_action.triggered.connect(self._on_type_crop_all)
        process_menu.addAction(crop_all_action)

        if len(self.signal_type_view.signals) >= 2:
            process_menu.addSeparator()
            l2_action = QAction("Create &L2 Norm Channel...", self)
            l2_action.triggered.connect(self._on_type_create_l2_norm)
            process_menu.addAction(l2_action)

        select_menu = self.menuBar().addMenu("&Select")
        self._add_select_menu_actions(select_menu)

        view_menu = self.menuBar().addMenu("&View")
        self._add_view_menu_actions(view_menu)

        help_menu = self.menuBar().addMenu("&Help")
        self._add_help_menu_actions(help_menu)

    def _build_channel_menus(self):
        """Menus for single-channel view: File, Edit, Process, View, Help."""
        file_menu = self.menuBar().addMenu("&File")
        self._add_file_menu_actions(file_menu)

        edit_menu = self.menuBar().addMenu("&Edit")
        self._add_edit_menu_actions(edit_menu)

        process_menu = self.menuBar().addMenu("&Process")
        self._add_process_menu_actions(process_menu)

        view_menu = self.menuBar().addMenu("&View")
        self._add_view_menu_actions(view_menu)

        help_menu = self.menuBar().addMenu("&Help")
        self._add_help_menu_actions(help_menu)

    def _add_file_menu_actions(self, menu):
        """Add File menu actions."""
        open_action = QAction("&Open...", self)
        open_action.setShortcut(get_keysequence("file_open"))
        open_action.triggered.connect(self._on_file_open)
        menu.addAction(open_action)

        close_action = QAction("&Close File", self)
        close_action.triggered.connect(self._on_file_close)
        close_action.setEnabled(self.current_session is not None)
        menu.addAction(close_action)

        append_action = QAction("&Append File...", self)
        append_action.triggered.connect(self._on_file_append)
        append_action.setEnabled(self.current_session is not None)
        menu.addAction(append_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut(get_keysequence("file_save"))
        save_action.triggered.connect(self._on_file_save)
        menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.triggered.connect(self._on_file_save_as)
        save_as_action.setEnabled(self.current_session is not None)
        menu.addAction(save_as_action)

        # Open Recent submenu
        recent_menu = menu.addMenu("Open &Recent")
        cfg = get_config()
        recent_sessions = cfg.gui.recent_session_files
        recent_sources = cfg.gui.recent_source_files
        if recent_sessions or recent_sources:
            if recent_sessions:
                for p in recent_sessions:
                    action = QAction(Path(p).name, self)
                    action.setToolTip(p)
                    action.triggered.connect(
                        lambda checked, fp=p: self._open_recent(fp)
                    )
                    recent_menu.addAction(action)
            if recent_sessions and recent_sources:
                recent_menu.addSeparator()
            if recent_sources:
                for p in recent_sources:
                    action = QAction(Path(p).name, self)
                    action.setToolTip(p)
                    action.triggered.connect(
                        lambda checked, fp=p: self._open_recent(fp)
                    )
                    recent_menu.addAction(action)
        else:
            empty = QAction("(No recent files)", self)
            empty.setEnabled(False)
            recent_menu.addAction(empty)

        export_action = QAction("&Export...", self)
        export_action.setShortcut(get_keysequence("file_export"))
        export_action.triggered.connect(self._on_file_export)
        menu.addAction(export_action)

        menu.addSeparator()

        import_events_action = QAction("Import &Events (CSV)...", self)
        import_events_action.triggered.connect(self._on_file_import_events)
        import_events_action.setEnabled(self.current_session is not None)
        menu.addAction(import_events_action)

        edit_events_action = QAction("&Edit Events...", self)
        edit_events_action.triggered.connect(self._on_file_edit_events)
        edit_events_action.setEnabled(self.current_session is not None)
        menu.addAction(edit_events_action)

        import_peaks_action = QAction("Import &Peaks (CSV)...", self)
        import_peaks_action.triggered.connect(self._on_file_import_peaks)
        import_peaks_action.setEnabled(
            self.current_view_level == "channel" and self.current_signal is not None
        )
        menu.addAction(import_peaks_action)

        menu.addSeparator()

        notes_action = QAction("Session &Notes...", self)
        notes_action.triggered.connect(self._on_session_notes)
        notes_action.setEnabled(self.current_session is not None)
        menu.addAction(notes_action)

        menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(get_keysequence("file_quit"))
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)

    def _add_edit_menu_actions(self, menu):
        """Add Edit menu actions (channel view only)."""
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(get_keysequence("edit_undo"))
        undo_action.triggered.connect(self._on_edit_undo)
        menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(get_keysequence("edit_redo"))
        redo_action.triggered.connect(self._on_edit_redo)
        menu.addAction(redo_action)

        menu.addSeparator()

        reviewed_action = QAction("Mark Channel as &Reviewed", self)
        reviewed_action.setCheckable(True)
        if self.current_signal is not None:
            key = self._channel_key(self.current_signal)
            reviewed_action.setChecked(self._channel_state.get(key, {}).get("reviewed", False))
        reviewed_action.triggered.connect(self._on_toggle_reviewed)
        menu.addAction(reviewed_action)

    def _add_select_menu_actions(self, menu):
        """Add Select menu actions - context-dependent on view level."""
        if self.current_view_level == "multi":
            # List signal types
            if self.current_session:
                types_seen = []
                for sig in self.current_session.signals:
                    if sig.signal_type not in types_seen:
                        types_seen.append(sig.signal_type)

                for signal_type in types_seen:
                    channels = self.multi_signal_view.get_signals_for_type(signal_type)
                    label = f"{signal_type.value.upper()} ({len(channels)} channel{'s' if len(channels) != 1 else ''})"
                    action = QAction(label, self)
                    action.triggered.connect(
                        lambda checked, st=signal_type: self._on_signal_type_selected(st)
                    )
                    menu.addAction(action)
            else:
                placeholder = QAction("(No signals loaded)", self)
                placeholder.setEnabled(False)
                menu.addAction(placeholder)

        elif self.current_view_level == "type":
            # List channels of current signal type
            if self.signal_type_view.signals:
                for signal in self.signal_type_view.signals:
                    key = self._channel_key(signal)
                    is_reviewed = self._channel_state.get(key, {}).get("reviewed", False)
                    label = f"[R] {signal.channel_name}" if is_reviewed else signal.channel_name
                    action = QAction(label, self)
                    action.triggered.connect(
                        lambda checked, sig=signal: self._on_channel_selected(sig)
                    )
                    menu.addAction(action)

                # Add derived channels
                for derived in self.signal_type_view.derived_signals:
                    action = QAction(f"{derived.channel_name} (derived)", self)
                    menu.addAction(action)
            else:
                placeholder = QAction("(No channels)", self)
                placeholder.setEnabled(False)
                menu.addAction(placeholder)

    def _add_process_menu_actions(self, menu):
        """Add Process menu actions, ordered from first step to last.

        Order: Standard macro | Crop/Resample | Bad segments |
               Filters | NK clean + detect | Reset
        """
        if self.current_signal is None:
            return

        sig_type = self.current_signal.signal_type

        # --- Standard Processing macro (ECG/PPG only) ---
        if sig_type in (SignalType.ECG, SignalType.PPG):
            standard_action = QAction("Standard &Processing...", self)
            standard_action.triggered.connect(self._on_process_standard)
            menu.addAction(standard_action)
            menu.addSeparator()

        # --- Structural edits (change signal extent / sample rate) ---
        crop_action = QAction("Cr&op...", self)
        crop_action.triggered.connect(self._on_process_crop)
        menu.addAction(crop_action)

        resample_action = QAction("&Resample...", self)
        resample_action.triggered.connect(self._on_process_resample)
        menu.addAction(resample_action)

        menu.addSeparator()

        # --- Timestamp gap detection ---
        detect_gaps_action = QAction("Detect &Timestamp Gaps...", self)
        detect_gaps_action.triggered.connect(self._on_detect_timestamp_gaps)
        menu.addAction(detect_gaps_action)

        clear_gaps_action = QAction("Clear Timestamp &Gaps", self)
        clear_gaps_action.triggered.connect(self._on_clear_gap_segments)
        clear_gaps_action.setEnabled(bool(self._current_gap_segments))
        menu.addAction(clear_gaps_action)

        menu.addSeparator()

        # --- Amplitude artifact detection and repair ---
        detect_bad_action = QAction("Detect &Bad Segments...", self)
        detect_bad_action.triggered.connect(self._on_detect_bad_segments)
        menu.addAction(detect_bad_action)

        mark_bad_action = QAction("&Mark Bad Segment (Manual)...", self)
        mark_bad_action.triggered.connect(self._on_mark_bad_segment)
        menu.addAction(mark_bad_action)

        interpolate_bad_action = QAction("&Interpolate Bad Segments", self)
        interpolate_bad_action.triggered.connect(self._on_interpolate_bad_segments)
        interpolate_bad_action.setEnabled(bool(self._current_bad_segments))
        menu.addAction(interpolate_bad_action)

        clear_bad_action = QAction("Clea&r Bad Segments", self)
        clear_bad_action.triggered.connect(self._on_clear_bad_segments)
        clear_bad_action.setEnabled(bool(self._current_bad_segments))
        menu.addAction(clear_bad_action)

        menu.addSeparator()

        # --- Signal filters ---
        filter_action = QAction("&Bandpass Filter...", self)
        filter_action.triggered.connect(self._on_process_filter)
        menu.addAction(filter_action)

        # Notch filter: ECG and PPG only (EDA is low-frequency; notch is not useful)
        if sig_type in (SignalType.ECG, SignalType.PPG):
            notch_action = QAction("&Notch Filter...", self)
            notch_action.triggered.connect(self._on_process_notch)
            menu.addAction(notch_action)

        baseline_action = QAction("&Detrend (Polynomial)...", self)
        baseline_action.triggered.connect(self._on_process_baseline)
        menu.addAction(baseline_action)

        zero_ref_action = QAction("&DC Offset Removal...", self)
        zero_ref_action.triggered.connect(self._on_process_zero_reference)
        menu.addAction(zero_ref_action)

        # EEMD artifact removal: PPG only
        if sig_type == SignalType.PPG:
            artifact_action = QAction("&Artifact Removal (EEMD)...", self)
            artifact_action.triggered.connect(self._on_process_artifact_removal)
            menu.addAction(artifact_action)

        menu.addSeparator()

        # --- NeuroKit2: clean -> decompose (EDA) -> detect peaks ---
        nk_menu = menu.addMenu("&NeuroKit2")

        if sig_type == SignalType.ECG:
            clean_action = QAction("&Clean Signal", self)
            clean_action.triggered.connect(self._on_nk_ecg_clean)
            nk_menu.addAction(clean_action)

            detect_action = QAction("Detect &R-Peaks", self)
            detect_action.triggered.connect(self._on_process_detect_peaks)
            nk_menu.addAction(detect_action)

        elif sig_type == SignalType.PPG:
            clean_action = QAction("&Clean Signal", self)
            clean_action.triggered.connect(self._on_nk_ppg_clean)
            nk_menu.addAction(clean_action)

            detect_action = QAction("Detect &Pulse Peaks", self)
            detect_action.triggered.connect(self._on_process_detect_peaks)
            nk_menu.addAction(detect_action)

        elif sig_type == SignalType.EDA:
            clean_action = QAction("&Clean Signal", self)
            clean_action.triggered.connect(self._on_nk_eda_clean)
            nk_menu.addAction(clean_action)

            decompose_action = QAction("&Decompose EDA...", self)
            decompose_action.triggered.connect(self._on_nk_eda_decompose)
            nk_menu.addAction(decompose_action)

            detect_action = QAction("Detect &SCR Peaks", self)
            detect_action.triggered.connect(self._on_process_detect_peaks)
            nk_menu.addAction(detect_action)

        else:
            detect_action = QAction("&Detect Peaks (generic)", self)
            detect_action.triggered.connect(self._on_process_detect_peaks)
            nk_menu.addAction(detect_action)

        menu.addSeparator()

        reset_action = QAction("&Reset Processing", self)
        reset_action.triggered.connect(self._on_process_reset)
        menu.addAction(reset_action)

    def _add_view_menu_actions(self, menu):
        """Add View menu actions."""
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(get_keysequence("view_zoom_in"))
        zoom_in_action.triggered.connect(self._on_view_zoom_in)
        menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(get_keysequence("view_zoom_out"))
        zoom_out_action.triggered.connect(self._on_view_zoom_out)
        menu.addAction(zoom_out_action)

        menu.addSeparator()

        reset_view_action = QAction("&Reset View", self)
        reset_view_action.setShortcut(get_keysequence("view_reset"))
        reset_view_action.triggered.connect(self._on_view_reset)
        menu.addAction(reset_view_action)

        fit_view_action = QAction("&Fit to Window", self)
        fit_view_action.setShortcut(get_keysequence("view_fit"))
        fit_view_action.triggered.connect(self._on_view_fit)
        menu.addAction(fit_view_action)

        menu.addSeparator()

        # Navigation actions
        jump_to_start_action = QAction("Jump to &Start", self)
        jump_to_start_action.setShortcut(get_keysequence("peak_first"))
        jump_to_start_action.triggered.connect(self._on_view_jump_to_start)
        menu.addAction(jump_to_start_action)

        jump_to_end_action = QAction("Jump to &End", self)
        jump_to_end_action.setShortcut(get_keysequence("peak_last"))
        jump_to_end_action.triggered.connect(self._on_view_jump_to_end)
        menu.addAction(jump_to_end_action)

        jump_to_time_action = QAction("Jump to &Time...", self)
        jump_to_time_action.setShortcut(get_keysequence("view_jump_to_time"))
        jump_to_time_action.triggered.connect(self._on_view_jump_to_time)
        menu.addAction(jump_to_time_action)

        zoom_to_range_action = QAction("Zoom to Time &Range...", self)
        zoom_to_range_action.triggered.connect(self._on_view_zoom_to_range)
        menu.addAction(zoom_to_range_action)

        menu.addSeparator()

        # Mouse mode toggles
        pan_mode_action = QAction("Pan Mode (drag to pan)", self)
        pan_mode_action.setShortcut(get_keysequence("view_pan_mode"))
        pan_mode_action.triggered.connect(self._on_view_pan_mode)
        menu.addAction(pan_mode_action)

        zoom_rect_action = QAction("Zoom Mode (drag rectangle to zoom)", self)
        zoom_rect_action.setShortcut(get_keysequence("view_zoom_mode"))
        zoom_rect_action.triggered.connect(self._on_view_zoom_mode)
        menu.addAction(zoom_rect_action)

        menu.addSeparator()

        # Toggle events (no single-letter shortcut — E is reserved for ectopic classification)
        toggle_events_action = QAction("Toggle Event Markers", self)
        toggle_events_action.triggered.connect(self._on_view_toggle_events)
        menu.addAction(toggle_events_action)

        # Toggle interpolated segment overlay
        toggle_interp_action = QAction("Show Interpolated Regions", self)
        toggle_interp_action.setCheckable(True)
        toggle_interp_action.setChecked(bool(self._interpolated_bad_segments) and self._show_interpolated_regions)
        toggle_interp_action.setEnabled(bool(self._interpolated_bad_segments))
        toggle_interp_action.triggered.connect(self._on_view_toggle_interpolated_regions)
        menu.addAction(toggle_interp_action)

        # Toggle gap segment overlay (enabled after resampling detects gaps)
        toggle_gaps_action = QAction("Display Gaps", self)
        toggle_gaps_action.setCheckable(True)
        toggle_gaps_action.setChecked(bool(self._current_gap_segments) and self._show_gap_segments)
        toggle_gaps_action.setEnabled(bool(self._current_gap_segments))
        toggle_gaps_action.triggered.connect(self._on_view_toggle_gap_segments)
        menu.addAction(toggle_gaps_action)

        # Toggle log panel
        toggle_log_action = QAction("Toggle Log Panel", self)
        toggle_log_action.setShortcut("L")
        toggle_log_action.triggered.connect(self._on_view_toggle_log)
        menu.addAction(toggle_log_action)

        # Toggle processing steps panel
        toggle_processing_action = QAction("Toggle Processing Panel", self)
        toggle_processing_action.setShortcut("K")
        toggle_processing_action.triggered.connect(self._on_view_toggle_processing)
        menu.addAction(toggle_processing_action)

        # Derived visualisation panels (channel view only, signal-type-specific)
        if self.current_view_level == "channel" and self.current_signal is not None:
            sig_type = self.current_signal.signal_type
            has_peaks = (
                self._current_peaks is not None
                and self._current_peaks.num_peaks >= 2
            )
            if sig_type in (SignalType.ECG, SignalType.PPG):
                hr_action = QAction("Show &Heart Rate", self)
                hr_action.setShortcut("H")
                hr_action.setCheckable(True)
                hr_action.setChecked(self.single_channel_view.is_derived_visible())
                hr_action.triggered.connect(self._on_view_toggle_heart_rate)
                menu.addAction(hr_action)

                menu.addSeparator()

                overlay_action = QAction("Heartbeat &Overlay...", self)
                overlay_action.setEnabled(has_peaks)
                overlay_action.triggered.connect(self._on_view_heartbeat_overlay)
                menu.addAction(overlay_action)

                hist_action = QAction("RR Interval &Histogram...", self)
                hist_action.setEnabled(has_peaks)
                hist_action.triggered.connect(self._on_view_rr_histogram)
                menu.addAction(hist_action)

            if sig_type == SignalType.EDA and self._eda_tonic is not None:
                eda_action = QAction("Show &EDA Components", self)
                eda_action.setShortcut("H")
                eda_action.setCheckable(True)
                eda_action.setChecked(self.single_channel_view.is_derived_visible())
                eda_action.triggered.connect(self._on_view_toggle_eda_components)
                menu.addAction(eda_action)

        # Navigation back actions
        menu.addSeparator()

        if self.current_view_level == "channel":
            if self._came_from_type_view and self.current_signal_type:
                type_name = self.current_signal_type.value.upper()
                return_action = QAction(f"Return to &{type_name} View", self)
                return_action.setShortcut(get_keysequence("view_multi_signal"))
                return_action.triggered.connect(self._on_return_to_type_view)
                menu.addAction(return_action)
            else:
                return_action = QAction("Return to &Multi-Signal View", self)
                return_action.setShortcut(get_keysequence("view_multi_signal"))
                return_action.triggered.connect(self._on_return_to_multi)
                menu.addAction(return_action)

        elif self.current_view_level == "type":
            return_action = QAction("Return to &Multi-Signal View", self)
            return_action.setShortcut(get_keysequence("view_multi_signal"))
            return_action.triggered.connect(self._on_return_to_multi)
            menu.addAction(return_action)

    def _add_help_menu_actions(self, menu):
        """Add Help menu actions."""
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(get_keysequence("help_show"))
        shortcuts_action.triggered.connect(self._on_help_shortcuts)
        menu.addAction(shortcuts_action)

        menu.addSeparator()

        about_action = QAction("&About CardioSignalLab", self)
        about_action.triggered.connect(self._on_help_about)
        menu.addAction(about_action)

    # ---- View Switching ----

    def _on_signal_type_selected(self, signal_type: SignalType):
        """Handle signal type selection from multi-signal view."""
        signals = self.multi_signal_view.get_signals_for_type(signal_type)
        self.current_signal_type = signal_type

        if len(signals) == 1:
            # Single channel - skip type view, go directly to channel view
            self._came_from_type_view = False
            self._switch_to_channel_view(signals[0])
        else:
            # Multiple channels - show type view
            self._switch_to_type_view(signal_type, signals)

    def _on_channel_selected(self, signal: SignalData):
        """Handle channel selection from signal-type view."""
        self._came_from_type_view = True
        self._switch_to_channel_view(signal)

    def _switch_to_type_view(self, signal_type: SignalType, signals: list[SignalData]):
        """Switch to signal-type view showing channels of one type."""
        self.current_view_level = "type"
        self.current_signal_type = signal_type

        # Pass events
        if self.current_session:
            self.signal_type_view.set_events(self.current_session.events or [])

        self.signal_type_view.set_signal_type(signal_type, signals)

        # Restore derived channels from specs when the view has none
        # (e.g., first visit after session load, or returning after viewing a different type).
        if not self.signal_type_view.derived_signals:
            signal_map = {s.channel_name: s for s in signals}
            for spec in self._derived_channel_specs:
                if spec.get("type") != "l2_norm":
                    continue
                if spec.get("signal_type") != signal_type.value:
                    continue
                source_sigs = [signal_map[ch] for ch in spec["source_channels"] if ch in signal_map]
                if len(source_sigs) >= 2:
                    try:
                        self.signal_type_view.add_l2_norm(source_sigs)
                        logger.info(f"Restored L2 Norm channel from session spec")
                    except Exception as e:
                        logger.warning(f"Failed to restore L2 Norm channel: {e}")

        self.stacked_widget.setCurrentWidget(self.signal_type_view)
        self._build_menus()

        # Update status bar
        n_channels = len(signals)
        self.statusBar().showMessage(
            f"{signal_type.value.upper()} View ({n_channels} channels)", 0
        )
        logger.info(f"Switched to type view: {signal_type.value} ({n_channels} channels)")

    def _channel_key(self, signal: SignalData) -> tuple:
        """Stable per-channel key for the state store."""
        return (signal.signal_type, signal.channel_name)

    def _save_channel_state(self, signal: SignalData):
        """Snapshot active processing/peak state for the given channel."""
        key = self._channel_key(signal)
        self._mark_dirty()
        # Preserve reviewed flag if already set
        existing = self._channel_state.get(key, {})
        self._channel_state[key] = {
            "peaks": self._current_peaks,
            "pipeline_steps": list(self.pipeline.steps),
            "raw_samples": self._raw_samples,
            "eda_tonic": self._eda_tonic,
            "eda_phasic": self._eda_phasic,
            "structural_ops": list(self._structural_ops),
            "bad_segments": list(self._current_bad_segments),
            "gap_segments": list(self._current_gap_segments),
            "original_samples": self._original_samples,
            "original_timestamps": self._original_timestamps,
            "original_sampling_rate": self._original_sampling_rate,
            "reviewed": existing.get("reviewed", False),
        }
        n_peaks = self._current_peaks.num_peaks if self._current_peaks else 0
        logger.debug(
            f"Saved channel state: {signal.channel_name} "
            f"({n_peaks} peaks, {len(self.pipeline.steps)} pipeline steps)"
        )

    def _switch_to_channel_view(self, signal: SignalData):
        """Switch to single-channel view for processing.

        Peaks and processing state are stored per channel so that navigating
        away and back (or switching between channels) never loses corrections.
        """
        is_same_signal = self.current_signal is signal

        # Persist state of the channel we are leaving
        if not is_same_signal and self.current_signal is not None:
            self._save_channel_state(self.current_signal)

        self.current_view_level = "channel"
        self.current_signal = signal

        if not is_same_signal:
            key = self._channel_key(signal)
            saved = self._channel_state.get(key)

            # Always clear the overlay before restoring or starting fresh.
            # Without this, the previous channel's peaks remain visible when
            # returning to a channel whose saved state has no peaks.
            self.single_channel_view.clear_peaks()
            self.single_channel_view.clear_derived()

            if saved is not None:
                # Returning to a previously visited channel — restore state
                self._current_peaks = saved["peaks"]
                self._raw_samples = saved["raw_samples"]
                self._eda_tonic = saved["eda_tonic"]
                self._eda_phasic = saved["eda_phasic"]
                self.pipeline.steps = list(saved["pipeline_steps"])
                self._structural_ops = list(saved.get("structural_ops", []))
                self._current_bad_segments = list(saved.get("bad_segments", []))
                self._current_gap_segments = list(saved.get("gap_segments", []))
                self._original_samples = saved.get("original_samples")
                self._original_timestamps = saved.get("original_timestamps")
                self._original_sampling_rate = saved.get("original_sampling_rate")
            else:
                # First visit to this channel — clean slate; capture write-once original snapshot
                self.pipeline.reset()
                self._raw_samples = None
                self._current_peaks = None
                self._eda_tonic = None
                self._eda_phasic = None
                self._structural_ops = []
                self._current_bad_segments = []
                self._current_gap_segments = []
                self._show_gap_segments = False
                self._original_samples = signal.samples.copy()
                self._original_timestamps = signal.timestamps.copy()
                self._original_sampling_rate = signal.sampling_rate
                self.processing_panel.clear()

        # Make the channel view visible first so setRange in set_signal operates on a
        # widget with proper geometry (avoids deferred sigRangeChanged with stale range).
        self.stacked_widget.setCurrentWidget(self.single_channel_view)

        # Always re-render (rebuilds LOD renderer, resets view range)
        self.single_channel_view.set_signal(signal)

        # Re-initialise the peak editor and overlay from stored peak data
        if self._current_peaks is not None:
            self.single_channel_view.set_peaks(self._current_peaks)

        # Restore bad segment overlay (may be empty on first visit)
        self.single_channel_view.set_bad_segments(self._current_bad_segments, signal)

        # Restore gap segment overlay only if toggle is on
        if self._show_gap_segments and self._current_gap_segments:
            self.single_channel_view.set_gap_segments(self._current_gap_segments, signal)
        else:
            self.single_channel_view.clear_gap_segments()

        # Restore interpolated segment overlay if toggle is on
        if self._show_interpolated_regions and self._interpolated_bad_segments:
            self.single_channel_view.set_interpolated_segments(self._interpolated_bad_segments, signal)

        # Auto-show EDA derived panel when returning to a channel where decomposition was done
        if (
            signal.signal_type == SignalType.EDA
            and self._eda_tonic is not None
            and self._eda_phasic is not None
        ):
            self._show_eda_components_panel()

        # Sync processing panel
        if self.pipeline.num_steps > 0:
            self._refresh_processing_panel()

        # Pass events
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        self._build_menus()

        n_peaks = self._current_peaks.num_peaks if self._current_peaks else 0
        restored = n_peaks > 0
        suffix = f" ({n_peaks} peaks restored)" if restored else ""
        self.statusBar().showMessage(
            f"Channel: {signal.signal_type.value.upper()} - {signal.channel_name}{suffix}", 0
        )
        logger.info(
            f"Switched to channel view: {signal.channel_name}"
            + (f" — restored {n_peaks} peaks" if restored else "")
        )

    def _on_return_to_type_view(self):
        """Return from channel view to type view."""
        if self.current_signal_type and self.current_session:
            signals = [
                s for s in self.current_session.signals
                if s.signal_type == self.current_signal_type
            ]
            self._switch_to_type_view(self.current_signal_type, signals)

    def _on_return_to_multi(self):
        """Return to multi-signal view."""
        self.current_view_level = "multi"
        self.current_signal_type = None
        self._came_from_type_view = False
        self.stacked_widget.setCurrentWidget(self.multi_signal_view)

        if self.multi_signal_view.plot_widgets:
            self.multi_signal_view.reset_view()

        self._build_menus()

        n_types = len(self.multi_signal_view.get_unique_signal_types())
        self.statusBar().showMessage(f"Multi-Signal View ({n_types} types)", 0)
        logger.info("Returned to multi-signal view")

    def _on_mode_changed(self, mode: str):
        """Handle mode change signal (legacy compat, maps to view levels)."""
        if mode == "multi":
            self._on_return_to_multi()

    # ---- File Operations ----

    def _open_recent(self, file_path: str):
        """Open a file from the recent-files list."""
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(
                self, "File Not Found",
                f"The file could not be found:\n{path}\n\nIt will be removed from the recent files list."
            )
            cfg = get_config_manager()
            cfg.get_config().gui.recent_source_files = [
                p for p in cfg.get_config().gui.recent_source_files if p != file_path
            ]
            cfg.get_config().gui.recent_session_files = [
                p for p in cfg.get_config().gui.recent_session_files if p != file_path
            ]
            cfg.save_user_config()
            self._build_menus()
            return
        if path.name.endswith(".csl.json"):
            self._load_session_file(path)
        else:
            if not self._confirm_discard_changes():
                return
            try:
                if path.suffix.lower() == ".csv":
                    loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
                else:
                    loader = get_loader(path)
                session = loader.load(path)
                self.current_session = session
                self._reset_all_channel_state()
                self._rebuild_views()
                self.current_view_level = "multi"
                self._session_notes = ""
                self._current_session_path = None
                self._mark_clean()
                get_config_manager().add_recent_file(path, "source")
                self._show_metadata_dialog(session)
                self.signals.file_loaded.emit(session)
                self._build_menus()
            except Exception as e:
                logger.exception(f"Failed to open recent file: {e}")
                QMessageBox.critical(self, "Open Error", f"Failed to open file:\n{e}")

    def _on_file_open(self):
        """Handle File > Open."""
        logger.info("File > Open triggered")

        if not self._confirm_discard_changes():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Physiological Signal File or Session",
            str(self._last_open_dir),
            "All Supported Files (*.xdf *.csv *.edf *.bdf *.csl.json);;Session Files (*.csl.json);;Physiological Signal Files (*.xdf *.csv *.edf *.bdf);;XDF Files (*.xdf);;CSV Files (*.csv);;EDF/BDF Files (*.edf *.bdf);;All Files (*.*)",
        )

        if not file_path:
            return

        path = Path(file_path)
        self._last_open_dir = path.parent
        logger.info(f"Loading file: {path}")

        try:
            # Check if it's a session file
            if path.suffix.lower() == ".json" and path.name.endswith(".csl.json"):
                self._load_session_file(path)
            else:
                # Load data file
                if path.suffix.lower() == ".csv":
                    loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
                else:
                    loader = get_loader(path)

                session = loader.load(path)
                self.current_session = session
                self._reset_all_channel_state()
                self._rebuild_views()
                self.current_view_level = "multi"
                self._session_notes = ""
                self._current_session_path = None
                self._mark_clean()
                get_config_manager().add_recent_file(path, "source")

                # Show warning if any streams were skipped due to corruption
                if hasattr(loader, 'skipped_streams') and loader.skipped_streams:
                    skipped_names = ", ".join(loader.skipped_streams)
                    QMessageBox.warning(
                        self,
                        "Data Quality Warning",
                        f"One or more signal streams could not be loaded due to timestamp corruption:\n\n"
                        f"Skipped: {skipped_names}\n\n"
                        f"Available signals ({session.num_signals}) are loaded and ready for analysis.\n\n"
                        f"Check the log panel (View → Log Panel) for details."
                    )

                self._show_metadata_dialog(session)
                self.signals.file_loaded.emit(session)
                self._build_menus()

                logger.info(f"File loaded: {session.num_signals} signals")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            QMessageBox.critical(self, "File Not Found", f"The file could not be found:\n{e}")
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Invalid file format: {error_msg}")

            # Check if this is a structural corruption error
            if "length mismatch" in error_msg.lower():
                QMessageBox.critical(
                    self,
                    "File Corrupted - Action Required",
                    "This XDF file has a structural corruption (length mismatch between data and timestamps).\n\n"
                    "This cannot be automatically fixed.\n\n"
                    "Please fix the file by:\n"
                    "1. Open the file in MATLAB\n"
                    "2. Wait for MATLAB to raise an exception\n"
                    "3. Delete the extra sample value as indicated\n"
                    "4. Re-save the file\n\n"
                    "Then you can load it here."
                )
            else:
                QMessageBox.critical(self, "Invalid File", f"The file format is invalid:\n{error_msg}")
        except Exception as e:
            logger.exception(f"Unexpected error loading file: {e}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{e}")

    def _on_file_close(self):
        """Close the current file and return to a clean initial state.

        Destroys and recreates all view widgets so that PyQtGraph scene state
        is guaranteed clean -- no stale PlotDataItems or detached ViewBox items.
        """
        if not self._confirm_discard_changes():
            return

        self._reset_all_channel_state()
        self._rebuild_views()
        self.current_session = None
        self.current_view_level = "multi"
        self.current_signal_type = None
        self._came_from_type_view = False
        self._session_notes = ""
        self._current_session_path = None
        self._mark_clean()
        self.status_bar.clear()
        self.processing_panel.clear()
        self._build_menus()
        logger.info("File closed — clean slate")

    def _load_session_file(self, path: Path):
        """Load a .csl.json session file."""
        if not self._confirm_discard_changes():
            return
        try:
            session_data = load_session(path)

            # Version compatibility check
            import cardio_signal_lab as _csl_pkg
            saved_version = session_data.get("meta", {}).get("app_version", "")
            if saved_version:
                def _version_tuple(v: str) -> tuple:
                    try:
                        return tuple(int(x) for x in v.split(".")[:3])
                    except ValueError:
                        return (0, 0, 0)
                if _version_tuple(saved_version) > _version_tuple(_csl_pkg.__version__):
                    QMessageBox.warning(
                        self, "Session Version Mismatch",
                        f"This session was saved by a newer version of CardioSignalLab "
                        f"({saved_version}) than the one currently running "
                        f"({_csl_pkg.__version__}).\n\n"
                        f"Some features may not restore correctly."
                    )

            # Load the source file
            source_path = Path(session_data["source_file"])
            if not source_path.exists():
                # Try relative path (session file and source in the same folder)
                relative_candidate = path.parent / source_path.name
                if relative_candidate.exists():
                    logger.info(
                        f"Source file not found at original path; using relative path: "
                        f"{relative_candidate}"
                    )
                    source_path = relative_candidate
                else:
                    reply = QMessageBox.question(
                        self, "Source File Not Found",
                        f"The session references a source file that could not be found:\n{source_path}\n\nWould you like to locate it manually?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    located, _ = QFileDialog.getOpenFileName(
                        self,
                        "Locate Source File",
                        str(self._last_open_dir),
                        "Data Files (*.xdf *.csv *.mat *.edf *.bdf);;All Files (*.*)",
                    )
                    if not located:
                        return
                    source_path = Path(located)
                    self._last_open_dir = source_path.parent

            # Verify source file integrity
            if not verify_source_checksum(source_path, session_data.get("source_file_sha256")):
                QMessageBox.warning(
                    self, "Source File Changed",
                    f"The source file '{source_path.name}' has changed since this session "
                    f"was saved.\n\nPeak indices and pipeline results may no longer align "
                    f"with the data. Proceed with caution."
                )

            # Load the data file
            if source_path.suffix.lower() == ".csv":
                loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
            else:
                loader = get_loader(source_path)

            session = loader.load(source_path)
            self.current_session = session
            self._reset_all_channel_state()
            self._rebuild_views()
            self.current_view_level = "multi"
            self._session_notes = session_data.get("notes", "")
            self._derived_channel_specs = list(session_data.get("derived_channels", []))

            # Restore per-channel state.  The session keys use string signal_type values;
            # map them back to (SignalType, channel_name) tuples matching _channel_key().
            signal_map = {
                (sig.signal_type.value, sig.channel_name): sig
                for sig in session.signals
            }
            restored_channels = 0
            failed_channels: list[str] = []
            all_skipped_ops: list[str] = []

            for (signal_type_str, channel_name), state in session_data.get("channels", {}).items():
                sig = signal_map.get((signal_type_str, channel_name))
                if sig is None:
                    logger.warning(
                        f"Session channel '{signal_type_str}|{channel_name}' not found in "
                        f"loaded file; skipping"
                    )
                    continue

                try:
                    key = self._channel_key(sig)

                    # Capture original samples/timestamps for this channel (needed for pipeline replay)
                    state["original_samples"] = sig.samples.copy()
                    state["original_timestamps"] = sig.timestamps.copy()
                    state["original_sampling_rate"] = sig.sampling_rate

                    # Replay the pipeline to produce processed samples
                    if state["pipeline_steps"]:
                        from cardio_signal_lab.processing.pipeline import ProcessingPipeline
                        replay_pipeline = ProcessingPipeline()
                        replay_pipeline.steps = list(state["pipeline_steps"])
                        skipped = getattr(replay_pipeline, "skipped_unknown_ops", [])
                        all_skipped_ops.extend(skipped)
                        processed = replay_pipeline.apply(
                            sig.samples.copy(), sig.sampling_rate, skip_on_error=True
                        )
                        state["raw_samples"] = processed
                        logger.info(
                            f"Replayed {len(replay_pipeline.steps)} pipeline step(s) for "
                            f"'{channel_name}'"
                        )

                    self._channel_state[key] = state
                    restored_channels += 1

                except Exception as ch_err:
                    logger.warning(
                        f"Failed to restore channel '{channel_name}': {ch_err}; skipping"
                    )
                    failed_channels.append(channel_name)

            logger.info(f"Restored {restored_channels} channel(s) from session")

            # Populate the multi-signal view and rebuild menus
            self.signals.file_loaded.emit(session)
            self._build_menus()

            total_peaks = sum(
                state["peaks"].num_peaks
                for state in self._channel_state.values()
                if state.get("peaks") is not None
            )
            meta = session_data.get("meta", {})
            saved_by = f"\nSaved by: {meta['operator']}" if meta.get("operator") else ""
            saved_at = f"\nSaved at: {meta['saved_at'][:19].replace('T', ' ')} UTC" if meta.get("saved_at") else ""
            warnings: list[str] = []
            if failed_channels:
                warnings.append(f"Channels not restored: {', '.join(failed_channels)}")
            if all_skipped_ops:
                unique_ops = sorted(set(all_skipped_ops))
                warnings.append(f"Unknown pipeline steps skipped: {', '.join(unique_ops)}")
            warning_text = ("\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)) if warnings else ""
            self._current_session_path = path
            self._mark_clean()
            get_config_manager().add_recent_file(path, "session")
            QMessageBox.information(
                self, "Session Loaded",
                f"Session loaded successfully.\n\nSource: {source_path.name}\n"
                f"Channels restored: {restored_channels}\nTotal peaks: {total_peaks}"
                f"{saved_by}{saved_at}{warning_text}"
            )

        except Exception as e:
            logger.exception(f"Failed to load session: {e}")
            QMessageBox.critical(self, "Session Load Error", f"Failed to load session:\n{e}")

    def _on_file_append(self):
        """Append a continuation file to the currently loaded session.

        Assumes the second file is a direct continuation of the first (e.g.,
        after a crash or disconnection).  For each channel, file2's timestamps
        are offset so the concatenated series is strictly continuous:

            new_ts = file2_ts + ((file1_last_ts + 1/sr) - file2_first_ts)

        Channels are matched by channel_name + signal_type, falling back to
        signal_type only if names differ.  Unmatched channels are skipped with
        a warning.  Events from file2 receive the same timestamp offset.
        """
        if self.current_session is None:
            return

        # Warn and confirm if in channel view (processing state will be lost)
        if self.current_view_level == "channel":
            reply = QMessageBox.question(
                self,
                "Append File",
                "Appending a file will clear the current processing state and "
                "return to the multi-signal view.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Append Continuation File",
            str(self._last_open_dir),
            "Physiological Signal Files (*.xdf *.csv *.edf *.bdf);;XDF Files (*.xdf);;CSV Files (*.csv);;EDF/BDF Files (*.edf *.bdf);;All Files (*.*)",
        )
        if not file_path:
            return
        self._last_open_dir = Path(file_path).parent

        path = Path(file_path)
        try:
            if path.suffix.lower() == ".csv":
                loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
            else:
                loader = get_loader(path)
            session2 = loader.load(path)
        except Exception as e:
            logger.error(f"Append: failed to load {path}: {e}")
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{e}")
            return

        if not session2.signals:
            QMessageBox.warning(self, "Append", "The selected file contains no signals.")
            return

        # ------------------------------------------------------------------
        # Match and concatenate channels
        # ------------------------------------------------------------------
        appended_channels = 0
        skipped: list[str] = []
        ts_offset_for_events: float | None = None  # computed from first matched pair

        for sig1 in self.current_session.signals:
            # Prefer exact name+type match; fall back to type-only
            sig2 = next(
                (s for s in session2.signals
                 if s.channel_name == sig1.channel_name
                 and s.signal_type == sig1.signal_type),
                None,
            )
            if sig2 is None:
                sig2 = next(
                    (s for s in session2.signals if s.signal_type == sig1.signal_type),
                    None,
                )

            if sig2 is None:
                skipped.append(
                    f"'{sig1.channel_name}' ({sig1.signal_type.value}): "
                    "no matching channel in second file"
                )
                continue

            if abs(sig1.sampling_rate - sig2.sampling_rate) > 1.0:
                skipped.append(
                    f"'{sig1.channel_name}': sampling rate mismatch "
                    f"({sig1.sampling_rate:.1f} vs {sig2.sampling_rate:.1f} Hz) — skipped"
                )
                continue

            # Compute offset: file2 starts exactly one sample after file1 ends
            dt = 1.0 / sig1.sampling_rate
            ts_offset = (float(sig1.timestamps[-1]) + dt) - float(sig2.timestamps[0])

            if ts_offset_for_events is None:
                ts_offset_for_events = ts_offset

            new_timestamps = sig2.timestamps + ts_offset
            combined_samples = np.concatenate([sig1.samples, sig2.samples])
            combined_timestamps = np.concatenate([sig1.timestamps, new_timestamps])

            object.__setattr__(sig1, "samples", combined_samples)
            object.__setattr__(sig1, "timestamps", combined_timestamps)
            appended_channels += 1
            logger.info(
                f"Append: '{sig1.channel_name}' — "
                f"{len(sig2.samples)} samples added, "
                f"offset={ts_offset:.4f} s, "
                f"total={len(combined_samples)} samples "
                f"({combined_timestamps[-1] - combined_timestamps[0]:.1f} s)"
            )

        if appended_channels == 0:
            QMessageBox.warning(
                self, "Append",
                "No channels could be matched between the two files.\n\n"
                + "\n".join(skipped),
            )
            return

        # ------------------------------------------------------------------
        # Append events from file2 (offset to match signal continuity)
        # ------------------------------------------------------------------
        if session2.events and ts_offset_for_events is not None:
            from cardio_signal_lab.core.data_models import EventData
            new_events = [
                EventData(
                    timestamp=ev.timestamp + ts_offset_for_events,
                    label=ev.label,
                )
                for ev in session2.events
            ]
            existing = list(self.current_session.events or [])
            existing.extend(new_events)
            object.__setattr__(self.current_session, "events", existing)
            logger.info(f"Append: {len(new_events)} events added from second file")

        # ------------------------------------------------------------------
        # Clear per-channel state and refresh view
        # ------------------------------------------------------------------
        self._channel_state.clear()
        self._current_peaks = None
        self._raw_samples = None
        self._original_samples = None
        self._original_timestamps = None
        self._original_sampling_rate = None
        self._eda_tonic = None
        self._eda_phasic = None
        self.pipeline.reset()
        self._structural_ops.clear()
        self.current_signal = None

        # Navigate back to multi-signal view
        self.current_view_level = "multi"
        self.current_signal_type = None
        self._came_from_type_view = False
        self.stacked_widget.setCurrentWidget(self.multi_signal_view)
        self.multi_signal_view.set_session(self.current_session)
        self._build_menus()

        # Status / warnings
        if skipped:
            QMessageBox.warning(
                self, "Append — Some Channels Skipped",
                "\n".join(skipped),
            )

        total_dur = max(
            (sig.timestamps[-1] - sig.timestamps[0])
            for sig in self.current_session.signals
        )
        self.statusBar().showMessage(
            f"Appended {appended_channels} channel(s) from '{path.name}'  "
            f"— total duration {total_dur:.1f} s",
            0,
        )
        logger.info(
            f"Append complete: {appended_channels} channels from '{path.name}', "
            f"total duration {total_dur:.1f} s"
        )

    def _show_metadata_dialog(self, session: RecordingSession):
        """Display file metadata in an info dialog."""
        signal_count = session.num_signals
        signal_types = {}
        for sig in session.signals:
            sig_type = sig.signal_type.value
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1

        type_summary = "\n".join(
            [f"  - {st.upper()}: {count} channel(s)" for st, count in signal_types.items()]
        )

        sampling_rates = set()
        durations = []
        for sig in session.signals:
            sampling_rates.add(f"{sig.sampling_rate:.2f} Hz")
            durations.append(sig.duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        metadata_msg = (
            f"File: {session.source_path.name}\n\n"
            f"Signals: {signal_count}\n"
            f"{type_summary}\n\n"
            f"Sampling Rates: {', '.join(sorted(sampling_rates))}\n"
            f"Average Duration: {avg_duration:.2f} seconds"
        )

        QMessageBox.information(self, "File Metadata", metadata_msg)

    def _on_file_loaded_signal(self, session: RecordingSession):
        """Handle file_loaded signal."""
        if not self.current_session:
            self.current_session = session
        self._last_open_dir = session.source_path.parent
        self.multi_signal_view.set_session(session)
        logger.debug(f"Session loaded: {session.num_signals} signals, {len(session.events)} events")

    def _on_file_save(self):
        """Handle File > Save.  Saves in-place if a session path is known."""
        self._do_save(use_existing_path=True)

    def _on_file_save_as(self):
        """Handle File > Save As — always prompts for a new path."""
        self._do_save(use_existing_path=False)

    def _do_save(self, *, use_existing_path: bool):
        """Core save logic, shared by Save and Save As."""
        logger.info("Save triggered")

        if self.current_session is None:
            QMessageBox.warning(self, "No Session", "No session to save. Load a file first.")
            return

        if use_existing_path and self._current_session_path is not None:
            path = self._current_session_path
        else:
            default_name = self.current_session.source_path.stem + ".csl.json"
            default_dir = (
                self._current_session_path.parent
                if self._current_session_path
                else self._last_save_dir
            )
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Session",
                str(default_dir / default_name),
                "Session Files (*.csl.json);;All Files (*.*)",
            )
            if not file_path:
                return
            path = Path(file_path)
            self._last_save_dir = path.parent

        try:
            # Flush the active channel's live state into _channel_state before saving
            if self.current_signal is not None:
                self._save_channel_state(self.current_signal)

            # Get view state (current zoom/pan)
            view_state = {}
            if self.current_view_level == "channel" and self.current_signal is not None:
                plot = self.single_channel_view.plot_widget
                x_min, x_max, y_min, y_max = plot.get_visible_range()
                view_state = {
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                    "signal_type": self.current_signal.signal_type.value,
                    "channel_name": self.current_signal.channel_name,
                }

            save_session(
                source_file=self.current_session.source_path,
                channel_states=self._channel_state,
                output_path=path,
                view_state=view_state,
                operator_name=get_config().gui.operator_name,
                notes=self._session_notes,
                derived_channels=self._derived_channel_specs,
            )

            self._current_session_path = path
            self._mark_clean()
            get_config_manager().add_recent_file(path, "session")

            n_channels = len(self._channel_state)
            self.statusBar().showMessage(
                f"Session saved to {path.name} ({n_channels} channel(s))", 5000
            )
            logger.info(f"Session saved to {path}")

        except Exception as e:
            logger.exception(f"Failed to save session: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{e}")

    def _on_file_export(self):
        """Handle File > Export - export processed signal and peaks."""
        logger.info("File > Export triggered")

        # Can only export from channel view
        if self.current_signal is None:
            QMessageBox.warning(
                self, "No Signal",
                "Export is only available when viewing a single signal.\n\nSelect a signal from the multi-signal or type view first."
            )
            return

        # Show export format dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Signal")
        layout = QFormLayout(dialog)

        format_combo = QComboBox()
        format_combo.addItems([
            "CSV (signal + peaks)",
            "NumPy Arrays (.npy)",
            "Annotations Only (CSV)",
            "RR-Intervals (CSV)",
            "NN-Intervals (CSV)",
        ])
        layout.addRow("Export format:", format_combo)

        from PySide6.QtWidgets import QCheckBox
        include_peaks_check = QCheckBox()
        include_peaks_check.setChecked(True)
        layout.addRow("Include peaks:", include_peaks_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        export_format = format_combo.currentText()

        # Interval exports need a parameter dialog before the file picker
        interval_params: dict | None = None
        if export_format in ("RR-Intervals (CSV)", "NN-Intervals (CSV)"):
            if self._current_peaks is None or self._current_peaks.num_peaks < 2:
                QMessageBox.warning(
                    self, "No Peaks",
                    "Interval export requires at least 2 detected peaks.\n"
                    "Run Process > Detect Peaks first."
                )
                return

            param_dialog = QDialog(self)
            param_dialog.setWindowTitle("Interval Export Parameters")
            playout = QFormLayout(param_dialog)

            playout.addRow(QLabel(
                "Physiological validity: reject intervals outside [min, max] ms.\n"
                "Statistical validity: reject outliers beyond threshold x MAD\n"
                "from the median of all intervals."
            ))

            min_spin = QDoubleSpinBox()
            min_spin.setRange(100.0, 1000.0)
            min_spin.setSingleStep(50.0)
            min_spin.setValue(300.0)
            min_spin.setSuffix(" ms")
            playout.addRow("Min interval (physiological):", min_spin)

            max_spin = QDoubleSpinBox()
            max_spin.setRange(500.0, 5000.0)
            max_spin.setSingleStep(100.0)
            max_spin.setValue(2000.0)
            max_spin.setSuffix(" ms")
            playout.addRow("Max interval (physiological):", max_spin)

            thresh_spin = QDoubleSpinBox()
            thresh_spin.setDecimals(1)
            thresh_spin.setRange(1.0, 10.0)
            thresh_spin.setSingleStep(0.5)
            thresh_spin.setValue(4.0)
            thresh_spin.setToolTip("Outlier threshold in multiples of MAD from the median.")
            playout.addRow("Statistical threshold (x MAD):", thresh_spin)

            pbuttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            pbuttons.accepted.connect(param_dialog.accept)
            pbuttons.rejected.connect(param_dialog.reject)
            playout.addRow(pbuttons)

            if param_dialog.exec() != QDialog.DialogCode.Accepted:
                return

            interval_params = {
                "min_rr_ms": min_spin.value(),
                "max_rr_ms": max_spin.value(),
                "stat_threshold": thresh_spin.value(),
            }

        # Determine file extension and filter for the file picker
        if export_format == "CSV (signal + peaks)":
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
            default_ext = ".csv"
        elif export_format == "NumPy Arrays (.npy)":
            file_filter = "NumPy Files (*.npy);;All Files (*.*)"
            default_ext = "_signal.npy"
        elif export_format == "RR-Intervals (CSV)":
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
            default_ext = "_rr_intervals.csv"
        elif export_format == "NN-Intervals (CSV)":
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
            default_ext = "_nn_intervals.csv"
        else:  # Annotations
            file_filter = "CSV Files (*.csv);;All Files (*.*)"
            default_ext = "_annotations.csv"

        channel_part = self.current_signal.channel_name.replace(" ", "_").lower()
        source_stem = (
            self.current_session.source_path.stem if self.current_session else ""
        )
        default_name = (
            f"{source_stem}_{channel_part}{default_ext}"
            if source_stem
            else f"{channel_part}{default_ext}"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Signal", str(self._last_save_dir / default_name), file_filter
        )

        if not file_path:
            return

        path = Path(file_path)
        self._last_save_dir = path.parent

        try:
            if export_format == "CSV (signal + peaks)":
                include_peaks = include_peaks_check.isChecked()
                export_csv(
                    signal=self.current_signal,
                    peaks=self._current_peaks,
                    output_path=path,
                    include_peaks=include_peaks,
                )
                self.statusBar().showMessage(f"Exported to CSV: {path.name}", 5000)

            elif export_format == "NumPy Arrays (.npy)":
                base_path = path.parent / path.stem.replace("_signal", "")
                export_npy(
                    signal=self.current_signal,
                    peaks=self._current_peaks,
                    output_path=base_path,
                )
                self.statusBar().showMessage(f"Exported to NPY: {base_path.name}*", 5000)

            elif export_format in ("RR-Intervals (CSV)", "NN-Intervals (CSV)"):
                mode = "rr" if export_format.startswith("RR") else "nn"
                events = self.current_session.events if self.current_session else None
                export_intervals(
                    signal=self.current_signal,
                    peaks=self._current_peaks,
                    output_path=path,
                    mode=mode,
                    events=events or [],
                    **interval_params,
                )
                label = "RR" if mode == "rr" else "NN"
                self.statusBar().showMessage(f"Exported {label} intervals: {path.name}", 5000)

            else:  # Annotations
                if self._current_peaks is None or self._current_peaks.num_peaks == 0:
                    QMessageBox.warning(
                        self, "No Peaks",
                        "No peaks detected. Run Process > Detect Peaks first or add peaks manually."
                    )
                    return
                export_annotations(
                    signal=self.current_signal,
                    peaks=self._current_peaks,
                    output_path=path,
                )
                self.statusBar().showMessage(f"Exported annotations: {path.name}", 5000)

            # Also save processing parameters as sidecar
            if self.pipeline.num_steps > 0:
                params_path = path.parent / (path.stem + "_processing.json")
                save_processing_parameters(
                    pipeline_steps=self.pipeline.steps,
                    signal_type=self.current_signal.signal_type.value,
                    sampling_rate=self.current_signal.sampling_rate,
                    output_path=params_path,
                )
                logger.info(f"Saved processing parameters to {params_path}")

            logger.info(f"Exported signal to {path}")

        except Exception as e:
            logger.exception(f"Failed to export: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _on_file_import_events(self):
        """Handle File > Import Events — replace session events from a CSV file."""
        if self.current_session is None:
            QMessageBox.warning(self, "No Session", "Load a file first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Events (CSV)",
            str(self._last_open_dir),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return
        self._last_open_dir = Path(file_path).parent

        try:
            events = load_events_csv(Path(file_path))
        except Exception as e:
            logger.exception(f"Failed to load events: {e}")
            QMessageBox.critical(self, "Import Error", f"Could not load events:\n{e}")
            return

        # Apply the same LSL zero-reference offset used when loading the XDF so that
        # imported event timestamps align with the zero-referenced signal x-axis.
        lsl_t0 = self.current_session.lsl_t0_reference or 0.0
        if lsl_t0 != 0.0:
            from cardio_signal_lab.core.data_models import EventData as _EventData
            events = [
                _EventData(
                    timestamp=ev.timestamp - lsl_t0,
                    label=ev.label,
                    duration=ev.duration,
                    metadata=ev.metadata,
                )
                for ev in events
            ]
            logger.info(
                f"Applied lsl_t0_reference offset ({lsl_t0:.3f} s) to {len(events)} imported events"
            )

        # Replace events on the session and refresh every active view
        object.__setattr__(self.current_session, "events", events)

        # multi_signal_view reads events through set_session; rebuild it with updated session
        self.multi_signal_view.set_session(self.current_session)
        self.signal_type_view.set_events(events)
        self.single_channel_view.set_events(events)

        self.statusBar().showMessage(
            f"Imported {len(events)} events from {Path(file_path).name}", 5000
        )
        logger.info(f"Replaced session events with {len(events)} from {file_path}")

    def _on_file_edit_events(self):
        """Handle File > Edit Events — open spreadsheet-style event editor."""
        if self.current_session is None:
            QMessageBox.warning(self, "No Session", "Load a file first.")
            return

        current_events = self.current_session.events or []
        lsl_t0 = self.current_session.lsl_t0_reference or 0.0
        dialog = EventEditorDialog(current_events, lsl_t0_reference=lsl_t0, parent=self)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        events = dialog.get_events()

        # Replace events on the session and refresh all views
        object.__setattr__(self.current_session, "events", events)
        self.multi_signal_view.set_session(self.current_session)
        self.signal_type_view.set_events(events)
        self.single_channel_view.set_events(events)

        self.statusBar().showMessage(f"Events updated ({len(events)} events)", 5000)
        logger.info(f"Events replaced via editor: {len(events)} events")

    def _on_file_import_peaks(self):
        """Handle File > Import Peaks — load pre-corrected peaks from a binary CSV."""
        if self.current_signal is None:
            QMessageBox.warning(
                self, "No Signal",
                "Navigate to a single-channel view before importing peaks."
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Corrected Peaks (CSV)",
            str(self._last_open_dir),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return
        self._last_open_dir = Path(file_path).parent

        try:
            peak_data = load_peaks_binary_csv(
                Path(file_path),
                signal_length=len(self.current_signal.samples),
            )
        except Exception as e:
            logger.exception(f"Failed to load peaks: {e}")
            QMessageBox.critical(self, "Import Error", f"Could not load peaks:\n{e}")
            return

        if peak_data.num_peaks == 0:
            QMessageBox.warning(
                self, "No Peaks Found",
                f"The file contained no peaks (no 1-values in the peaks column).\n\n"
                f"File: {Path(file_path).name}"
            )
            return

        # Confirm before overwriting any existing peaks
        if self._current_peaks is not None and self._current_peaks.num_peaks > 0:
            reply = QMessageBox.question(
                self, "Replace Peaks?",
                f"This channel already has {self._current_peaks.num_peaks} peaks.\n"
                f"Replace them with {peak_data.num_peaks} imported peaks?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._current_peaks = peak_data
        self.single_channel_view.set_peaks(peak_data)
        self.signals.peaks_updated.emit(peak_data)
        self._build_menus()
        self.statusBar().showMessage(
            f"Imported {self._peak_status(peak_data)} from {Path(file_path).name}", 0
        )
        logger.info(
            f"Imported {peak_data.num_peaks} peaks from {file_path}"
        )

    # ---- Edit Operations ----

    def _on_edit_undo(self):
        self.single_channel_view.undo()

    def _on_edit_redo(self):
        self.single_channel_view.redo()

    # ------------------------------------------------------------------
    # Dirty state
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        """Mark session as having unsaved changes."""
        if not self._is_dirty:
            self._is_dirty = True
            title = self.windowTitle().rstrip(" *")
            self.setWindowTitle(f"{title} *")

    def _mark_clean(self):
        """Clear unsaved-changes flag."""
        self._is_dirty = False
        title = self.windowTitle().rstrip(" *")
        self.setWindowTitle(title)

    def _confirm_discard_changes(self) -> bool:
        """Prompt to save if dirty.  Returns True if it is safe to proceed.

        Returns False if the user cancelled (work should not be discarded).
        """
        if not self._is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._on_file_save()
            # If still dirty the save was cancelled — abort the caller's action
            return not self._is_dirty
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    def closeEvent(self, event):
        if not self._confirm_discard_changes():
            event.ignore()
            return
        event.accept()

    def _on_toggle_reviewed(self, checked: bool):
        """Toggle the reviewed flag for the current channel."""
        if self.current_signal is None:
            return
        key = self._channel_key(self.current_signal)
        if key not in self._channel_state:
            self._save_channel_state(self.current_signal)
        self._channel_state[key]["reviewed"] = checked
        self._mark_dirty()
        label = self.current_signal.channel_name
        state = "reviewed" if checked else "unreviewed"
        self.statusBar().showMessage(f"{label} marked as {state}", 3000)
        logger.info(f"Channel '{label}' marked {state}")
        # Rebuild menus so the checkmark reflects the new state
        self._build_menus()

    def _on_session_notes(self):
        """Open the session notes editor dialog."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit, QVBoxLayout, QLabel
        dialog = QDialog(self)
        dialog.setWindowTitle("Session Notes")
        dialog.resize(500, 300)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Notes about this recording session:"))
        editor = QTextEdit()
        editor.setPlainText(self._session_notes)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._session_notes = editor.toPlainText()
            self._mark_dirty()

    @staticmethod
    def _peak_status(peak_data) -> str:
        """One-line peak breakdown for the status bar.

        Example: '847 peaks  auto=820  manual=12  ectopic=8  bad=7'
        """
        p = peak_data
        parts = [f"{p.num_peaks} peaks"]
        if p.num_auto:
            parts.append(f"auto={p.num_auto}")
        if p.num_manual:
            parts.append(f"manual={p.num_manual}")
        if p.num_ectopic:
            parts.append(f"ectopic={p.num_ectopic}")
        if p.num_bad:
            parts.append(f"bad={p.num_bad}")
        return "  ".join(parts)

    def _on_peaks_changed(self, peak_data):
        """Sync peak data from the editor back to MainWindow state."""
        self._current_peaks = peak_data
        self._mark_dirty()
        self.signals.peaks_updated.emit(peak_data)

        # Show live breakdown in the status bar
        self.statusBar().showMessage(self._peak_status(peak_data), 0)

        # Auto-refresh heart rate panel if it is visible (ECG/PPG only)
        if (
            self.single_channel_view.is_derived_visible()
            and self.current_signal is not None
            and self.current_signal.signal_type in (SignalType.ECG, SignalType.PPG)
        ):
            self._refresh_heart_rate_panel()

    # ---- Process Operations ----

    def _refresh_processing_panel(self):
        """Sync the processing panel with current structural ops + pipeline steps."""
        self.processing_panel.update_combined(self._structural_ops, self.pipeline.steps)

    def _ensure_raw_backup(self):
        """Ensure we have a backup of the raw signal before processing."""
        if self._raw_samples is None and self.current_signal is not None:
            self._raw_samples = self.current_signal.samples.copy()

    def _refresh_gap_overlay(self):
        """Re-render gap overlay using current signal samples (call after any processing step)."""
        if self._show_gap_segments and self._current_gap_segments and self.current_signal is not None:
            self.single_channel_view.set_gap_segments(self._current_gap_segments, self.current_signal)

    def _apply_pipeline_and_update(self):
        """Re-apply pipeline from raw signal and update the plot."""
        if self._raw_samples is None or self.current_signal is None:
            return

        processed = self.pipeline.apply(self._raw_samples, self.current_signal.sampling_rate)

        # Update signal samples in-place (attrs doesn't allow direct assignment)
        object.__setattr__(self.current_signal, "samples", processed)

        # Refresh the plot
        self.single_channel_view.set_signal(self.current_signal)
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        # Gap overlay y-positions depend on sample values — refresh after processing
        self._refresh_gap_overlay()

        self._refresh_processing_panel()
        self.statusBar().showMessage(
            f"Processing applied ({self.pipeline.num_steps} steps)", 3000
        )

        # Re-derive EDA components if the pipeline contains a decompose step
        self._recompute_eda_components_if_needed()

    def _recompute_eda_components_if_needed(self):
        """Recompute EDA tonic/phasic after pipeline replay if a decompose step exists.

        Called after _apply_pipeline_and_update so the derived panel stays in
        sync when the user adds further filters on top of a decomposed EDA signal.
        """
        if self.current_signal is None:
            return
        if self.current_signal.signal_type != SignalType.EDA:
            return

        decompose_step = next(
            (s for s in self.pipeline.steps if s.operation == "eda_decompose"), None
        )
        if decompose_step is None:
            return

        import neurokit2 as nk

        method = decompose_step.parameters.get("method", "highpass")
        try:
            sr = int(self.current_signal.sampling_rate)
            components = nk.eda_phasic(
                self.current_signal.samples, sampling_rate=sr, method=method
            )
            self._eda_tonic = np.asarray(components["EDA_Tonic"])
            self._eda_phasic = np.asarray(components["EDA_Phasic"])

            # Refresh derived panel if it is currently shown
            if self.single_channel_view.is_derived_visible():
                self._show_eda_components_panel()

            logger.debug("EDA components recomputed after pipeline replay")
        except Exception as e:
            logger.warning(f"EDA component recomputation failed: {e}")

    def _on_process_filter(self):
        """Handle Process > Bandpass Filter."""
        if self.current_signal is None:
            return

        config = get_config()
        sig_type = self.current_signal.signal_type

        # Set defaults based on signal type
        if sig_type == SignalType.ECG:
            default_low = config.processing.ecg_lowcut
            default_high = config.processing.ecg_highcut
            default_order = config.processing.ecg_filter_order
        elif sig_type == SignalType.PPG:
            default_low = config.processing.ppg_lowcut
            default_high = config.processing.ppg_highcut
            default_order = config.processing.ppg_filter_order
        else:
            default_low = config.processing.eda_lowcut
            default_high = config.processing.eda_highcut
            default_order = 4

        dialog = QDialog(self)
        dialog.setWindowTitle("Bandpass Filter")
        layout = QFormLayout(dialog)

        low_spin = QDoubleSpinBox()
        low_spin.setDecimals(2)
        low_spin.setRange(0.01, 500.0)
        low_spin.setValue(default_low)
        low_spin.setSuffix(" Hz")
        layout.addRow("Low cutoff:", low_spin)

        high_spin = QDoubleSpinBox()
        high_spin.setDecimals(2)
        high_spin.setRange(0.01, 500.0)
        high_spin.setValue(default_high)
        high_spin.setSuffix(" Hz")
        layout.addRow("High cutoff:", high_spin)

        order_spin = QSpinBox()
        order_spin.setRange(1, 10)
        order_spin.setValue(default_order)
        layout.addRow("Filter order:", order_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        lowcut = low_spin.value()
        highcut = high_spin.value()
        order = order_spin.value()

        if lowcut >= highcut:
            QMessageBox.warning(self, "Invalid", "Low cutoff must be less than high cutoff.")
            return

        self._ensure_raw_backup()
        self.pipeline.add_step("bandpass", {
            "lowcut": lowcut, "highcut": highcut, "order": order
        })
        self._apply_pipeline_and_update()
        logger.info(f"Bandpass filter applied: {lowcut}-{highcut} Hz, order {order}")

    def _on_process_notch(self):
        """Handle Process > Notch Filter."""
        if self.current_signal is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Notch Filter")
        layout = QFormLayout(dialog)

        freq_spin = QDoubleSpinBox()
        freq_spin.setDecimals(1)
        freq_spin.setRange(1.0, 500.0)
        freq_spin.setValue(60.0)
        freq_spin.setSuffix(" Hz")
        layout.addRow("Notch frequency:", freq_spin)

        q_spin = QDoubleSpinBox()
        q_spin.setDecimals(1)
        q_spin.setRange(1.0, 100.0)
        q_spin.setValue(30.0)
        layout.addRow("Quality factor:", q_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._ensure_raw_backup()
        self.pipeline.add_step("notch", {
            "freq": freq_spin.value(), "quality_factor": q_spin.value()
        })
        self._apply_pipeline_and_update()

    def _on_process_baseline(self):
        """Handle Process > Detrend (Polynomial)."""
        if self.current_signal is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Detrend (Polynomial)")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Fits a polynomial to the signal and subtracts it,\n"
            "removing slow baseline drift (respiration, electrode motion).\n"
            "Order 1 = linear detrend, order 3 = cubic detrend."
        ))

        order_spin = QSpinBox()
        order_spin.setRange(1, 10)
        order_spin.setValue(3)
        layout.addRow("Polynomial order:", order_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._ensure_raw_backup()
        self.pipeline.add_step("baseline_correction", {"poly_order": order_spin.value()})
        self._apply_pipeline_and_update()

    def _on_process_zero_reference(self):
        """Handle Process > DC Offset Removal."""
        if self.current_signal is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("DC Offset Removal")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Removes a constant offset from the signal.\n"
            "'mean': subtracts the signal mean (centers around zero).\n"
            "'first_n': subtracts the mean of the first N samples (useful\n"
            "when the signal starts at a known baseline)."
        ))

        method_combo = QComboBox()
        method_combo.addItems(["mean", "first_n"])
        layout.addRow("Method:", method_combo)

        n_spin = QSpinBox()
        n_spin.setRange(1, 10000)
        n_spin.setValue(100)
        layout.addRow("N samples (first_n only):", n_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._ensure_raw_backup()
        params = {"method": method_combo.currentText()}
        if method_combo.currentText() == "first_n":
            params["n_samples"] = n_spin.value()
        self.pipeline.add_step("zero_reference", params)
        self._apply_pipeline_and_update()

    def _on_process_artifact_removal(self):
        """Handle Process > Artifact Removal (EEMD)."""
        if self.current_signal is None:
            return

        config = get_config()

        dialog = QDialog(self)
        dialog.setWindowTitle("EEMD Artifact Removal")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel("Warning: EEMD is slow (30-90s for long signals)."))

        ensemble_spin = QSpinBox()
        ensemble_spin.setRange(50, 2000)
        ensemble_spin.setValue(config.processing.eemd_ensemble_size)
        layout.addRow("Ensemble size:", ensemble_spin)

        noise_spin = QDoubleSpinBox()
        noise_spin.setDecimals(2)
        noise_spin.setRange(0.01, 1.0)
        noise_spin.setValue(config.processing.eemd_noise_width)
        layout.addRow("Noise width:", noise_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._ensure_raw_backup()

        ensemble_size = ensemble_spin.value()
        noise_width = noise_spin.value()

        # Show progress dialog
        progress = QProgressDialog("Running EEMD decomposition...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Artifact Removal")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Run EEMD decomposition in background thread.
        # The worker returns (imfs, residue); reconstruction happens after the
        # user reviews and confirms IMF selection in the dialog.
        from cardio_signal_lab.processing.eemd import (
            eemd_decompose,
            analyze_imf_characteristics,
            auto_select_artifact_imfs,
            reconstruct_from_imfs,
        )
        from cardio_signal_lab.gui.imf_selection_dialog import ImfSelectionDialog

        # Get current processed signal (pipeline applied up to now)
        current_samples = self.current_signal.samples.copy()
        sr = self.current_signal.sampling_rate

        def run_eemd():
            return eemd_decompose(
                current_samples,
                ensemble_size=ensemble_size,
                noise_width=noise_width,
                random_seed=config.processing.random_seed,
            )

        worker = ProcessingWorker(run_eemd)
        self._processing_worker = worker

        def on_finished(result):
            progress.close()
            self._processing_worker = None

            imfs, residue = result
            characteristics = analyze_imf_characteristics(imfs, sr)
            auto_excluded   = auto_select_artifact_imfs(characteristics)

            sel_dialog = ImfSelectionDialog(
                imfs=imfs,
                residue=residue,
                characteristics=characteristics,
                auto_excluded=auto_excluded,
                sampling_rate=sr,
                parent=self,
            )

            if sel_dialog.exec() != QDialog.DialogCode.Accepted:
                logger.info("EEMD: IMF selection cancelled — signal unchanged")
                return

            exclude_imfs = sel_dialog.get_excluded_imfs()
            reconstructed = reconstruct_from_imfs(imfs, residue, exclude_imfs=exclude_imfs)

            # EEMD is order-dependent on the input state, so we apply it directly
            # rather than through the pipeline replay mechanism.
            self.pipeline.add_step("eemd_artifact_removal", {
                "ensemble_size": ensemble_size,
                "noise_width": noise_width,
                "exclude_imfs": exclude_imfs,
            })
            object.__setattr__(self.current_signal, "samples", reconstructed)
            self.single_channel_view.set_signal(self.current_signal)
            if self.current_session:
                self.single_channel_view.set_events(self.current_session.events or [])
            self._refresh_gap_overlay()
            self._refresh_processing_panel()
            n_excl  = len(exclude_imfs)
            n_total = imfs.shape[0]
            self.statusBar().showMessage(
                f"EEMD complete: excluded {n_excl}/{n_total} IMFs", 5000
            )
            logger.info(
                f"EEMD artifact removal applied: excluded IMFs {exclude_imfs}"
            )

        def on_error(msg):
            progress.close()
            self._processing_worker = None
            QMessageBox.critical(self, "EEMD Error", f"Artifact removal failed:\n{msg}")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(worker.cancel)
        worker.start()

    def _on_nk_ecg_clean(self):
        """Apply NeuroKit2 ECG-specific cleaning."""
        if self.current_signal is None:
            return

        import neurokit2 as nk

        self._ensure_raw_backup()
        try:
            cleaned = nk.ecg_clean(
                self.current_signal.samples,
                sampling_rate=int(self.current_signal.sampling_rate),
            )
            self.pipeline.add_step("ecg_clean", {})
            object.__setattr__(self.current_signal, "samples", cleaned)
            self.single_channel_view.set_signal(self.current_signal)
            if self.current_session:
                self.single_channel_view.set_events(self.current_session.events or [])
            self._refresh_gap_overlay()
            self._refresh_processing_panel()
            self.statusBar().showMessage("ECG cleaned (NeuroKit2)", 3000)
            logger.info("Applied nk.ecg_clean()")
        except Exception as e:
            logger.error(f"ECG clean failed: {e}")
            QMessageBox.critical(self, "Error", f"ECG cleaning failed:\n{e}")

    def _on_nk_ppg_clean(self):
        """Apply NeuroKit2 PPG-specific cleaning."""
        if self.current_signal is None:
            return

        import neurokit2 as nk

        self._ensure_raw_backup()
        try:
            cleaned = nk.ppg_clean(
                self.current_signal.samples,
                sampling_rate=int(self.current_signal.sampling_rate),
            )
            self.pipeline.add_step("ppg_clean", {})
            object.__setattr__(self.current_signal, "samples", cleaned)
            self.single_channel_view.set_signal(self.current_signal)
            if self.current_session:
                self.single_channel_view.set_events(self.current_session.events or [])
            self._refresh_gap_overlay()
            self._refresh_processing_panel()
            self.statusBar().showMessage("PPG cleaned (NeuroKit2)", 3000)
            logger.info("Applied nk.ppg_clean()")
        except Exception as e:
            logger.error(f"PPG clean failed: {e}")
            QMessageBox.critical(self, "Error", f"PPG cleaning failed:\n{e}")

    def _on_nk_eda_clean(self):
        """Apply NeuroKit2 EDA-specific cleaning (lowpass + smoothing)."""
        if self.current_signal is None:
            return

        import neurokit2 as nk

        self._ensure_raw_backup()
        try:
            cleaned = np.asarray(nk.eda_clean(
                self.current_signal.samples,
                sampling_rate=int(self.current_signal.sampling_rate),
            ))
            self.pipeline.add_step("eda_clean", {})
            object.__setattr__(self.current_signal, "samples", cleaned)
            self.single_channel_view.set_signal(self.current_signal)
            if self.current_session:
                self.single_channel_view.set_events(self.current_session.events or [])
            self._refresh_gap_overlay()
            self._refresh_processing_panel()
            self.statusBar().showMessage("EDA cleaned (NeuroKit2)", 3000)
            logger.info("Applied nk.eda_clean()")
        except Exception as e:
            logger.error(f"EDA clean failed: {e}")
            QMessageBox.critical(self, "Error", f"EDA cleaning failed:\n{e}")

    def _on_nk_eda_decompose(self):
        """Decompose EDA into tonic/phasic components using NeuroKit2.

        The main signal is kept unchanged. Tonic (SCL) and phasic (SCR) are
        shown as two stacked plots in the derived panel below the signal.
        SCR peak detection (nk.eda_process) always decomposes internally and
        is unaffected by which view is shown here.
        """
        if self.current_signal is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Decompose EDA")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Decompose EDA into tonic (SCL) and phasic (SCR) components.\n"
            "Both components are shown in the derived panel below the signal.\n"
            "The main signal is not modified."
        ))

        method_combo = QComboBox()
        method_combo.addItems(["highpass", "cvxEDA", "sparse"])
        layout.addRow("Decomposition method:", method_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        method = method_combo.currentText()

        import neurokit2 as nk

        try:
            sr = int(self.current_signal.sampling_rate)
            # nk.eda_phasic() decomposes already-cleaned EDA into tonic + phasic.
            components = nk.eda_phasic(
                self.current_signal.samples, sampling_rate=sr, method=method
            )
            self._eda_tonic = np.asarray(components["EDA_Tonic"])
            self._eda_phasic = np.asarray(components["EDA_Phasic"])

            # Record in pipeline for history (step returns signal unchanged)
            self.pipeline.add_step("eda_decompose", {"method": method})
            self._refresh_processing_panel()

            # Show tonic + phasic in the derived panel
            self._show_eda_components_panel()

            self.statusBar().showMessage(
                f"EDA decomposed ({method}): tonic + phasic shown below", 5000
            )
            logger.info(f"EDA decomposed: method={method}")
        except Exception as e:
            logger.error(f"EDA decompose failed: {e}")
            QMessageBox.critical(self, "Error", f"EDA decomposition failed:\n{e}")

    def _on_process_standard(self):
        """Run Standard Processing macro: crop -> resample 250 Hz -> NK clean -> detect peaks.

        Each step delegates to the same method that the individual menu action calls,
        so behaviour is identical to triggering each step manually.
        """
        if self.current_signal is None:
            return

        sig_type = self.current_signal.signal_type
        timestamps = self.current_signal.timestamps
        t_start = float(timestamps[0])
        t_end_signal = float(timestamps[-1])

        events = (self.current_session.events or []) if self.current_session else []
        has_events = bool(events)
        crop_end: float | None = None
        if has_events:
            last_event_t = max(ev.timestamp for ev in events)
            crop_end = min(last_event_t + 1.0, t_end_signal)

        # Build preview
        lines = []
        if has_events and crop_end is not None:
            lines.append(
                f"1. Crop: {t_start:.2f} s  to  {crop_end:.2f} s  "
                f"(last event {(crop_end - 1.0):.2f} s + 1 s)"
            )
        else:
            lines.append("1. Crop: SKIPPED (no events loaded)")

        current_sr = self.current_signal.sampling_rate
        if abs(current_sr - 250.0) > 0.01:
            lines.append(f"2. Resample: {current_sr:.1f} Hz -> 250 Hz")
        else:
            lines.append("2. Resample: SKIPPED (already 250 Hz)")

        clean_label = {"ecg": "nk.ecg_clean()", "ppg": "nk.ppg_clean()"}.get(
            sig_type.value.lower(), "?"
        )
        detect_label = {"ecg": "Detect R-Peaks", "ppg": "Detect Pulse Peaks"}.get(
            sig_type.value.lower(), "?"
        )
        lines.append(f"3. Clean: {clean_label}")
        lines.append(f"4. Detect peaks: {detect_label}")

        msg = QMessageBox(self)
        msg.setWindowTitle("Standard Processing")
        msg.setText("The following steps will be applied in order:\n\n" + "\n".join(lines))
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Ok)
        if msg.exec() != QMessageBox.StandardButton.Ok:
            return

        try:
            if has_events and crop_end is not None:
                self._do_crop(t_start, crop_end)

            self._do_resample(250.0)

            if sig_type == SignalType.ECG:
                self._on_nk_ecg_clean()
            else:
                self._on_nk_ppg_clean()

            self._on_process_detect_peaks()

            self.statusBar().showMessage(
                f"Standard Processing complete — {self._peak_status(self._current_peaks)}", 0
            )

        except Exception as e:
            logger.error(f"Standard Processing failed: {e}")
            QMessageBox.critical(self, "Standard Processing Error", str(e))

    def _on_process_detect_peaks(self):
        """Handle Process > Detect Peaks."""
        if self.current_signal is None:
            return

        sig_type = self.current_signal.signal_type
        samples = self.current_signal.samples
        sr = self.current_signal.sampling_rate

        self.statusBar().showMessage("Detecting peaks...", 0)

        try:
            if sig_type == SignalType.ECG:
                from cardio_signal_lab.processing.peak_detection import detect_ecg_peaks
                peak_indices = detect_ecg_peaks(samples, sr)
            elif sig_type == SignalType.PPG:
                from cardio_signal_lab.processing.peak_detection import detect_ppg_peaks
                peak_indices = detect_ppg_peaks(samples, sr)
            elif sig_type == SignalType.EDA:
                from cardio_signal_lab.processing.peak_detection import detect_eda_features
                peak_indices = detect_eda_features(samples, sr)
            else:
                QMessageBox.warning(
                    self, "Unknown Signal Type",
                    "Cannot detect peaks for UNKNOWN signal type."
                )
                return

            # Create PeakData (all auto-detected, classification=AUTO)
            classifications = np.full(len(peak_indices), PeakClassification.AUTO.value, dtype=int)
            self._current_peaks = PeakData(
                indices=peak_indices.astype(int),
                classifications=classifications,
            )

            # Initialize interactive peak editor + overlay
            self.single_channel_view.set_peaks(self._current_peaks)
            self.signals.peaks_updated.emit(self._current_peaks)
            self._build_menus()  # Show Heart Rate action now that peaks exist
            self.statusBar().showMessage(
                f"{sig_type.value.upper()} — {self._peak_status(self._current_peaks)}", 0
            )

        except Exception as e:
            logger.error(f"Peak detection failed: {e}")
            QMessageBox.critical(self, "Error", f"Peak detection failed:\n{e}")
            self.statusBar().clearMessage()

    def _do_resample(self, target_sr: float):
        """Resample current signal to target_sr Hz. No dialog — call from handlers.

        Uses pandas resample to bin samples onto a regular target-rate grid.
        Bins spanning minor timing jitter (<=2 bins) are interpolated; bins
        inside actual timestamp gaps remain NaN and are dropped, preserving the
        gap structure of the original signal.
        """
        if self.current_signal is None:
            return

        import pandas as pd
        from cardio_signal_lab.core.data_models import ProcessingStep

        current_sr = self.current_signal.sampling_rate
        if abs(current_sr - target_sr) < 0.01:
            return

        source = self._raw_samples if self._raw_samples is not None else self.current_signal.samples
        source_timestamps = self.current_signal.timestamps

        t_start = float(source_timestamps[0])
        period_ms = 1000.0 / target_sr
        period_str = f"{period_ms}ms"

        td_index = pd.to_timedelta(source_timestamps - t_start, unit="s")
        series = pd.Series(source, index=td_index)
        resampled_series = series.resample(period_str).mean()
        resampled_series = resampled_series.interpolate(method="linear", limit=2)
        resampled_series = resampled_series.dropna()

        resampled_raw = resampled_series.values.astype(np.float64)
        new_timestamps = t_start + resampled_series.index.total_seconds().values
        new_n = len(resampled_raw)

        if self._current_peaks and self._current_peaks.num_peaks > 0:
            peak_timestamps = source_timestamps[
                np.clip(self._current_peaks.indices, 0, len(source_timestamps) - 1)
            ]
            new_indices = np.searchsorted(new_timestamps, peak_timestamps)
            new_indices = np.clip(new_indices, 0, new_n - 1)
            self._current_peaks = PeakData(
                indices=new_indices,
                classifications=self._current_peaks.classifications.copy(),
            )

        object.__setattr__(self.current_signal, "samples", resampled_raw.copy())
        object.__setattr__(self.current_signal, "timestamps", new_timestamps)
        object.__setattr__(self.current_signal, "sampling_rate", float(target_sr))
        self._raw_samples = resampled_raw.copy()
        self._eda_tonic = None
        self._eda_phasic = None
        self.pipeline.reset()

        # Detect gaps in the resampled signal and store for the Display Gaps toggle.
        # The pandas dropna preserves gap structure as timestamp jumps, so standard
        # gap detection finds them. The limit=2 interpolation fills only tiny jitter
        # bins, which are below the gap_multiplier threshold and won't be flagged.
        from cardio_signal_lab.processing.bad_segment_detection import detect_timestamp_gaps
        gap_pairs = detect_timestamp_gaps(new_timestamps, target_sr, gap_multiplier=2.0)
        self._current_gap_segments = [
            BadSegment(start_idx=s, end_idx=e, source="gap")
            for s, e in gap_pairs
        ]
        if self._show_gap_segments and self._current_gap_segments:
            self.single_channel_view.set_gap_segments(
                self._current_gap_segments, self.current_signal
            )
        else:
            self.single_channel_view.clear_gap_segments()

        self._structural_ops.append(ProcessingStep(
            operation="resample",
            parameters={
                "original_sr": current_sr,
                "target_sr": target_sr,
                "n_samples": new_n,
            },
        ))
        self._refresh_processing_panel()

        self.single_channel_view.set_signal(self.current_signal)
        if self._current_peaks is not None:
            self.single_channel_view.set_peaks(self._current_peaks)
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        logger.info(f"Signal resampled from {current_sr:.1f} to {target_sr:.1f} Hz ({new_n} samples)")

    def _on_process_resample(self):
        """Handle Process > Resample — show dialog then delegate to _do_resample."""
        if self.current_signal is None:
            return

        current_sr = self.current_signal.sampling_rate
        n_samples = len(self.current_signal.samples)

        dialog = QDialog(self)
        dialog.setWindowTitle("Resample Signal")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(f"Current: {current_sr:.1f} Hz  ({n_samples} samples)"))

        if self._current_peaks and self._current_peaks.num_peaks > 0:
            layout.addRow(QLabel(
                f"Note: {self._current_peaks.num_peaks} peak(s) will be rescaled.\n"
                "Verify positions after resampling."
            ))

        target_spin = QDoubleSpinBox()
        target_spin.setDecimals(1)
        target_spin.setRange(1.0, 10000.0)
        target_spin.setValue(current_sr)
        target_spin.setSuffix(" Hz")
        layout.addRow("Target rate:", target_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        target_sr = target_spin.value()
        if abs(target_sr - current_sr) < 0.01:
            return

        self._do_resample(target_sr)
        new_n = len(self.current_signal.samples)
        self.statusBar().showMessage(
            f"Resampled: {current_sr:.1f} Hz -> {target_sr:.1f} Hz  ({new_n} samples)", 5000
        )

    def _do_crop(self, crop_start: float, crop_end: float):
        """Crop current signal to [crop_start, crop_end] seconds. No dialog — call from handlers."""
        if self.current_signal is None:
            return

        from cardio_signal_lab.core.data_models import ProcessingStep

        timestamps = self.current_signal.timestamps
        start_idx = int(np.searchsorted(timestamps, crop_start, side="left"))
        end_idx = int(np.searchsorted(timestamps, crop_end, side="right"))

        source = self._raw_samples if self._raw_samples is not None else self.current_signal.samples
        cropped_raw = source[start_idx:end_idx]
        cropped_timestamps = timestamps[start_idx:end_idx]

        if self._current_peaks and self._current_peaks.num_peaks > 0:
            mask = (
                (self._current_peaks.indices >= start_idx)
                & (self._current_peaks.indices < end_idx)
            )
            kept_indices = self._current_peaks.indices[mask] - start_idx
            kept_cls = self._current_peaks.classifications[mask]
            if len(kept_indices) >= 1:
                self._current_peaks = PeakData(indices=kept_indices, classifications=kept_cls)
                logger.info(f"Crop: kept {len(kept_indices)} peak(s)")
            else:
                self._current_peaks = None
                logger.info("Crop: all peaks fell outside crop range — cleared")

        object.__setattr__(self.current_signal, "samples", cropped_raw.copy())
        object.__setattr__(self.current_signal, "timestamps", cropped_timestamps)
        self._raw_samples = cropped_raw.copy()
        self._eda_tonic = None
        self._eda_phasic = None
        self.pipeline.reset()

        self._current_gap_segments = []
        self._show_gap_segments = False
        self.single_channel_view.clear_gap_segments()

        self._structural_ops.append(ProcessingStep(
            operation="crop",
            parameters={
                "start": crop_start,
                "end": crop_end,
                "n_samples": len(cropped_raw),
            },
        ))
        self._refresh_processing_panel()

        self.single_channel_view.set_signal(self.current_signal)
        if self._current_peaks is not None:
            self.single_channel_view.set_peaks(self._current_peaks)
        else:
            self.single_channel_view.clear_peaks()
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        logger.info(
            f"Signal cropped: [{crop_start:.2f}, {crop_end:.2f}] s  "
            f"({len(cropped_raw)} samples)"
        )

    def _on_process_crop(self):
        """Handle Process > Crop — show dialog then delegate to _do_crop."""
        if self.current_signal is None:
            return

        timestamps = self.current_signal.timestamps
        t_start = float(timestamps[0])
        t_end = float(timestamps[-1])

        events = (self.current_session.events or []) if self.current_session else []
        # Only keep events within the signal range
        events_in_range = [ev for ev in events if t_start <= ev.timestamp <= t_end]

        dialog = QDialog(self)
        dialog.setWindowTitle("Crop Signal")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(f"Signal range: {t_start:.3f} s  to  {t_end:.3f} s"))

        # Event quick-select combos (populate only when events are available)
        _MANUAL = "(manual)"
        event_labels = [_MANUAL] + [
            f"{ev.label}  @ {ev.timestamp:.3f} s" for ev in events_in_range
        ]

        from_event_combo = QComboBox()
        from_event_combo.addItems(event_labels)
        if events_in_range:
            layout.addRow("Start from event:", from_event_combo)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(t_start, t_end)
        start_spin.setValue(t_start)
        start_spin.setSuffix(" s")
        start_spin.setSingleStep(1.0)
        layout.addRow("Crop start:", start_spin)

        to_event_combo = QComboBox()
        to_event_combo.addItems(event_labels)
        if events_in_range:
            layout.addRow("End at event:", to_event_combo)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(t_start, t_end)
        end_spin.setValue(t_end)
        end_spin.setSuffix(" s")
        end_spin.setSingleStep(1.0)
        layout.addRow("Crop end:", end_spin)

        # Wire event combos to fill the spinboxes
        def _on_from_event(idx):
            if idx > 0:
                start_spin.setValue(events_in_range[idx - 1].timestamp)

        def _on_to_event(idx):
            if idx > 0:
                end_spin.setValue(events_in_range[idx - 1].timestamp)

        from_event_combo.currentIndexChanged.connect(_on_from_event)
        to_event_combo.currentIndexChanged.connect(_on_to_event)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        crop_start = start_spin.value()
        crop_end = end_spin.value()

        if crop_start >= crop_end:
            QMessageBox.warning(self, "Invalid Range", "Crop start must be before crop end.")
            return

        start_idx = int(np.searchsorted(timestamps, crop_start, side="left"))
        end_idx = int(np.searchsorted(timestamps, crop_end, side="right"))

        if end_idx - start_idx < 2:
            QMessageBox.warning(self, "Too Short", "Crop range contains fewer than 2 samples.")
            return

        self._do_crop(crop_start, crop_end)

        cropped_timestamps = self.current_signal.timestamps
        duration = float(cropped_timestamps[-1] - cropped_timestamps[0])
        self.statusBar().showMessage(
            f"Cropped to {crop_start:.2f}s - {crop_end:.2f}s  "
            f"({duration:.2f}s, {len(cropped_timestamps)} samples)", 5000
        )

    def _on_detect_timestamp_gaps(self):
        """Handle Process > Detect Timestamp Gaps — mark missing data in timestamp sequence."""
        if self.current_signal is None:
            return

        from cardio_signal_lab.processing.bad_segment_detection import detect_timestamp_gaps

        dialog = QDialog(self)
        dialog.setWindowTitle("Detect Timestamp Gaps")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Finds places where the timestamp sequence jumps by more than\n"
            "the expected sample interval, indicating missing data.\n"
            "Gaps are shown as red segments on the signal trace."
        ))

        gap_spin = QDoubleSpinBox()
        gap_spin.setDecimals(1)
        gap_spin.setRange(1.1, 20.0)
        gap_spin.setSingleStep(0.5)
        gap_spin.setValue(2.0)
        gap_spin.setToolTip("Minimum gap size as a multiple of the expected sample interval (default 2x).")
        layout.addRow("Gap threshold (x interval):", gap_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        gap_index_pairs = detect_timestamp_gaps(
            self.current_signal.timestamps,
            self.current_signal.sampling_rate,
            gap_multiplier=gap_spin.value(),
        )

        self._current_gap_segments = [
            BadSegment(start_idx=s, end_idx=e, source="gap")
            for s, e in gap_index_pairs
        ]
        self.single_channel_view.set_gap_segments(self._current_gap_segments, self.current_signal)

        msg = f"Found {len(self._current_gap_segments)} timestamp gap(s)"
        self.statusBar().showMessage(msg, 5000)
        logger.info(msg)
        self._build_menus()

    def _on_clear_gap_segments(self):
        """Handle Process > Clear Timestamp Gaps."""
        self._current_gap_segments = []
        self.single_channel_view.clear_gap_segments()
        self.statusBar().showMessage("Timestamp gaps cleared", 3000)
        self._build_menus()

    def _on_detect_bad_segments(self):
        """Handle Process > Detect Bad Segments - auto-detect amplitude artifacts."""
        if self.current_signal is None:
            return

        from cardio_signal_lab.processing.bad_segment_detection import detect_amplitude_artifacts

        dialog = QDialog(self)
        dialog.setWindowTitle("Detect Bad Segments")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Detects amplitude transients using a rolling MAD threshold.\n"
            "Results are shown as amber shaded regions and can be repaired\n"
            "with Interpolate Bad Segments."
        ))

        mad_spin = QDoubleSpinBox()
        mad_spin.setDecimals(1)
        mad_spin.setRange(1.0, 20.0)
        mad_spin.setSingleStep(0.5)
        mad_spin.setValue(4.0)
        mad_spin.setToolTip("Threshold in multiples of rolling MAD. Lower = more sensitive.")
        layout.addRow("MAD threshold:", mad_spin)

        window_spin = QDoubleSpinBox()
        window_spin.setDecimals(1)
        window_spin.setRange(1.0, 60.0)
        window_spin.setSingleStep(1.0)
        window_spin.setValue(10.0)
        window_spin.setToolTip("Rolling window duration in seconds for baseline estimation.")
        layout.addRow("Window (s):", window_spin)

        dilation_spin = QDoubleSpinBox()
        dilation_spin.setDecimals(2)
        dilation_spin.setRange(0.0, 2.0)
        dilation_spin.setSingleStep(0.05)
        dilation_spin.setValue(0.3)
        dilation_spin.setToolTip("Padding added to each side of detected regions (captures edge ringing).")
        layout.addRow("Dilation (s):", dilation_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        amp_segs = detect_amplitude_artifacts(
            self.current_signal.samples,
            self.current_signal.sampling_rate,
            mad_threshold=mad_spin.value(),
            window_s=window_spin.value(),
            dilation_s=dilation_spin.value(),
        )

        bad_segs = [
            BadSegment(start_idx=s, end_idx=e, source="amplitude")
            for s, e in amp_segs
        ]

        self._current_bad_segments = bad_segs
        self.single_channel_view.set_bad_segments(bad_segs, self.current_signal)

        msg = f"Found {len(bad_segs)} bad segment(s) (amplitude)"
        self.statusBar().showMessage(msg, 5000)
        logger.info(msg)

        # Rebuild menu so Interpolate/Clear actions are enabled
        self._build_menus()

    def _on_mark_bad_segment(self):
        """Handle Process > Mark Bad Segment (Manual) - user-specified time range."""
        if self.current_signal is None:
            return

        sig = self.current_signal
        t_start_sig = float(sig.timestamps[0])
        t_end_sig = float(sig.timestamps[-1])

        dialog = QDialog(self)
        dialog.setWindowTitle("Mark Bad Segment")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Enter the time range to mark as a bad segment.\n"
            "The region will be added to the current bad segment list."
        ))

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(t_start_sig, t_end_sig)
        start_spin.setValue(t_start_sig)
        layout.addRow("Start time (s):", start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(t_start_sig, t_end_sig)
        end_spin.setValue(min(t_start_sig + 1.0, t_end_sig))
        layout.addRow("End time (s):", end_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        t0 = start_spin.value()
        t1 = end_spin.value()
        if t1 <= t0:
            QMessageBox.warning(self, "Invalid Range", "End time must be greater than start time.")
            return

        timestamps = sig.timestamps
        start_idx = int(np.searchsorted(timestamps, t0))
        end_idx = int(np.searchsorted(timestamps, t1, side="right")) - 1
        start_idx = max(0, min(start_idx, len(timestamps) - 1))
        end_idx = max(start_idx, min(end_idx, len(timestamps) - 1))

        new_seg = BadSegment(start_idx=start_idx, end_idx=end_idx, source="manual")
        self._current_bad_segments = list(self._current_bad_segments) + [new_seg]
        self.single_channel_view.set_bad_segments(self._current_bad_segments, sig)

        logger.info(
            f"Marked bad segment: [{t0:.3f}, {t1:.3f}]s "
            f"(samples {start_idx}-{end_idx})"
        )
        self.statusBar().showMessage(
            f"Marked bad segment: {t0:.3f}-{t1:.3f}s ({end_idx - start_idx + 1} samples)",
            4000,
        )
        self._build_menus()

    def _on_interpolate_bad_segments(self):
        """Handle Process > Interpolate Bad Segments - cubic spline repair."""
        if self.current_signal is None or not self._current_bad_segments:
            self.statusBar().showMessage("No bad segments to interpolate", 3000)
            return

        from cardio_signal_lab.processing.bad_segment_detection import interpolate_bad_segments

        self._ensure_raw_backup()

        # Serialize segment list as JSON-compatible pairs for pipeline storage
        segments_param = [
            [seg.start_idx, seg.end_idx]
            for seg in self._current_bad_segments
        ]

        self.pipeline.add_step("interpolate_bad_segments", {
            "segments": segments_param,
            "anchor_s": 2.0,
        })

        self._apply_pipeline_and_update()

        # Segments have been repaired — save for overlay toggle, then clear active overlay
        self._interpolated_bad_segments = list(self._current_bad_segments)
        self._current_bad_segments = []
        self.single_channel_view.clear_bad_segments()

        # If the toggle is on, show the saved regions immediately
        if self._show_interpolated_regions:
            self.single_channel_view.set_interpolated_segments(
                self._interpolated_bad_segments, self.current_signal
            )

        n = len(segments_param)
        self.statusBar().showMessage(
            f"Interpolated {n} bad segment(s) (PCHIP, anchor=2s)", 5000
        )
        logger.info(f"Interpolated {n} bad segment(s)")
        self._build_menus()

    def _on_clear_bad_segments(self):
        """Handle Process > Clear Bad Segments - discard current detection without repairing."""
        self._current_bad_segments = []
        self.single_channel_view.clear_bad_segments()
        self.statusBar().showMessage("Bad segments cleared", 3000)
        self._build_menus()

    def _on_process_reset(self):
        """Handle Process > Reset Processing - revert to the original file-loaded signal.

        Restores samples, timestamps, and sampling rate to their state at file load,
        undoing all structural ops (crop, resample) and pipeline filter steps.
        """
        has_anything = (
            self._original_samples is not None
            or self._raw_samples is not None
            or self._structural_ops
            or self.pipeline.steps
        )
        if not has_anything:
            self.statusBar().showMessage("No processing to reset", 3000)
            return

        if self.current_signal is None:
            return

        if self._original_samples is not None:
            object.__setattr__(self.current_signal, "samples", self._original_samples.copy())
            object.__setattr__(self.current_signal, "timestamps", self._original_timestamps.copy())
            object.__setattr__(self.current_signal, "sampling_rate", self._original_sampling_rate)
            self._raw_samples = self._original_samples.copy()
        elif self._raw_samples is not None:
            # Fallback for channels loaded before this feature (e.g. restored sessions)
            object.__setattr__(self.current_signal, "samples", self._raw_samples.copy())

        self.pipeline.reset()
        self._structural_ops.clear()
        self._current_peaks = None
        self._eda_tonic = None
        self._eda_phasic = None
        self._current_bad_segments = []
        self._current_gap_segments = []
        self._show_gap_segments = False
        self._interpolated_bad_segments = []
        self._show_interpolated_regions = False
        self._mark_dirty()

        self.single_channel_view.set_signal(self.current_signal)
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        self.single_channel_view.clear_peaks()
        self.single_channel_view.clear_derived()
        self.single_channel_view.clear_bad_segments()
        self.single_channel_view.clear_gap_segments()
        self.single_channel_view.clear_interpolated_segments()
        self.processing_panel.clear()
        self.statusBar().showMessage("Processing reset to original signal", 3000)
        logger.info("Processing reset to original file-loaded signal")

    def _reset_all_channel_state(self):
        """Reset ALL per-channel processing state. Call before loading a new file."""
        self.current_signal = None
        self._current_peaks = None
        self._raw_samples = None
        self._eda_tonic = None
        self._eda_phasic = None
        self._current_bad_segments = []
        self._current_gap_segments = []
        self._show_gap_segments = False
        self._interpolated_bad_segments = []
        self._show_interpolated_regions = False
        self._structural_ops = []
        self._channel_state = {}
        self._derived_channel_specs = []
        self._original_samples = None
        self._original_timestamps = None
        self._original_sampling_rate = None
        self.pipeline.reset()

    # ---- View Operations ----

    def _get_current_view(self):
        """Get the currently active view widget."""
        if self.current_view_level == "multi":
            return self.multi_signal_view
        elif self.current_view_level == "type":
            return self.signal_type_view
        else:
            return self.single_channel_view

    def _on_view_zoom_in(self):
        self._get_current_view().zoom_in()

    def _on_view_zoom_out(self):
        self._get_current_view().zoom_out()

    def _on_view_reset(self):
        self._get_current_view().reset_view()

    def _on_view_fit(self):
        self._on_view_reset()

    def _on_view_jump_to_start(self):
        self._get_current_view().jump_to_start()

    def _on_view_jump_to_end(self):
        self._get_current_view().jump_to_end()

    def _on_view_jump_to_time(self):
        """Handle View > Jump to Time."""
        view = self._get_current_view()

        # Get plot bounds
        if self.current_view_level == "channel":
            plot = self.single_channel_view.plot_widget
        elif hasattr(view, 'plot_widgets') and view.plot_widgets:
            plot = view.plot_widgets[0]
        else:
            QMessageBox.warning(self, "No Signal", "No signal loaded.")
            return

        if plot.lod_renderer is None:
            QMessageBox.warning(self, "No Signal", "No signal loaded.")
            return

        x_min, x_max, _, _ = plot.lod_renderer.get_full_range()
        time = self._show_jump_to_time_dialog(x_min, x_max)
        if time is not None:
            view.jump_to_time(time)

    def _show_jump_to_time_dialog(self, min_time: float, max_time: float) -> float | None:
        """Show dialog to input jump time."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Jump to Time")
        layout = QFormLayout(dialog)

        time_spinbox = QDoubleSpinBox()
        time_spinbox.setDecimals(2)
        time_spinbox.setRange(min_time, max_time)
        time_spinbox.setValue(min_time)
        time_spinbox.setSuffix(" s")
        time_spinbox.setSingleStep(1.0)
        layout.addRow("Jump to time:", time_spinbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return time_spinbox.value()
        return None

    def _on_view_zoom_to_range(self):
        """Handle View > Zoom to Time Range."""
        view = self._get_current_view()

        if self.current_view_level == "channel":
            plot = self.single_channel_view.plot_widget
        elif hasattr(view, 'plot_widgets') and view.plot_widgets:
            plot = view.plot_widgets[0]
        else:
            QMessageBox.warning(self, "No Signal", "No signal loaded.")
            return

        if plot.lod_renderer is None:
            QMessageBox.warning(self, "No Signal", "No signal loaded.")
            return

        x_min, x_max, _, _ = plot.lod_renderer.get_full_range()
        current_x_min, current_x_max, _, _ = plot.get_visible_range()

        range_tuple = self._show_zoom_to_range_dialog(x_min, x_max, current_x_min, current_x_max)
        if range_tuple is None:
            return

        start_time, end_time = range_tuple
        view_box = plot.plotItem.getViewBox()
        view_box.setXRange(start_time, end_time, padding=0)
        logger.info(f"Zoomed to range [{start_time:.2f}, {end_time:.2f}]")

    def _show_zoom_to_range_dialog(self, min_time: float, max_time: float,
                                     current_start: float, current_end: float) -> tuple[float, float] | None:
        """Show dialog to input zoom time range."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Zoom to Time Range")
        layout = QFormLayout(dialog)

        start_spinbox = QDoubleSpinBox()
        start_spinbox.setDecimals(2)
        start_spinbox.setRange(min_time, max_time)
        start_spinbox.setValue(current_start)
        start_spinbox.setSuffix(" s")
        start_spinbox.setSingleStep(1.0)

        end_spinbox = QDoubleSpinBox()
        end_spinbox.setDecimals(2)
        end_spinbox.setRange(min_time, max_time)
        end_spinbox.setValue(current_end)
        end_spinbox.setSuffix(" s")
        end_spinbox.setSingleStep(1.0)

        layout.addRow("Start time:", start_spinbox)
        layout.addRow("End time:", end_spinbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            start = start_spinbox.value()
            end = end_spinbox.value()
            if start >= end:
                QMessageBox.warning(self, "Invalid Range", "Start time must be less than end time.")
                return None
            return (start, end)
        return None

    def _on_view_pan_mode(self):
        """Set mouse drag to pan on current view."""
        view = self._get_current_view()
        if self.current_view_level == "channel":
            vb = self.single_channel_view.plot_widget.plotItem.getViewBox()
            vb.setMouseMode(vb.PanMode)
        elif hasattr(view, 'plot_widgets'):
            for pw in view.plot_widgets:
                vb = pw.plotItem.getViewBox()
                vb.setMouseMode(vb.PanMode)
        self.statusBar().showMessage("Pan Mode: Drag to pan, wheel to zoom", 3000)

    def _on_view_zoom_mode(self):
        """Set mouse drag to zoom rectangle on current view."""
        view = self._get_current_view()
        if self.current_view_level == "channel":
            vb = self.single_channel_view.plot_widget.plotItem.getViewBox()
            vb.setMouseMode(vb.RectMode)
        elif hasattr(view, 'plot_widgets'):
            for pw in view.plot_widgets:
                vb = pw.plotItem.getViewBox()
                vb.setMouseMode(vb.RectMode)
        self.statusBar().showMessage("Zoom Mode: Drag rectangle to zoom", 5000)

    def _on_view_toggle_events(self):
        """Toggle event overlay visibility."""
        view = self._get_current_view()
        view.toggle_events()
        visible = view.are_events_visible()
        status = "visible" if visible else "hidden"
        self.statusBar().showMessage(f"Event markers {status}", 3000)

    def _on_view_toggle_interpolated_regions(self, checked: bool):
        """Toggle the blue overlay showing previously-interpolated bad segment locations."""
        self._show_interpolated_regions = checked
        if checked and self._interpolated_bad_segments:
            self.single_channel_view.set_interpolated_segments(
                self._interpolated_bad_segments, self.current_signal
            )
            n = len(self._interpolated_bad_segments)
            self.statusBar().showMessage(f"Showing {n} interpolated region(s)", 3000)
        else:
            self.single_channel_view.clear_interpolated_segments()
            self.statusBar().showMessage("Interpolated regions hidden", 3000)

    def _on_view_toggle_gap_segments(self, checked: bool):
        """Toggle the red trace overlay showing timestamp gap locations."""
        self._show_gap_segments = checked
        if checked and self._current_gap_segments:
            self.single_channel_view.set_gap_segments(
                self._current_gap_segments, self.current_signal
            )
            n = len(self._current_gap_segments)
            self.statusBar().showMessage(f"Showing {n} gap location(s)", 3000)
        else:
            self.single_channel_view.clear_gap_segments()
            self.statusBar().showMessage("Gap segments hidden", 3000)

    def _on_view_toggle_log(self):
        """Toggle log panel visibility."""
        if self.log_panel.isVisible():
            self.log_panel.hide()
            self.statusBar().showMessage("Log panel hidden", 3000)
        else:
            self.log_panel.show()
            self.statusBar().showMessage("Log panel visible", 3000)

    def _on_view_toggle_processing(self):
        """Toggle processing steps panel visibility."""
        if self.processing_panel.isVisible():
            self.processing_panel.hide()
            self.statusBar().showMessage("Processing panel hidden", 3000)
        else:
            self.processing_panel.show()
            self.statusBar().showMessage("Processing panel visible", 3000)

    def _on_view_toggle_heart_rate(self):
        """Toggle heart rate panel for ECG/PPG signals."""
        if self.single_channel_view.is_derived_visible():
            self.single_channel_view.clear_derived()
            self.statusBar().showMessage("Heart rate panel hidden", 3000)
        elif self._current_peaks is None or self._current_peaks.num_peaks < 2:
            self.statusBar().showMessage(
                "Detect peaks first (Process > NeuroKit2 > Detect R-Peaks / Detect Pulse Peaks)", 5000
            )
        else:
            self._refresh_heart_rate_panel()

    def _on_view_toggle_eda_components(self):
        """Toggle EDA tonic/phasic panel."""
        if self.single_channel_view.is_derived_visible():
            self.single_channel_view.clear_derived()
            self.statusBar().showMessage("EDA components panel hidden", 3000)
        else:
            self._show_eda_components_panel()

    def _refresh_heart_rate_panel(self):
        """Compute and display (or update) the heart rate panel."""
        if self.current_signal is None or self._current_peaks is None:
            return
        from cardio_signal_lab.processing.derived import compute_heart_rate
        times, bpm, rolling_bpm = compute_heart_rate(self.current_signal, self._current_peaks)
        sig_type = self.current_signal.signal_type.value  # "ecg" or "ppg"
        self.single_channel_view.update_heart_rate(times, bpm, rolling_bpm, sig_type)
        if len(bpm) > 0:
            self.statusBar().showMessage(
                f"Heart rate: mean {bpm.mean():.1f} bpm, {len(bpm)} intervals", 5000
            )

    def _show_eda_components_panel(self):
        """Display the EDA tonic/phasic derived panel."""
        if (
            self.current_signal is None
            or self._eda_tonic is None
            or self._eda_phasic is None
        ):
            return
        self.single_channel_view.show_eda_components(
            self.current_signal.timestamps,
            self._eda_tonic,
            self._eda_phasic,
        )
        self.statusBar().showMessage("EDA components: tonic (SCL) and phasic (SCR)", 5000)

    def _on_view_heartbeat_overlay(self):
        """Open the heartbeat overlay plot dialog (ECG/PPG with peaks)."""
        if self.current_signal is None or self._current_peaks is None:
            return
        from cardio_signal_lab.gui.analysis_plots import HeartbeatOverlayDialog
        dlg = HeartbeatOverlayDialog(
            self.current_signal, self._current_peaks, parent=self
        )
        dlg.exec()

    def _on_view_rr_histogram(self):
        """Open the RR interval histogram dialog (ECG/PPG with peaks)."""
        if self.current_signal is None or self._current_peaks is None:
            return
        from cardio_signal_lab.gui.analysis_plots import RRHistogramDialog
        dlg = RRHistogramDialog(
            self.current_signal, self._current_peaks, parent=self
        )
        dlg.exec()

    # ---- Type-View Operations ----

    def _on_type_crop_all(self):
        """Crop all channels in the current session to a shared time range."""
        if self.current_session is None or not self.current_session.signals:
            return

        signals = self.current_session.signals

        # Common time range: the intersection of all channel timestamp ranges
        t_start = max(float(s.timestamps[0]) for s in signals)
        t_end = min(float(s.timestamps[-1]) for s in signals)

        if t_start >= t_end:
            QMessageBox.warning(self, "No Overlap", "The channels have no common time range.")
            return

        events = (self.current_session.events or []) if self.current_session else []
        events_in_range = [ev for ev in events if t_start <= ev.timestamp <= t_end]

        dialog = QDialog(self)
        dialog.setWindowTitle("Crop All Channels")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            f"Crops all {len(signals)} channel(s) to the selected range.\n"
            f"Common range: {t_start:.3f} s  to  {t_end:.3f} s"
        ))

        _MANUAL = "(manual)"
        event_labels = [_MANUAL] + [
            f"{ev.label}  @ {ev.timestamp:.3f} s" for ev in events_in_range
        ]

        from_combo = QComboBox()
        from_combo.addItems(event_labels)
        if events_in_range:
            layout.addRow("Start from event:", from_combo)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(t_start, t_end)
        start_spin.setValue(t_start)
        start_spin.setSuffix(" s")
        start_spin.setSingleStep(1.0)
        layout.addRow("Crop start:", start_spin)

        to_combo = QComboBox()
        to_combo.addItems(event_labels)
        if events_in_range:
            layout.addRow("End at event:", to_combo)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(t_start, t_end)
        end_spin.setValue(t_end)
        end_spin.setSuffix(" s")
        end_spin.setSingleStep(1.0)
        layout.addRow("Crop end:", end_spin)

        def _on_from_event(idx):
            if idx > 0:
                start_spin.setValue(events_in_range[idx - 1].timestamp)

        def _on_to_event(idx):
            if idx > 0:
                end_spin.setValue(events_in_range[idx - 1].timestamp)

        from_combo.currentIndexChanged.connect(_on_from_event)
        to_combo.currentIndexChanged.connect(_on_to_event)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        crop_start = start_spin.value()
        crop_end = end_spin.value()

        if crop_start >= crop_end:
            QMessageBox.warning(self, "Invalid Range", "Crop start must be before crop end.")
            return

        n_cropped = 0
        for signal in signals:
            ts = signal.timestamps
            start_idx = int(np.searchsorted(ts, crop_start, side="left"))
            end_idx = int(np.searchsorted(ts, crop_end, side="right"))
            if end_idx - start_idx < 2:
                continue
            object.__setattr__(signal, "samples", signal.samples[start_idx:end_idx].copy())
            object.__setattr__(signal, "timestamps", ts[start_idx:end_idx])
            n_cropped += 1

        # All per-channel state (peaks, bad segments, pipeline) is now invalid
        self._channel_state.clear()
        self.current_signal = None

        # Refresh the type view with the cropped signals
        if self.current_signal_type is not None:
            self.signal_type_view.set_signal_type(
                self.current_signal_type, self.signal_type_view.signals
            )
        self._mark_dirty()
        self.statusBar().showMessage(
            f"Cropped {n_cropped} channel(s) to {crop_start:.2f}s – {crop_end:.2f}s", 5000
        )
        logger.info(
            f"Crop all: {n_cropped} channel(s) -> [{crop_start:.2f}, {crop_end:.2f}] s"
        )

    def _on_type_create_l2_norm(self):
        """Handle Process > Create L2 Norm Channel from the signal-type view.

        Shows a channel selection dialog so the user can choose which channels
        to include in the norm computation.
        """
        signals = self.signal_type_view.signals
        if len(signals) < 2:
            return

        from PySide6.QtWidgets import QListWidget, QListWidgetItem

        dialog = QDialog(self)
        dialog.setWindowTitle("Create L2 Norm Channel")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Select the channels to include in the L2 Norm.\n"
            "L2 Norm = sqrt(sum of squares across selected channels)."
        ))

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for signal in signals:
            item = QListWidgetItem(signal.channel_name)
            item.setSelected(True)  # Pre-select all
            list_widget.addItem(item)
        layout.addRow("Channels:", list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_indices = [list_widget.row(item) for item in list_widget.selectedItems()]
        if len(selected_indices) < 2:
            QMessageBox.warning(self, "Selection", "Select at least 2 channels for L2 Norm.")
            return

        selected_signals = [signals[i] for i in selected_indices]

        try:
            self.signal_type_view.add_l2_norm(selected_signals)
            self._derived_channel_specs.append({
                "type": "l2_norm",
                "signal_type": self.current_signal_type.value,
                "source_channels": [s.channel_name for s in selected_signals],
            })
            self._mark_dirty()
            self._build_menus()  # Refresh Select menu to include derived channel
            self.statusBar().showMessage(
                f"L2 Norm channel created from {len(selected_signals)} channels", 5000
            )
        except Exception as e:
            logger.error(f"L2 Norm creation failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create L2 Norm:\n{e}")

    # ---- Help Operations ----

    def _on_help_shortcuts(self):
        logger.info("Help > Keyboard Shortcuts triggered")
        QMessageBox.information(self, "Keyboard Shortcuts", get_help_text())

    def _on_help_about(self):
        logger.info("Help > About triggered")
        QMessageBox.about(
            self,
            "About CardioSignalLab",
            (
                "<h3>CardioSignalLab</h3>"
                f"<p>Version {__version__}</p>"
                "<p>Viewing, processing, and correcting physiological "
                "signals (ECG, PPG, EDA).</p>"
            ),
        )
