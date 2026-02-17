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

from cardio_signal_lab.config import get_config, get_keysequence
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

        # Processing state (active channel)
        self.pipeline = ProcessingPipeline()
        self._raw_samples: np.ndarray | None = None  # Baseline signal for pipeline replay
        self._current_peaks: PeakData | None = None
        self._current_bad_segments: list[BadSegment] = []
        self._processing_worker: ProcessingWorker | None = None
        # Structural ops (crop / resample) that have been applied directly to the
        # raw signal and cannot be replayed or reverted by the pipeline.
        self._structural_ops: list = []

        # Derived visualization state (active channel)
        self._eda_tonic: np.ndarray | None = None   # Tonic (SCL) component from eda_process()
        self._eda_phasic: np.ndarray | None = None  # Phasic (SCR) component from eda_process()

        # Per-channel state store: (SignalType, channel_name) -> state dict
        # Allows peaks and processing to survive navigation between channels.
        self._channel_state: dict[tuple, dict] = {}

        # Window setup
        self.setWindowTitle("CardioSignalLab")
        self.resize(self.config.gui.window_width, self.config.gui.window_height)

        # Create views
        self.multi_signal_view = MultiSignalView()
        self.signal_type_view = SignalTypeView()
        self.single_channel_view = SingleChannelView()

        # Create stacked widget
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.multi_signal_view)   # Index 0
        self.stacked_widget.addWidget(self.signal_type_view)    # Index 1
        self.stacked_widget.addWidget(self.single_channel_view) # Index 2
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentWidget(self.multi_signal_view)

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

        # Connect view signals
        self.multi_signal_view.signal_type_selected.connect(self._on_signal_type_selected)
        self.signal_type_view.channel_selected.connect(self._on_channel_selected)
        self.single_channel_view.return_to_multi_requested.connect(self._on_return_to_multi)
        self.single_channel_view.peaks_changed.connect(self._on_peaks_changed)

        # Connect app signals
        self.signals.mode_changed.connect(self._on_mode_changed)
        self.signals.file_loaded.connect(self._on_file_loaded_signal)

        logger.info("MainWindow initialized with 3-level view hierarchy")

    # ---- Menu Building ----

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

        # Process menu: only show when >=2 channels (L2 Norm requires multiple channels)
        if len(self.signal_type_view.signals) >= 2:
            process_menu = self.menuBar().addMenu("&Process")
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

        save_action = QAction("&Save", self)
        save_action.setShortcut(get_keysequence("file_save"))
        save_action.triggered.connect(self._on_file_save)
        menu.addAction(save_action)

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
                    action = QAction(signal.channel_name, self)
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
        """Add Process menu actions, adapted to the current signal type.

        Menu structure varies by signal type:
        - ECG: Filters (Bandpass, Notch, Baseline, Zero-Ref) | NeuroKit2 > (Detect R-Peaks)
        - PPG: Filters | EEMD Artifact Removal | NeuroKit2 > (Detect Pulse Peaks)
        - EDA: Filters (Bandpass, Baseline, Zero-Ref) | NeuroKit2 > (Clean, Decompose, Detect SCR)
        """
        if self.current_signal is None:
            return

        sig_type = self.current_signal.signal_type

        # --- Generic signal filters (all types) ---
        filter_action = QAction("&Bandpass Filter...", self)
        filter_action.triggered.connect(self._on_process_filter)
        menu.addAction(filter_action)

        baseline_action = QAction("&Detrend (Polynomial)...", self)
        baseline_action.triggered.connect(self._on_process_baseline)
        menu.addAction(baseline_action)

        zero_ref_action = QAction("&DC Offset Removal...", self)
        zero_ref_action.triggered.connect(self._on_process_zero_reference)
        menu.addAction(zero_ref_action)

        # Notch filter: ECG and PPG only (EDA is low-frequency; notch is not useful)
        if sig_type in (SignalType.ECG, SignalType.PPG):
            notch_action = QAction("&Notch Filter...", self)
            notch_action.triggered.connect(self._on_process_notch)
            menu.addAction(notch_action)

        menu.addSeparator()

        # --- Artifact removal: PPG only (EEMD algorithm is PPG-specific) ---
        if sig_type == SignalType.PPG:
            artifact_action = QAction("&Artifact Removal (EEMD)...", self)
            artifact_action.triggered.connect(self._on_process_artifact_removal)
            menu.addAction(artifact_action)
            menu.addSeparator()

        # --- NeuroKit2 submenu ---
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
            # UNKNOWN type: offer generic peak detection with a warning
            detect_action = QAction("&Detect Peaks (generic)", self)
            detect_action.triggered.connect(self._on_process_detect_peaks)
            nk_menu.addAction(detect_action)

        menu.addSeparator()

        # --- Bad segment repair ---
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

        resample_action = QAction("&Resample...", self)
        resample_action.triggered.connect(self._on_process_resample)
        menu.addAction(resample_action)

        crop_action = QAction("Cr&op...", self)
        crop_action.triggered.connect(self._on_process_crop)
        menu.addAction(crop_action)

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
            if sig_type in (SignalType.ECG, SignalType.PPG):
                hr_action = QAction("Show &Heart Rate", self)
                hr_action.setShortcut("H")
                hr_action.setCheckable(True)
                hr_action.setChecked(self.single_channel_view.is_derived_visible())
                hr_action.triggered.connect(self._on_view_toggle_heart_rate)
                menu.addAction(hr_action)
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
        self._channel_state[key] = {
            "peaks": self._current_peaks,
            "pipeline_steps": list(self.pipeline.steps),
            "raw_samples": self._raw_samples,
            "eda_tonic": self._eda_tonic,
            "eda_phasic": self._eda_phasic,
            "structural_ops": list(self._structural_ops),
            "bad_segments": list(self._current_bad_segments),
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
            else:
                # First visit to this channel — clean slate
                self.pipeline.reset()
                self._raw_samples = None
                self._current_peaks = None
                self._eda_tonic = None
                self._eda_phasic = None
                self._structural_ops = []
                self._current_bad_segments = []
                self.processing_panel.clear()

        # Always re-render (rebuilds LOD renderer, resets view range)
        self.single_channel_view.set_signal(signal)

        # Re-initialise the peak editor and overlay from stored peak data
        if self._current_peaks is not None:
            self.single_channel_view.set_peaks(self._current_peaks)

        # Restore bad segment overlay (may be empty on first visit)
        self.single_channel_view.set_bad_segments(self._current_bad_segments, signal)

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

        self.stacked_widget.setCurrentWidget(self.single_channel_view)
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

    def _on_file_open(self):
        """Handle File > Open."""
        logger.info("File > Open triggered")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Physiological Signal File or Session",
            str(Path.home()),
            "All Supported Files (*.xdf *.csv *.csl.json);;Session Files (*.csl.json);;Physiological Signal Files (*.xdf *.csv);;XDF Files (*.xdf);;CSV Files (*.csv);;All Files (*.*)",
        )

        if not file_path:
            return

        path = Path(file_path)
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
                self._channel_state.clear()  # Discard per-channel state from any previous file
                self._structural_ops.clear()

                self._show_metadata_dialog(session)
                self.signals.file_loaded.emit(session)

                # Go to multi-signal view (rebuilds menus with session-dependent actions enabled)
                if self.current_view_level != "multi":
                    self._on_return_to_multi()
                else:
                    self._build_menus()  # Already in multi view — still need to re-enable actions

                logger.info(f"File loaded: {session.num_signals} signals")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            QMessageBox.critical(self, "File Not Found", f"The file could not be found:\n{e}")
        except ValueError as e:
            logger.error(f"Invalid file format: {e}")
            QMessageBox.critical(self, "Invalid File", f"The file format is invalid:\n{e}")
        except Exception as e:
            logger.exception(f"Unexpected error loading file: {e}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{e}")

    def _load_session_file(self, path: Path):
        """Load a .csl.json session file."""
        try:
            session_data = load_session(path)

            # Load the source file
            source_path = Path(session_data["source_file"])
            if not source_path.exists():
                QMessageBox.warning(
                    self, "Source File Not Found",
                    f"The session references a source file that no longer exists:\n{source_path}\n\nPlease locate the file manually."
                )
                return

            # Load the data file
            if source_path.suffix.lower() == ".csv":
                loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
            else:
                loader = get_loader(source_path)

            session = loader.load(source_path)
            self.current_session = session
            self._channel_state.clear()  # Discard per-channel state from any previous file

            # Restore processing pipeline
            pipeline_data = session_data.get("processing_pipeline", {})
            if pipeline_data:
                # Deserialize and apply to first signal (or user-selected signal)
                # For now, we just log that we have pipeline data
                logger.info(f"Session has {len(pipeline_data.get('steps', []))} processing steps")

            # Restore peaks
            peaks_data = session_data.get("peaks")
            if peaks_data:
                self._current_peaks = PeakData(
                    indices=np.array(peaks_data["indices"], dtype=int),
                    classifications=np.array(peaks_data["classifications"], dtype=int),
                )
                logger.info(f"Restored {self._current_peaks.num_peaks} peaks from session")

            # Go to multi-signal view (rebuilds menus with session-dependent actions enabled)
            self.signals.file_loaded.emit(session)
            if self.current_view_level != "multi":
                self._on_return_to_multi()
            else:
                self._build_menus()

            QMessageBox.information(
                self, "Session Loaded",
                f"Session loaded successfully.\n\nSource: {source_path.name}\nPeaks: {len(peaks_data['indices']) if peaks_data else 0}"
            )

        except Exception as e:
            logger.exception(f"Failed to load session: {e}")
            QMessageBox.critical(self, "Session Load Error", f"Failed to load session:\n{e}")

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
        self.multi_signal_view.set_session(session)
        logger.debug(f"Session loaded: {session.num_signals} signals, {len(session.events)} events")

    def _on_file_save(self):
        """Handle File > Save - save current session."""
        logger.info("File > Save triggered")

        # Can only save if we have a session and are in channel view with processing
        if self.current_session is None:
            QMessageBox.warning(self, "No Session", "No session to save. Load a file first.")
            return

        if self.current_signal is None:
            QMessageBox.warning(
                self, "No Signal",
                "Session save is only available when viewing a single signal.\n\nSelect a signal from the multi-signal or type view first."
            )
            return

        # Get save path
        default_name = self.current_session.source_path.stem + ".csl.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            str(self.current_session.source_path.parent / default_name),
            "Session Files (*.csl.json);;All Files (*.*)"
        )

        if not file_path:
            return

        path = Path(file_path)

        try:
            # Serialize pipeline
            pipeline_data = self.pipeline.serialize()

            # Get view state (current zoom/pan)
            view_state = {}
            if self.current_view_level == "channel":
                plot = self.single_channel_view.plot_widget
                x_min, x_max, y_min, y_max = plot.get_visible_range()
                view_state = {
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                    "signal_type": self.current_signal.signal_type.value,
                    "channel_name": self.current_signal.channel_name,
                }

            # Save session
            save_session(
                source_file=self.current_session.source_path,
                pipeline_data=pipeline_data,
                peaks=self._current_peaks,
                output_path=path,
                view_state=view_state,
            )

            self.statusBar().showMessage(f"Session saved to {path.name}", 5000)
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
            thresh_spin.setValue(3.0)
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

        default_name = (
            self.current_signal.channel_name.replace(" ", "_").lower() + default_ext
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Signal", default_name, file_filter
        )

        if not file_path:
            return

        path = Path(file_path)

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
            str(self.current_session.source_path.parent),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return

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
            str(self.current_session.source_path.parent) if self.current_session else str(Path.home()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return

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

    def _on_process_resample(self):
        """Handle Process > Resample — change signal sampling rate."""
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

        import pandas as pd

        # Resample from raw (pre-pipeline) so pipeline steps can be re-applied.
        # Use pandas time-grid resampling + linear interpolation to match the
        # behaviour of the original processing pipeline (mean-bin then interpolate).
        # This produces a sample count determined by actual signal duration, not a
        # sample-count ratio, which avoids the drift that FFT resampling introduces.
        source = self._raw_samples if self._raw_samples is not None else self.current_signal.samples
        source_timestamps = self.current_signal.timestamps

        t_start = float(source_timestamps[0])
        period_ms = 1000.0 / target_sr
        period_str = f"{period_ms}ms"

        td_index = pd.to_timedelta(source_timestamps - t_start, unit="s")
        series = pd.Series(source, index=td_index)
        resampled_series = series.resample(period_str).mean().interpolate(method="linear")

        resampled_raw = resampled_series.values.astype(np.float64)
        new_timestamps = t_start + resampled_series.index.total_seconds().values
        new_n = len(resampled_raw)

        # Scale peak indices proportionally to the new sample count
        if self._current_peaks and self._current_peaks.num_peaks > 0:
            scale = new_n / len(source)
            new_indices = np.round(self._current_peaks.indices * scale).astype(int)
            new_indices = np.clip(new_indices, 0, new_n - 1)
            self._current_peaks = PeakData(
                indices=new_indices,
                classifications=self._current_peaks.classifications.copy(),
            )

        # Update signal structure; reset pipeline (subsequent filters are now stale)
        object.__setattr__(self.current_signal, "samples", resampled_raw.copy())
        object.__setattr__(self.current_signal, "timestamps", new_timestamps)
        object.__setattr__(self.current_signal, "sampling_rate", float(target_sr))
        self._raw_samples = resampled_raw.copy()
        self._eda_tonic = None
        self._eda_phasic = None
        self.pipeline.reset()

        # Record as a permanent structural op (shown in panel, warned on Reset)
        from cardio_signal_lab.core.data_models import ProcessingStep
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

        self.statusBar().showMessage(
            f"Resampled: {current_sr:.1f} Hz -> {target_sr:.1f} Hz  ({new_n} samples)", 5000
        )
        logger.info(f"Signal resampled from {current_sr:.1f} to {target_sr:.1f} Hz")

    def _on_process_crop(self):
        """Handle Process > Crop — trim signal to a time range or between two events."""
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

        # Crop from the raw (pre-pipeline) signal so Reset still works correctly
        source = self._raw_samples if self._raw_samples is not None else self.current_signal.samples
        cropped_raw = source[start_idx:end_idx]
        cropped_timestamps = timestamps[start_idx:end_idx]

        # Adjust peak indices: keep only peaks inside the crop range, re-zero indices
        if self._current_peaks and self._current_peaks.num_peaks > 0:
            mask = (
                (self._current_peaks.indices >= start_idx)
                & (self._current_peaks.indices < end_idx)
            )
            kept_indices = self._current_peaks.indices[mask] - start_idx
            kept_cls = self._current_peaks.classifications[mask]
            if len(kept_indices) >= 1:
                self._current_peaks = PeakData(
                    indices=kept_indices, classifications=kept_cls
                )
                logger.info(
                    f"Crop: kept {len(kept_indices)} of "
                    f"{self._current_peaks.num_peaks + int(np.sum(~mask))} peaks"
                )
            else:
                self._current_peaks = None
                logger.info("Crop: all peaks fell outside crop range — cleared")

        # Update signal structure; reset pipeline (subsequent filters are now stale)
        object.__setattr__(self.current_signal, "samples", cropped_raw.copy())
        object.__setattr__(self.current_signal, "timestamps", cropped_timestamps)
        self._raw_samples = cropped_raw.copy()
        self._eda_tonic = None
        self._eda_phasic = None
        self.pipeline.reset()

        # Record as a permanent structural op (shown in panel, warned on Reset)
        from cardio_signal_lab.core.data_models import ProcessingStep
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

        duration = float(cropped_timestamps[-1] - cropped_timestamps[0])
        self.statusBar().showMessage(
            f"Cropped to {crop_start:.2f}s - {crop_end:.2f}s  "
            f"({duration:.2f}s, {len(cropped_raw)} samples)", 5000
        )
        logger.info(
            f"Signal cropped: [{crop_start:.2f}, {crop_end:.2f}] s  "
            f"({len(cropped_raw)} samples)"
        )

    def _on_detect_bad_segments(self):
        """Handle Process > Detect Bad Segments - auto-detect amplitude and gap artifacts."""
        if self.current_signal is None:
            return

        from cardio_signal_lab.processing.bad_segment_detection import detect_bad_segments

        dialog = QDialog(self)
        dialog.setWindowTitle("Detect Bad Segments")
        layout = QFormLayout(dialog)

        layout.addRow(QLabel(
            "Detects amplitude transients (rolling MAD) and timestamp gaps.\n"
            "Results are shown as red shaded regions and can be repaired\n"
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

        gap_spin = QDoubleSpinBox()
        gap_spin.setDecimals(1)
        gap_spin.setRange(1.1, 20.0)
        gap_spin.setSingleStep(0.5)
        gap_spin.setValue(2.0)
        gap_spin.setToolTip("Minimum gap size as a multiple of the expected sample interval.")
        layout.addRow("Gap threshold (x interval):", gap_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        bad_segs = detect_bad_segments(
            self.current_signal.samples,
            self.current_signal.timestamps,
            self.current_signal.sampling_rate,
            mad_threshold=mad_spin.value(),
            window_s=window_spin.value(),
            dilation_s=dilation_spin.value(),
            gap_multiplier=gap_spin.value(),
        )

        self._current_bad_segments = bad_segs
        self.single_channel_view.set_bad_segments(bad_segs, self.current_signal)

        n_amp = sum(1 for s in bad_segs if "amplitude" in s.source)
        n_gap = sum(1 for s in bad_segs if "gap" in s.source)
        msg = f"Found {len(bad_segs)} bad segment(s): {n_amp} amplitude, {n_gap} gap"
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

        # Segments have been repaired — clear the overlay
        self._current_bad_segments = []
        self.single_channel_view.clear_bad_segments()

        n = len(segments_param)
        self.statusBar().showMessage(
            f"Interpolated {n} bad segment(s) (cubic spline, anchor=2s)", 5000
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
        """Handle Process > Reset Processing - revert to raw signal.

        Clears all pipeline (filter) steps and restores _raw_samples.
        Structural ops (crop, resample) are displayed as permanent and cannot
        be reverted — the user is warned if any are present.
        """
        if self._raw_samples is None and not self._structural_ops and not self.pipeline.steps:
            self.statusBar().showMessage("No processing to reset", 3000)
            return

        if self.current_signal is None:
            return

        # Warn if structural ops exist — signal cannot be restored to its original
        # pre-crop / pre-resample state because that data is no longer held in memory.
        if self._structural_ops:
            op_names = ", ".join(
                ("Crop" if s.operation == "crop" else "Resample")
                for s in self._structural_ops
            )
            QMessageBox.warning(
                self,
                "Cannot Revert Structural Operations",
                f"The following operation(s) cannot be reverted:\n\n"
                f"  {op_names}\n\n"
                f"Crop and Resample change the signal length and are applied "
                f"directly to the stored raw data.  The original signal is no "
                f"longer held in memory.\n\n"
                f"Reset will clear any filters applied AFTER these operations, "
                f"reverting to the most-recent cropped/resampled state.",
            )

        if self._raw_samples is not None:
            object.__setattr__(self.current_signal, "samples", self._raw_samples.copy())

        self.pipeline.reset()
        self._structural_ops.clear()
        self._current_peaks = None
        self._eda_tonic = None
        self._eda_phasic = None
        self._current_bad_segments = []

        self.single_channel_view.set_signal(self.current_signal)
        if self.current_session:
            self.single_channel_view.set_events(self.current_session.events or [])

        self.single_channel_view.clear_derived()
        self.single_channel_view.clear_bad_segments()
        self.processing_panel.clear()
        self.statusBar().showMessage("Processing reset", 3000)
        logger.info("Processing pipeline reset")

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

    # ---- Type-View Operations ----

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

    def _on_help_about(self):
        logger.info("Help > About triggered")
