"""Main application window with three-level view hierarchy and dynamic menus."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QDialog,
    QFormLayout,
    QDoubleSpinBox,
    QDialogButtonBox,
    QStackedWidget,
)
from loguru import logger

from cardio_signal_lab.config import get_config, get_keysequence
from cardio_signal_lab.core import get_loader, SignalType, CsvLoader
from cardio_signal_lab.gui.multi_signal_view import MultiSignalView
from cardio_signal_lab.gui.signal_type_view import SignalTypeView
from cardio_signal_lab.gui.single_channel_view import SingleChannelView
from cardio_signal_lab.gui.status_bar import AppStatusBar
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

        # Build initial menus
        self._build_menus()

        # Connect view signals
        self.multi_signal_view.signal_type_selected.connect(self._on_signal_type_selected)
        self.signal_type_view.channel_selected.connect(self._on_channel_selected)
        self.single_channel_view.return_to_multi_requested.connect(self._on_return_to_multi)

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
        """Menus for signal-type view: File, Edit(disabled), Select(channels), View, Help."""
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

        add_peak_action = QAction("&Add Peak", self)
        add_peak_action.setShortcut(get_keysequence("edit_add_peak"))
        add_peak_action.triggered.connect(self._on_edit_add_peak)
        menu.addAction(add_peak_action)

        delete_peak_action = QAction("&Delete Peak", self)
        delete_peak_action.setShortcut(get_keysequence("edit_delete_peak"))
        delete_peak_action.triggered.connect(self._on_edit_delete_peak)
        menu.addAction(delete_peak_action)

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
        """Add Process menu actions (channel view only)."""
        filter_action = QAction("&Filter...", self)
        filter_action.triggered.connect(self._on_process_filter)
        menu.addAction(filter_action)

        artifact_action = QAction("&Artifact Removal (EEMD)", self)
        artifact_action.triggered.connect(self._on_process_artifact_removal)
        menu.addAction(artifact_action)

        detect_peaks_action = QAction("&Detect Peaks", self)
        detect_peaks_action.triggered.connect(self._on_process_detect_peaks)
        menu.addAction(detect_peaks_action)

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

        # Toggle events
        toggle_events_action = QAction("Toggle Event Markers", self)
        toggle_events_action.setShortcut("E")
        toggle_events_action.triggered.connect(self._on_view_toggle_events)
        menu.addAction(toggle_events_action)

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

    def _switch_to_channel_view(self, signal: SignalData):
        """Switch to single-channel view for processing."""
        self.current_view_level = "channel"
        self.current_signal = signal

        self.single_channel_view.set_signal(signal)

        # Pass events
        if self.current_session:
            events = self.current_session.events or []
            self.single_channel_view.set_events(events)

        self.stacked_widget.setCurrentWidget(self.single_channel_view)
        self._build_menus()

        # Update status bar
        self.statusBar().showMessage(
            f"Channel: {signal.signal_type.value.upper()} - {signal.channel_name}", 0
        )
        logger.info(f"Switched to channel view: {signal.channel_name}")

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
            "Open Physiological Signal File",
            str(Path.home()),
            "Physiological Signal Files (*.xdf *.csv);;XDF Files (*.xdf);;CSV Files (*.csv);;All Files (*.*)",
        )

        if not file_path:
            return

        path = Path(file_path)
        logger.info(f"Loading file: {path}")

        try:
            if path.suffix.lower() == ".csv":
                loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
            else:
                loader = get_loader(path)

            session = loader.load(path)
            self.current_session = session

            self._show_metadata_dialog(session)
            self.signals.file_loaded.emit(session)

            # Go to multi-signal view
            if self.current_view_level != "multi":
                self._on_return_to_multi()

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
        logger.info("File > Save triggered")
        self.signals.file_save_requested.emit()

    def _on_file_export(self):
        logger.info("File > Export triggered")
        self.signals.file_export_requested.emit()

    # ---- Edit Operations ----

    def _on_edit_undo(self):
        logger.info("Edit > Undo triggered")

    def _on_edit_redo(self):
        logger.info("Edit > Redo triggered")

    def _on_edit_add_peak(self):
        logger.info("Edit > Add Peak triggered")

    def _on_edit_delete_peak(self):
        logger.info("Edit > Delete Peak triggered")

    # ---- Process Operations ----

    def _on_process_filter(self):
        logger.info("Process > Filter triggered")

    def _on_process_artifact_removal(self):
        logger.info("Process > Artifact Removal triggered")

    def _on_process_detect_peaks(self):
        logger.info("Process > Detect Peaks triggered")

    def _on_process_reset(self):
        logger.info("Process > Reset Processing triggered")

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

    # ---- Help Operations ----

    def _on_help_shortcuts(self):
        logger.info("Help > Keyboard Shortcuts triggered")

    def _on_help_about(self):
        logger.info("Help > About triggered")
