"""Single-signal view for detailed signal processing and correction.

Displays one signal prominently in a full-size plot with interactive features
for peak correction (double-click to add, click to select, hotkeys to classify).
Provides zoom/pan via mouse wheel and drag.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cardio_signal_lab.core import PeakClassification, PeakData
from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.peak_overlay import PeakOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.processing import PeakEditor
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import SignalData


class SingleSignalView(QWidget):
    """Single-signal view for detailed processing and peak correction.

    Displays one signal in a full-size plot with interactive features:
    - Zoom/pan with mouse wheel and drag
    - Double-click to add peak (MANUAL classification)
    - Click on peak to select it
    - Delete/Backspace to remove selected peak
    - D/M/E/B hotkeys to classify peak
    - Arrow keys to navigate peaks
    - Ctrl+Z/Ctrl+Y for undo/redo

    Signals:
        return_to_multi_requested: Emitted when user wants to return to multi-signal mode
        peaks_changed: Emitted when peaks are added/removed/classified
    """

    return_to_multi_requested = Signal()
    peaks_changed = Signal()

    def __init__(self, parent=None):
        """Initialize single-signal view.

        Args:
            parent: Parent QWidget
        """
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.signal_data: SignalData | None = None
        self.event_overlay: EventOverlay | None = None
        self.peak_overlay: PeakOverlay | None = None
        self.peak_editor: PeakEditor | None = None
        self.session_events: list = []

        # Double-click tracking
        self.last_click_time = 0
        self.double_click_threshold = 0.3  # seconds

        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget
        self.plot_widget = SignalPlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Create overlays
        self.event_overlay = EventOverlay(self.plot_widget)
        self.peak_overlay = PeakOverlay(self.plot_widget)

        # Connect signals
        self._connect_signals()

        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        self.plot_widget.setFocusPolicy(Qt.StrongFocus)

        logger.debug("SingleSignalView initialized with peak correction")

    def _connect_signals(self):
        """Connect to app signals and plot events."""
        # Connect to plot click events
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Connect to peak overlay click
        self.peak_overlay.scatter.sigClicked.connect(self._on_peak_clicked)

    def set_signal(self, signal: SignalData):
        """Set signal to display.

        Args:
            signal: SignalData to display
        """
        self.signal_data = signal
        self.plot_widget.set_signal(signal)

        # Re-apply events after signal is set (in case they were cleared)
        if self.session_events:
            logger.debug(f"Re-applying {len(self.session_events)} events after set_signal")
            self.event_overlay.set_events(self.session_events)

        logger.info(
            f"Single-signal view loaded: {signal.signal_type.value}, "
            f"{signal.channel_name}, {len(signal.samples)} samples"
        )

    def clear(self):
        """Clear plot and reset to empty state."""
        self.plot_widget.clear()
        self.signal_data = None

        logger.debug("SingleSignalView cleared")

    def reset_view(self):
        """Reset view to show full signal range."""
        self.plot_widget.reset_view()

    def get_visible_range(self) -> tuple[float, float, float, float]:
        """Get current visible range.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        return self.plot_widget.get_visible_range()

    def zoom_in(self):
        """Zoom in on plot (centered on current view)."""
        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.scaleBy((0.5, 0.5))
        logger.debug("Zoomed in")

    def zoom_out(self):
        """Zoom out on plot (centered on current view)."""
        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.scaleBy((2.0, 2.0))
        logger.debug("Zoomed out")

    def enable_mouse_interaction(self, enabled: bool = True):
        """Enable or disable mouse interaction.

        Args:
            enabled: True to enable mouse pan/zoom, False to disable
        """
        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.setMouseEnabled(x=enabled, y=enabled)
        logger.debug(f"Mouse interaction {'enabled' if enabled else 'disabled'}")

    def get_signal_data(self) -> SignalData | None:
        """Get current signal data.

        Returns:
            Current SignalData or None if no signal loaded
        """
        return self.signal_data

    def set_events(self, events: list):
        """Set events to display on plot.

        Args:
            events: List of EventData objects
        """
        logger.info(f"SingleSignalView.set_events() called with {len(events)} events")
        self.session_events = events
        if self.event_overlay:
            logger.debug(f"EventOverlay exists, calling set_events")
            self.event_overlay.set_events(events)
            logger.debug(f"EventOverlay now has {self.event_overlay.get_num_events()} events")
        else:
            logger.warning("EventOverlay is None in set_events!")
        logger.info(f"Set {len(events)} events on single signal view")

    def toggle_events(self):
        """Toggle event overlay visibility."""
        if self.event_overlay:
            self.event_overlay.toggle_visibility()
            visible = self.event_overlay.is_visible()
            logger.debug(f"Toggled events: {'visible' if visible else 'hidden'}")

    def set_events_visible(self, visible: bool):
        """Set event overlay visibility.

        Args:
            visible: True to show events, False to hide
        """
        if self.event_overlay:
            self.event_overlay.set_visible(visible)
            logger.debug(f"Set events visibility: {visible}")

    def are_events_visible(self) -> bool:
        """Check if events are currently visible.

        Returns:
            True if events are visible, False otherwise
        """
        return self.event_overlay.is_visible() if self.event_overlay else False

    def jump_to_start(self):
        """Jump to start of signal."""
        if self.plot_widget.lod_renderer is None:
            return

        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()
        duration = x_max - x_min

        # Show first 10% of signal (or 10 seconds, whichever is smaller)
        view_width = min(duration * 0.1, 10.0)

        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.setXRange(x_min, x_min + view_width, padding=0)

        logger.debug(f"Jumped to signal start: [{x_min:.2f}, {x_min + view_width:.2f}]")

    def jump_to_end(self):
        """Jump to end of signal."""
        if self.plot_widget.lod_renderer is None:
            return

        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()
        duration = x_max - x_min

        # Show last 10% of signal (or 10 seconds, whichever is smaller)
        view_width = min(duration * 0.1, 10.0)

        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.setXRange(x_max - view_width, x_max, padding=0)

        logger.debug(f"Jumped to signal end: [{x_max - view_width:.2f}, {x_max:.2f}]")

    def jump_to_time(self, time: float):
        """Jump to specific timestamp and center view on it.

        Args:
            time: Timestamp to jump to (in seconds)
        """
        if self.plot_widget.lod_renderer is None:
            return

        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()

        # Clamp time to valid range
        time = max(x_min, min(x_max, time))

        # Get current view width to maintain zoom level
        current_x_min, current_x_max, _, _ = self.plot_widget.get_visible_range()
        view_width = current_x_max - current_x_min

        # Center on requested time
        new_x_min = time - view_width / 2
        new_x_max = time + view_width / 2

        view_box = self.plot_widget.plotItem.getViewBox()
        view_box.setXRange(new_x_min, new_x_max, padding=0)

        logger.debug(f"Jumped to time {time:.2f}s")

    # ===== Peak Correction Methods =====

    def set_peaks(self, peaks: PeakData):
        """Set peaks to display and enable editing.

        Args:
            peaks: PeakData with indices and classifications
        """
        if self.signal_data is None:
            logger.warning("Cannot set peaks: no signal loaded")
            return

        # Initialize peak editor if needed
        if self.peak_editor is None:
            self.peak_editor = PeakEditor(peaks=peaks)
        else:
            self.peak_editor.indices = peaks.indices.copy()
            self.peak_editor.classifications = peaks.classifications.copy()

        # Display peaks
        self.peak_overlay.set_peaks(self.signal_data, peaks)

        logger.info(f"Loaded {peaks.num_peaks} peaks for editing")

    def _on_mouse_clicked(self, event):
        """Handle mouse click on plot for peak add/select.

        Double-click: Add peak at cursor
        Single-click (not on peak): Deselect
        """
        import time

        if self.signal_data is None or self.peak_editor is None:
            return

        # Check for double-click
        current_time = time.time()
        is_double_click = (current_time - self.last_click_time) < self.double_click_threshold
        self.last_click_time = current_time

        # Get click position in data coordinates
        scene_pos = event.scenePos()
        view_pos = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
        click_time = view_pos.x()

        if is_double_click:
            # Double-click: Add peak at nearest sample
            self._add_peak_at_time(click_time)
        else:
            # Single-click: Deselect if not clicking on peak
            # (Peak clicks handled by _on_peak_clicked)
            pass

    def _on_peak_clicked(self, scatter, points):
        """Handle click on peak marker (select it).

        Args:
            scatter: ScatterPlotItem
            points: List of clicked points
        """
        if not points or self.peak_editor is None:
            return

        point = points[0]
        peak_index = int(point.data())

        # Select peak
        self.peak_editor.select_peak(peak_index)
        self.peak_overlay.select_peak(peak_index)

        logger.info(f"Selected peak {peak_index}")

    def _add_peak_at_time(self, time: float):
        """Add peak at nearest sample to given time.

        Args:
            time: Time position (seconds)
        """
        if self.signal_data is None or self.peak_editor is None:
            return

        # Find nearest sample index
        time_diffs = np.abs(self.signal_data.timestamps - time)
        sample_index = int(np.argmin(time_diffs))

        # Add peak
        success = self.peak_editor.add_peak(sample_index, PeakClassification.MANUAL)

        if success:
            # Update display
            self._update_peak_display()
            self.peaks_changed.emit()
            logger.info(f"Added peak at sample {sample_index}, time {time:.3f}s")
        else:
            logger.warning(f"Failed to add peak at sample {sample_index}")

    def _update_peak_display(self):
        """Update peak overlay after peak edits."""
        if self.signal_data is None or self.peak_editor is None:
            return

        peak_data = self.peak_editor.get_peak_data()
        self.peak_overlay.set_peaks(self.signal_data, peak_data)

        # Restore selection
        if self.peak_editor.selected_index is not None:
            self.peak_overlay.select_peak(self.peak_editor.selected_index)

    def keyPressEvent(self, event):
        """Handle keyboard events for peak correction.

        Delete/Backspace: Remove selected peak
        D: Mark as Auto-detected
        M: Mark as Manual
        E: Mark as Ectopic
        B: Mark as Bad
        C: Cycle classification
        Left/Right arrow: Navigate peaks
        Home/End: Jump to first/last peak
        Ctrl+Z: Undo
        Ctrl+Y/Ctrl+Shift+Z: Redo
        """
        if self.peak_editor is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # Delete peak
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.peak_editor.selected_index is not None:
                success = self.peak_editor.delete_peak(self.peak_editor.selected_index)
                if success:
                    self._update_peak_display()
                    self.peaks_changed.emit()
                    logger.info("Deleted selected peak")

        # Classify peaks
        elif key == Qt.Key_D:
            if self.peak_editor.selected_index is not None:
                self.peak_editor.classify_peak(self.peak_editor.selected_index, PeakClassification.AUTO)
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Classified peak as AUTO")

        elif key == Qt.Key_M:
            if self.peak_editor.selected_index is not None:
                self.peak_editor.classify_peak(self.peak_editor.selected_index, PeakClassification.MANUAL)
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Classified peak as MANUAL")

        elif key == Qt.Key_E:
            if self.peak_editor.selected_index is not None:
                self.peak_editor.classify_peak(self.peak_editor.selected_index, PeakClassification.ECTOPIC)
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Classified peak as ECTOPIC")

        elif key == Qt.Key_B:
            if self.peak_editor.selected_index is not None:
                self.peak_editor.classify_peak(self.peak_editor.selected_index, PeakClassification.BAD)
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Classified peak as BAD")

        elif key == Qt.Key_C:
            if self.peak_editor.selected_index is not None:
                self.peak_editor.cycle_classification(self.peak_editor.selected_index)
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Cycled peak classification")

        # Navigate peaks
        elif key == Qt.Key_Right:
            new_idx = self.peak_editor.navigate_peaks('next')
            if new_idx is not None:
                self.peak_overlay.select_peak(new_idx)
                self._center_on_peak(new_idx)
                logger.info(f"Navigated to next peak: {new_idx}")

        elif key == Qt.Key_Left:
            new_idx = self.peak_editor.navigate_peaks('prev')
            if new_idx is not None:
                self.peak_overlay.select_peak(new_idx)
                self._center_on_peak(new_idx)
                logger.info(f"Navigated to previous peak: {new_idx}")

        elif key == Qt.Key_Home:
            new_idx = self.peak_editor.navigate_peaks('first')
            if new_idx is not None:
                self.peak_overlay.select_peak(new_idx)
                self._center_on_peak(new_idx)
                logger.info(f"Navigated to first peak: {new_idx}")

        elif key == Qt.Key_End:
            new_idx = self.peak_editor.navigate_peaks('last')
            if new_idx is not None:
                self.peak_overlay.select_peak(new_idx)
                self._center_on_peak(new_idx)
                logger.info(f"Navigated to last peak: {new_idx}")

        # Undo/Redo
        elif key == Qt.Key_Z and modifiers & Qt.ControlModifier:
            if modifiers & Qt.ShiftModifier:
                # Ctrl+Shift+Z: Redo
                if self.peak_editor.redo():
                    self._update_peak_display()
                    self.peaks_changed.emit()
                    logger.info("Redid peak operation")
            else:
                # Ctrl+Z: Undo
                if self.peak_editor.undo():
                    self._update_peak_display()
                    self.peaks_changed.emit()
                    logger.info("Undid peak operation")

        elif key == Qt.Key_Y and modifiers & Qt.ControlModifier:
            # Ctrl+Y: Redo
            if self.peak_editor.redo():
                self._update_peak_display()
                self.peaks_changed.emit()
                logger.info("Redid peak operation")

        # Deselect
        elif key == Qt.Key_Escape:
            self.peak_editor.select_peak(None)
            self.peak_overlay.select_peak(None)
            logger.info("Deselected peak")

        else:
            super().keyPressEvent(event)

    def _center_on_peak(self, peak_index: int):
        """Center view on a peak.

        Args:
            peak_index: Index in peaks array
        """
        if self.signal_data is None or peak_index >= len(self.peak_editor.indices):
            return

        sample_index = self.peak_editor.indices[peak_index]
        peak_time = self.signal_data.timestamps[sample_index]

        # Center view on peak
        view_box = self.plot_widget.plotItem.getViewBox()
        x_range = view_box.viewRange()[0]
        view_width = x_range[1] - x_range[0]

        new_x_min = peak_time - view_width / 2
        new_x_max = peak_time + view_width / 2

        view_box.setXRange(new_x_min, new_x_max, padding=0)

    def get_peak_data(self) -> PeakData | None:
        """Get current peak data from editor.

        Returns:
            PeakData or None if no editor active
        """
        if self.peak_editor is None:
            return None
        return self.peak_editor.get_peak_data()

    def reset_peak_correction(self):
        """Reset peak editor and clear peaks."""
        if self.peak_editor is not None:
            self.peak_editor.reset()
        self.peak_overlay.clear()
        logger.info("Peak correction reset")
