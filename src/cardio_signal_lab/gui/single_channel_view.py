"""Single-channel view for detailed signal processing and peak correction.

Displays one channel prominently in a full-size plot with fully interactive
peak correction: double-click to add peaks, click to select, Delete to remove,
arrow keys to navigate, D/M/E/B to classify, Ctrl+Z/Y to undo/redo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cardio_signal_lab.core import PeakClassification, PeakData
from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.peak_overlay import PeakOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.processing.peak_correction import PeakEditor
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import SignalData


class SingleChannelView(QWidget):
    """Single-channel view for processing and interactive peak correction.

    Signals:
        return_to_multi_requested: User wants to return to multi-signal mode.
        peaks_changed: Emitted after any peak edit (add/delete/classify/undo/redo).
    """

    return_to_multi_requested = Signal()
    peaks_changed = Signal(object)  # PeakData

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.signal_data: SignalData | None = None
        self.session_events: list = []

        self.peak_editor: PeakEditor | None = None
        self.peak_overlay: PeakOverlay | None = None
        self.event_overlay: EventOverlay | None = None

        # Accept keyboard focus so keyPressEvent fires
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = SignalPlotWidget()
        layout.addWidget(self.plot_widget)

        self.event_overlay = EventOverlay(self.plot_widget)

        # Connect scene-level mouse clicks (double-click to add peaks)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

        logger.debug("SingleChannelView initialized")

    # ---- Signal Display ----

    def set_signal(self, signal: SignalData):
        """Display a signal. Clears any existing peaks."""
        self.signal_data = signal
        self.plot_widget.set_signal(signal)

        if self.session_events:
            self.event_overlay.set_events(self.session_events)

        logger.info(
            f"Single-channel view: {signal.signal_type.value} / {signal.channel_name} "
            f"({len(signal.samples)} samples)"
        )

    def clear(self):
        self.plot_widget.clear()
        self.signal_data = None
        self.clear_peaks()

    # ---- Peak Correction ----

    def set_peaks(self, peaks: PeakData):
        """Initialize the peak editor and overlay with detected peaks."""
        self.peak_editor = PeakEditor(peaks)

        if self.peak_overlay is None:
            self.peak_overlay = PeakOverlay(self.plot_widget)
            # Sync selection clicks to the editor
            self.peak_overlay.scatter.sigClicked.connect(self._on_overlay_peak_clicked)

        self._refresh_overlay(emit=False)
        self.setFocus()

    def clear_peaks(self):
        """Remove all peak markers and reset the editor."""
        self.peak_editor = None
        if self.peak_overlay is not None:
            self.peak_overlay.clear()

    def undo(self):
        """Undo last peak edit."""
        if self.peak_editor and self.peak_editor.undo():
            self._refresh_overlay()

    def redo(self):
        """Redo last undone peak edit."""
        if self.peak_editor and self.peak_editor.redo():
            self._refresh_overlay()

    def _refresh_overlay(self, emit: bool = True):
        """Sync editor state to overlay and optionally notify listeners."""
        if self.peak_editor is None or self.signal_data is None or self.peak_overlay is None:
            return

        peak_data = self.peak_editor.get_peak_data()
        self.peak_overlay.set_peaks(self.signal_data, peak_data)

        if self.peak_editor.selected_index is not None:
            self.peak_overlay.select_peak(self.peak_editor.selected_index)

        if emit:
            self.peaks_changed.emit(peak_data)

    # ---- Mouse Events ----

    def _on_scene_clicked(self, event):
        """Double-click on signal background adds a new peak."""
        if self.peak_editor is None or self.signal_data is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not event.double():
            return

        # Map scene position to signal data coordinates
        view_box = self.plot_widget.plotItem.getViewBox()
        data_pos = view_box.mapSceneToView(event.scenePos())
        time_clicked = data_pos.x()

        timestamps = self.signal_data.timestamps
        sample_idx = int(np.clip(
            np.searchsorted(timestamps, time_clicked),
            0, len(timestamps) - 1
        ))

        # Ignore double-clicks very close to an existing peak (treat as selection)
        tolerance = int(0.05 * self.signal_data.sampling_rate)  # 50 ms
        if self.peak_editor.find_nearest_peak(sample_idx, max_distance=tolerance) is not None:
            return

        if self.peak_editor.add_peak(sample_idx):
            self._refresh_overlay()
            logger.info(f"Added peak at sample {sample_idx} (t={time_clicked:.3f}s)")

        event.accept()

    def _on_overlay_peak_clicked(self, scatter, points):
        """Sync peak selection from overlay click to the editor."""
        if not points or self.peak_editor is None:
            return
        peak_index = int(points[0].data())
        self.peak_editor.select_peak(peak_index)
        self.peak_overlay.select_peak(peak_index)
        self.setFocus()  # Regrab keyboard focus after mouse click

    # ---- Keyboard Events ----

    def keyPressEvent(self, event: QKeyEvent):
        if self.peak_editor is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        mods = event.modifiers()
        no_mods = mods == Qt.KeyboardModifier.NoModifier
        ctrl = mods == Qt.KeyboardModifier.ControlModifier

        selected = self.peak_editor.selected_index

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and no_mods:
            if selected is not None:
                self.peak_editor.delete_peak(selected)
                self._refresh_overlay()

        elif key == Qt.Key.Key_Left and no_mods:
            new_idx = self.peak_editor.navigate_peaks("prev")
            if new_idx is not None:
                self._refresh_overlay(emit=False)
                self._scroll_to_peak(new_idx)

        elif key == Qt.Key.Key_Right and no_mods:
            new_idx = self.peak_editor.navigate_peaks("next")
            if new_idx is not None:
                self._refresh_overlay(emit=False)
                self._scroll_to_peak(new_idx)

        elif key == Qt.Key.Key_D and no_mods:
            if selected is not None:
                self.peak_editor.classify_peak(selected, PeakClassification.AUTO)
                self._refresh_overlay()

        elif key == Qt.Key.Key_M and no_mods:
            if selected is not None:
                self.peak_editor.classify_peak(selected, PeakClassification.MANUAL)
                self._refresh_overlay()

        elif key == Qt.Key.Key_E and no_mods:
            if selected is not None:
                self.peak_editor.classify_peak(selected, PeakClassification.ECTOPIC)
                self._refresh_overlay()

        elif key == Qt.Key.Key_B and no_mods:
            if selected is not None:
                self.peak_editor.classify_peak(selected, PeakClassification.BAD)
                self._refresh_overlay()

        elif key == Qt.Key.Key_Z and ctrl:
            self.undo()

        elif key == Qt.Key.Key_Y and ctrl:
            self.redo()

        else:
            super().keyPressEvent(event)

    def _scroll_to_peak(self, peak_idx: int):
        """Center view on the given peak without changing zoom level."""
        if self.signal_data is None or self.peak_editor is None:
            return
        sample = self.peak_editor.indices[peak_idx]
        peak_time = self.signal_data.timestamps[sample]
        self.jump_to_time(peak_time)

    # ---- Events Display ----

    def set_events(self, events: list):
        self.session_events = events
        if self.event_overlay:
            self.event_overlay.set_events(events)

    def toggle_events(self):
        if self.event_overlay:
            self.event_overlay.toggle_visibility()

    def set_events_visible(self, visible: bool):
        if self.event_overlay:
            self.event_overlay.set_visible(visible)

    def are_events_visible(self) -> bool:
        return self.event_overlay.is_visible() if self.event_overlay else False

    # ---- Navigation ----

    def reset_view(self):
        self.plot_widget.reset_view()

    def get_visible_range(self) -> tuple[float, float, float, float]:
        return self.plot_widget.get_visible_range()

    def zoom_in(self):
        self.plot_widget.plotItem.getViewBox().scaleBy((0.5, 0.5))

    def zoom_out(self):
        self.plot_widget.plotItem.getViewBox().scaleBy((2.0, 2.0))

    def enable_mouse_interaction(self, enabled: bool = True):
        self.plot_widget.plotItem.getViewBox().setMouseEnabled(x=enabled, y=enabled)

    def get_signal_data(self) -> SignalData | None:
        return self.signal_data

    def jump_to_start(self):
        if self.plot_widget.lod_renderer is None:
            return
        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()
        view_width = min((x_max - x_min) * 0.1, 10.0)
        self.plot_widget.plotItem.getViewBox().setXRange(x_min, x_min + view_width, padding=0)

    def jump_to_end(self):
        if self.plot_widget.lod_renderer is None:
            return
        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()
        view_width = min((x_max - x_min) * 0.1, 10.0)
        self.plot_widget.plotItem.getViewBox().setXRange(x_max - view_width, x_max, padding=0)

    def jump_to_time(self, time: float):
        if self.plot_widget.lod_renderer is None:
            return
        x_min, x_max, _, _ = self.plot_widget.lod_renderer.get_full_range()
        time = max(x_min, min(x_max, time))
        current_x_min, current_x_max, _, _ = self.plot_widget.get_visible_range()
        view_width = current_x_max - current_x_min
        self.plot_widget.plotItem.getViewBox().setXRange(
            time - view_width / 2, time + view_width / 2, padding=0
        )
