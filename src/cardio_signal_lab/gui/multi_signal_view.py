"""Multi-signal view with synchronized stacked plots grouped by signal type.

Displays signals from a RecordingSession grouped by signal type (one plot per type).
Multi-channel types (e.g., ECG with 4 channels) overlay all channels in the same plot.
Click on any subplot to select that signal type and drill down.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import RecordingSession, SignalData, SignalType


# Colors for overlaying multiple channels of the same type
CHANNEL_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]


class MultiSignalView(QWidget):
    """Multi-signal overview with stacked synchronized plots grouped by type.

    Groups signals by signal_type and creates one plot per type. Multi-channel
    types overlay all channels in the same plot with different colors. Click on
    any plot to select that signal type.

    Signals:
        signal_type_selected: Emitted when user clicks on a plot (SignalType)
    """

    signal_type_selected = Signal(object)  # SignalType

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.session: RecordingSession | None = None
        self.plot_widgets: list[SignalPlotWidget] = []
        self.event_overlays: list[EventOverlay] = []
        # Maps SignalType -> list of SignalData for that type
        self.type_signals: OrderedDict[SignalType, list[SignalData]] = OrderedDict()
        # Maps plot index -> SignalType
        self.plot_type_map: dict[int, SignalType] = {}

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)

        logger.debug("MultiSignalView initialized")

    def set_session(self, session: RecordingSession):
        """Load recording session and create stacked plots grouped by signal type.

        Args:
            session: RecordingSession containing multiple signals
        """
        self.session = session
        self.clear()

        # Group signals by type (preserve insertion order)
        self.type_signals = OrderedDict()
        for signal in session.signals:
            if signal.signal_type not in self.type_signals:
                self.type_signals[signal.signal_type] = []
            self.type_signals[signal.signal_type].append(signal)

        logger.info(
            f"Grouped {len(session.signals)} signals into {len(self.type_signals)} types: "
            f"{[f'{t.value}({len(s)})' for t, s in self.type_signals.items()]}"
        )

        # Create one plot per signal type
        for plot_idx, (signal_type, signals) in enumerate(self.type_signals.items()):
            plot_widget = SignalPlotWidget()

            if len(signals) == 1:
                # Single channel - use standard set_signal
                plot_widget.set_signal(signals[0])
            else:
                # Multiple channels - overlay in same plot
                self._overlay_signals(plot_widget, signals, signal_type)

            # Make plot clickable for signal type selection
            plot_widget.plotItem.scene().sigMouseClicked.connect(
                lambda event, st=signal_type: self._on_plot_clicked(event, st)
            )

            self.plot_widgets.append(plot_widget)
            self.plot_type_map[plot_idx] = signal_type
            self.layout.addWidget(plot_widget)

        # Link x-axes after all plots have data
        if len(self.plot_widgets) > 1:
            first_plot = self.plot_widgets[0]
            for plot_widget in self.plot_widgets[1:]:
                plot_widget.plotItem.setXLink(first_plot.plotItem)
            logger.debug(f"Linked {len(self.plot_widgets)} plots for synchronized x-axis")

        # Add event overlays
        if session.events:
            for plot_widget in self.plot_widgets:
                event_overlay = EventOverlay(plot_widget)
                event_overlay.set_events(session.events)
                self.event_overlays.append(event_overlay)
            logger.debug(f"Added event overlays with {len(session.events)} events")

        logger.info(f"Multi-signal view loaded: {len(self.plot_widgets)} type plots")

    def _overlay_signals(self, plot_widget: SignalPlotWidget, signals: list[SignalData],
                         signal_type: SignalType):
        """Overlay multiple channels in a single plot with different colors.

        Args:
            plot_widget: Plot widget to add curves to
            signals: List of SignalData for this type
            signal_type: Signal type for title
        """
        from cardio_signal_lab.gui.lod_renderer import LODRenderer

        plot_widget._updating_range = True
        try:
            # Use first signal for LOD renderer (for navigation)
            first = signals[0]
            plot_widget.signal_data = first
            plot_widget.lod_renderer = LODRenderer(first.timestamps, first.samples, num_levels=8)

            x_min, x_max, y_min, y_max = plot_widget.lod_renderer.get_full_range()

            # Track global y range across all channels
            global_y_min = y_min
            global_y_max = y_max

            # Add each channel as a separate curve
            for i, signal in enumerate(signals):
                color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                pen = pg.mkPen(color=color, width=1)

                lod = LODRenderer(signal.timestamps, signal.samples, num_levels=8)
                t_data, s_data = lod.get_render_data(x_min, x_max, 1000)

                curve = plot_widget.plotItem.plot(t_data, s_data, pen=pen, name=signal.channel_name)

                # Update global y range
                _, _, sy_min, sy_max = lod.get_full_range()
                global_y_min = min(global_y_min, sy_min)
                global_y_max = max(global_y_max, sy_max)

            # Set the first curve as the main plot_curve for LOD updates
            items = plot_widget.plotItem.listDataItems()
            if items:
                plot_widget.plot_curve = items[0]

            # Set title and ranges
            n_channels = len(signals)
            plot_widget.plotItem.setTitle(
                f"{signal_type.value.upper()} ({n_channels} channels)"
            )
            plot_widget.plotItem.disableAutoRange()
            view_box = plot_widget.plotItem.getViewBox()
            view_box.setRange(
                xRange=(x_min, x_max),
                yRange=(global_y_min, global_y_max),
                padding=0.05, update=True
            )
        finally:
            plot_widget._updating_range = False

    def _on_plot_clicked(self, event, signal_type: SignalType):
        """Handle click on a plot - select that signal type.

        Args:
            event: Mouse click event
            signal_type: SignalType associated with clicked plot
        """
        from PySide6.QtCore import Qt
        if event.button() != Qt.MouseButton.LeftButton:
            return
        logger.info(f"Signal type selected from multi-view: {signal_type.value}")
        self.signal_type_selected.emit(signal_type)
        self.app_signals.signal_type_selected.emit(signal_type)

    def clear(self):
        """Clear all plots and reset view."""
        for overlay in self.event_overlays:
            overlay.clear()
        self.event_overlays.clear()

        for plot_widget in self.plot_widgets:
            self.layout.removeWidget(plot_widget)
            plot_widget.deleteLater()

        self.plot_widgets.clear()
        self.type_signals.clear()
        self.plot_type_map.clear()
        self.session = None

        logger.debug("MultiSignalView cleared")

    def reset_view(self):
        """Reset all plots to show full signal range."""
        for plot_widget in self.plot_widgets:
            plot_widget.reset_view()
        logger.debug("All plots reset to full range")

    def get_num_signals(self) -> int:
        """Get number of signal type plots currently displayed."""
        return len(self.plot_widgets)

    def get_unique_signal_types(self) -> list[SignalType]:
        """Get list of unique signal types in current session."""
        return list(self.type_signals.keys())

    def get_signals_for_type(self, signal_type: SignalType) -> list[SignalData]:
        """Get all signals for a given type."""
        return self.type_signals.get(signal_type, [])

    def zoom_in(self):
        """Zoom in on all plots."""
        for plot_widget in self.plot_widgets:
            view_box = plot_widget.plotItem.getViewBox()
            view_box.scaleBy((0.5, 0.5))

    def zoom_out(self):
        """Zoom out on all plots."""
        for plot_widget in self.plot_widgets:
            view_box = plot_widget.plotItem.getViewBox()
            view_box.scaleBy((2.0, 2.0))

    def jump_to_start(self):
        """Jump to start of all signals."""
        if not self.plot_widgets:
            return
        first_plot = self.plot_widgets[0]
        if first_plot.lod_renderer is None:
            return
        x_min, x_max, _, _ = first_plot.lod_renderer.get_full_range()
        duration = x_max - x_min
        view_width = min(duration * 0.1, 10.0)
        view_box = first_plot.plotItem.getViewBox()
        view_box.setXRange(x_min, x_min + view_width, padding=0)

    def jump_to_end(self):
        """Jump to end of all signals."""
        if not self.plot_widgets:
            return
        first_plot = self.plot_widgets[0]
        if first_plot.lod_renderer is None:
            return
        x_min, x_max, _, _ = first_plot.lod_renderer.get_full_range()
        duration = x_max - x_min
        view_width = min(duration * 0.1, 10.0)
        view_box = first_plot.plotItem.getViewBox()
        view_box.setXRange(x_max - view_width, x_max, padding=0)

    def jump_to_time(self, time: float):
        """Jump to specific timestamp and center view on it."""
        if not self.plot_widgets:
            return
        first_plot = self.plot_widgets[0]
        if first_plot.lod_renderer is None:
            return
        x_min, x_max, _, _ = first_plot.lod_renderer.get_full_range()
        time = max(x_min, min(x_max, time))
        current_x_min, current_x_max, _, _ = first_plot.get_visible_range()
        view_width = current_x_max - current_x_min
        new_x_min = time - view_width / 2
        new_x_max = time + view_width / 2
        view_box = first_plot.plotItem.getViewBox()
        view_box.setXRange(new_x_min, new_x_max, padding=0)

    def toggle_events(self):
        """Toggle event overlay visibility on all plots."""
        for overlay in self.event_overlays:
            overlay.toggle_visibility()

    def set_events_visible(self, visible: bool):
        """Set event overlay visibility on all plots."""
        for overlay in self.event_overlays:
            overlay.set_visible(visible)

    def are_events_visible(self) -> bool:
        """Check if events are currently visible."""
        return self.event_overlays[0].is_visible() if self.event_overlays else False
