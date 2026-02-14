"""Signal-type view showing all channels of one signal type.

Intermediate view between multi-signal overview and single-channel processing.
Displays channels as stacked plots with synchronized x-axes. Supports creating
derived channels (L2 Norm) from multiple channels.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import DerivedSignalData, SignalData, SignalType


class SignalTypeView(QWidget):
    """Signal-type view showing all channels of one signal type.

    Displays all channels in stacked plots with synchronized x-axes.
    Click on a channel to drill down to single-channel processing view.

    Signals:
        channel_selected: Emitted when user clicks on a channel plot (SignalData)
        return_to_multi_requested: Emitted when user wants to go back to multi view
    """

    channel_selected = Signal(object)  # SignalData
    return_to_multi_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.signal_type: SignalType | None = None
        self.signals: list[SignalData] = []
        self.derived_signals: list[DerivedSignalData] = []
        self.plot_widgets: list[SignalPlotWidget] = []
        self.event_overlays: list[EventOverlay] = []
        self.session_events: list = []

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # Toolbar for derived channel creation
        self.toolbar = QWidget()
        toolbar_layout = QHBoxLayout(self.toolbar)
        toolbar_layout.setContentsMargins(5, 2, 5, 2)

        self.l2_norm_button = QPushButton("Create L2 Norm Channel")
        self.l2_norm_button.clicked.connect(self._on_create_l2_norm)
        self.l2_norm_button.setVisible(False)  # Hidden until >=2 channels
        toolbar_layout.addWidget(self.l2_norm_button)
        toolbar_layout.addStretch()

        main_layout.addWidget(self.toolbar)

        # Plot area
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(5)
        main_layout.addWidget(self.plot_container, stretch=1)

        logger.debug("SignalTypeView initialized")

    def set_signal_type(self, signal_type: SignalType, signals: list[SignalData]):
        """Load channels for a signal type and create stacked plots.

        Args:
            signal_type: The signal type being viewed
            signals: List of SignalData channels for this type
        """
        self.signal_type = signal_type
        self.signals = list(signals)
        self.derived_signals = []

        self.clear_plots()

        # Show L2 Norm button only when >=2 channels
        self.l2_norm_button.setVisible(len(signals) >= 2)

        # Create one plot per channel
        for signal in signals:
            self._add_channel_plot(signal)

        # Link x-axes
        if len(self.plot_widgets) > 1:
            first_plot = self.plot_widgets[0]
            for plot_widget in self.plot_widgets[1:]:
                plot_widget.plotItem.setXLink(first_plot.plotItem)

        # Add event overlays
        if self.session_events:
            for plot_widget in self.plot_widgets:
                event_overlay = EventOverlay(plot_widget)
                event_overlay.set_events(self.session_events)
                self.event_overlays.append(event_overlay)

        logger.info(
            f"Signal type view loaded: {signal_type.value.upper()} "
            f"with {len(signals)} channels"
        )

    def _add_channel_plot(self, signal: SignalData):
        """Add a channel plot to the stacked view.

        Args:
            signal: SignalData for this channel
        """
        plot_widget = SignalPlotWidget()
        plot_widget.set_signal(signal)

        # Click handler for channel selection
        plot_widget.plotItem.scene().sigMouseClicked.connect(
            lambda event, sig=signal: self._on_channel_clicked(event, sig)
        )

        self.plot_widgets.append(plot_widget)
        self.plot_layout.addWidget(plot_widget)

    def _on_channel_clicked(self, event, signal: SignalData):
        """Handle click on a channel plot."""
        logger.info(f"Channel selected: {signal.channel_name}")
        self.channel_selected.emit(signal)
        self.app_signals.signal_selected.emit(signal)

    def _on_create_l2_norm(self):
        """Create L2 Norm derived channel from all current channels."""
        from cardio_signal_lab.core import create_l2_norm

        if len(self.signals) < 2:
            return

        try:
            derived = create_l2_norm(self.signals)
            self.derived_signals.append(derived)

            # Add as a new plot
            # Create a SignalData-compatible wrapper for plotting
            from cardio_signal_lab.core import SignalData
            plot_signal = SignalData(
                samples=derived.samples,
                sampling_rate=derived.sampling_rate,
                timestamps=derived.timestamps,
                channel_name=derived.channel_name,
                signal_type=derived.signal_type,
            )

            plot_widget = SignalPlotWidget()
            plot_widget.set_signal(plot_signal)

            # Click handler
            plot_widget.plotItem.scene().sigMouseClicked.connect(
                lambda event, sig=plot_signal: self._on_channel_clicked(event, sig)
            )

            # Link x-axis to first plot
            if self.plot_widgets:
                plot_widget.plotItem.setXLink(self.plot_widgets[0].plotItem)

            self.plot_widgets.append(plot_widget)
            self.plot_layout.addWidget(plot_widget)

            # Add event overlay if events exist
            if self.session_events:
                event_overlay = EventOverlay(plot_widget)
                event_overlay.set_events(self.session_events)
                self.event_overlays.append(event_overlay)

            logger.info(f"Created L2 Norm derived channel from {len(self.signals)} channels")
        except Exception as e:
            logger.error(f"Failed to create L2 Norm: {e}")

    def set_events(self, events: list):
        """Set events to display on all channel plots.

        Args:
            events: List of EventData objects
        """
        self.session_events = events
        # Update existing overlays
        for overlay in self.event_overlays:
            overlay.set_events(events)

    def clear_plots(self):
        """Clear all channel plots."""
        for overlay in self.event_overlays:
            overlay.clear()
        self.event_overlays.clear()

        for plot_widget in self.plot_widgets:
            self.plot_layout.removeWidget(plot_widget)
            plot_widget.deleteLater()
        self.plot_widgets.clear()

    def clear(self):
        """Full clear including signal data."""
        self.clear_plots()
        self.signal_type = None
        self.signals = []
        self.derived_signals = []

    def reset_view(self):
        """Reset all plots to show full signal range."""
        for plot_widget in self.plot_widgets:
            plot_widget.reset_view()

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
