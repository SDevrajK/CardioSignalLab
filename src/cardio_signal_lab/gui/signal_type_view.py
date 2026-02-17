"""Signal-type view showing all channels of one signal type.

Intermediate view between multi-signal overview and single-channel processing.
Displays channels as stacked plots with synchronized x-axes. Supports creating
derived channels (L2 Norm) from multiple channels via the Process menu.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import SignalData, SignalType


class SignalTypeView(QWidget):
    """Signal-type view showing all channels of one signal type.

    Displays all channels in stacked plots with synchronized x-axes.
    Click on a channel (left-click) to drill down to single-channel view.
    Derived channels (e.g. L2 Norm) created via Process menu persist when
    navigating back from single-channel view.

    Signals:
        channel_selected: Emitted when user left-clicks on a channel plot (SignalData)
        return_to_multi_requested: Emitted when user wants to go back to multi view
    """

    channel_selected = Signal(object)  # SignalData
    return_to_multi_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.signal_type: SignalType | None = None
        self.signals: list[SignalData] = []
        # Derived channels are stored as SignalData for consistent plotting and persistence
        self.derived_signals: list[SignalData] = []
        self.plot_widgets: list[SignalPlotWidget] = []
        self.event_overlays: list[EventOverlay] = []
        self.session_events: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(5)
        layout.addWidget(self.plot_container, stretch=1)

        logger.debug("SignalTypeView initialized")

    def set_signal_type(self, signal_type: SignalType, signals: list[SignalData]):
        """Load channels for a signal type and create stacked plots.

        Preserves any derived channels (e.g. L2 Norm) if the signal type is
        the same as the previous call (i.e. navigating back from channel view).

        Args:
            signal_type: The signal type being viewed
            signals: List of SignalData channels for this type
        """
        # Preserve derived channels when returning to the same signal type
        prev_derived = list(self.derived_signals) if self.signal_type == signal_type else []

        self.signal_type = signal_type
        self.signals = list(signals)
        self.derived_signals = []

        self.clear_plots()

        for signal in signals:
            self._add_channel_plot(signal)

        if len(self.plot_widgets) > 1:
            first_plot = self.plot_widgets[0]
            for plot_widget in self.plot_widgets[1:]:
                plot_widget.plotItem.setXLink(first_plot.plotItem)

        if self.session_events:
            for plot_widget in self.plot_widgets:
                event_overlay = EventOverlay(plot_widget)
                event_overlay.set_events(self.session_events)
                self.event_overlays.append(event_overlay)

        # Re-add any derived channels that existed before navigation
        for derived_signal in prev_derived:
            self._add_derived_plot(derived_signal)
            self.derived_signals.append(derived_signal)

        logger.info(
            f"Signal type view loaded: {signal_type.value.upper()} "
            f"with {len(signals)} channels"
            + (f" + {len(prev_derived)} derived" if prev_derived else "")
        )

    def _add_channel_plot(self, signal: SignalData):
        """Add a base channel plot.

        Args:
            signal: SignalData for this channel
        """
        plot_widget = SignalPlotWidget()
        plot_widget.set_signal(signal)
        plot_widget.plotItem.scene().sigMouseClicked.connect(
            lambda event, sig=signal: self._on_channel_clicked(event, sig)
        )
        self.plot_widgets.append(plot_widget)
        self.plot_layout.addWidget(plot_widget)

    def _add_derived_plot(self, signal: SignalData):
        """Add a derived channel plot (e.g. L2 Norm).

        Args:
            signal: SignalData wrapper for the derived channel
        """
        plot_widget = SignalPlotWidget()
        plot_widget.set_signal(signal)
        plot_widget.plotItem.scene().sigMouseClicked.connect(
            lambda event, sig=signal: self._on_channel_clicked(event, sig)
        )

        if self.plot_widgets:
            plot_widget.plotItem.setXLink(self.plot_widgets[0].plotItem)

        self.plot_widgets.append(plot_widget)
        self.plot_layout.addWidget(plot_widget)

        if self.session_events:
            event_overlay = EventOverlay(plot_widget)
            event_overlay.set_events(self.session_events)
            self.event_overlays.append(event_overlay)

    def add_l2_norm(self, selected_signals: list[SignalData]) -> SignalData:
        """Create L2 Norm derived channel from the given signals and add it to the view.

        Args:
            selected_signals: Subset of self.signals to include in the norm

        Returns:
            The SignalData wrapper for the derived channel
        """
        from cardio_signal_lab.core import create_l2_norm, SignalData as SD

        derived = create_l2_norm(selected_signals)

        # Wrap as a plain SignalData for plotting and persistence
        plot_signal = SD(
            samples=derived.samples,
            sampling_rate=derived.sampling_rate,
            timestamps=derived.timestamps,
            channel_name=derived.channel_name,
            signal_type=derived.signal_type,
        )

        self.derived_signals.append(plot_signal)
        self._add_derived_plot(plot_signal)

        logger.info(f"L2 Norm created from {len(selected_signals)} channels")
        return plot_signal

    def _on_channel_clicked(self, event, signal: SignalData):
        """Handle left-click on a channel plot - select that channel."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        logger.info(f"Channel selected: {signal.channel_name}")
        self.channel_selected.emit(signal)
        self.app_signals.signal_selected.emit(signal)

    def set_events(self, events: list):
        """Set events to display on all channel plots.

        Args:
            events: List of EventData objects
        """
        self.session_events = events
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
            plot_widget.plotItem.getViewBox().scaleBy((0.5, 0.5))

    def zoom_out(self):
        """Zoom out on all plots."""
        for plot_widget in self.plot_widgets:
            plot_widget.plotItem.getViewBox().scaleBy((2.0, 2.0))

    def jump_to_start(self):
        """Jump to start of all signals."""
        if not self.plot_widgets:
            return
        first_plot = self.plot_widgets[0]
        if first_plot.lod_renderer is None:
            return
        x_min, x_max, _, _ = first_plot.lod_renderer.get_full_range()
        view_width = min((x_max - x_min) * 0.1, 10.0)
        first_plot.plotItem.getViewBox().setXRange(x_min, x_min + view_width, padding=0)

    def jump_to_end(self):
        """Jump to end of all signals."""
        if not self.plot_widgets:
            return
        first_plot = self.plot_widgets[0]
        if first_plot.lod_renderer is None:
            return
        x_min, x_max, _, _ = first_plot.lod_renderer.get_full_range()
        view_width = min((x_max - x_min) * 0.1, 10.0)
        first_plot.plotItem.getViewBox().setXRange(x_max - view_width, x_max, padding=0)

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
        first_plot.plotItem.getViewBox().setXRange(
            time - view_width / 2, time + view_width / 2, padding=0
        )

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
