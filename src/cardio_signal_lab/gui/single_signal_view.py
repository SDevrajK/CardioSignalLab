"""Single-signal view for detailed signal processing and correction.

Displays one signal prominently in a full-size plot with interactive features
for peak correction (clicking to add/delete peaks). Provides zoom/pan via
mouse wheel and drag.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from cardio_signal_lab.gui.event_overlay import EventOverlay
from cardio_signal_lab.gui.plot_widget import SignalPlotWidget
from cardio_signal_lab.signals import get_app_signals

if TYPE_CHECKING:
    from cardio_signal_lab.core import SignalData


class SingleSignalView(QWidget):
    """Single-signal view for detailed processing and peak correction.

    Displays one signal in a full-size plot with interactive features:
    - Zoom/pan with mouse wheel and drag
    - Click to add/delete peaks (to be implemented in peak correction task)
    - Keyboard shortcuts for navigation

    Signals:
        return_to_multi_requested: Emitted when user wants to return to multi-signal mode
    """

    return_to_multi_requested = Signal()

    def __init__(self, parent=None):
        """Initialize single-signal view.

        Args:
            parent: Parent QWidget
        """
        super().__init__(parent=parent)

        self.app_signals = get_app_signals()
        self.signal_data: SignalData | None = None
        self.event_overlay: EventOverlay | None = None  # Event overlay for this plot
        self.session_events: list = []  # Store session events

        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget (will occupy full size)
        self.plot_widget = SignalPlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Create event overlay
        self.event_overlay = EventOverlay(self.plot_widget)

        # Connect signals
        self._connect_signals()

        logger.debug("SingleSignalView initialized")

    def _connect_signals(self):
        """Connect to app signals."""
        # Will add peak correction signals later
        pass

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
