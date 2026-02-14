"""Event marker overlay for PyQtGraph plots.

Renders event markers as vertical lines with text labels at event timestamps.
Supports toggling visibility and displays across all signal plots.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from loguru import logger
from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from cardio_signal_lab.core import EventData


class EventOverlay:
    """Event marker overlay for signal plots.

    Renders events as vertical lines with text labels:
    - Vertical line at event timestamp
    - Text label above the line showing event name
    - Color-coded by event type (customizable)
    """

    def __init__(self, plot_widget, parent=None):
        """Initialize event overlay.

        Args:
            plot_widget: SignalPlotWidget or PlotItem to attach markers to
            parent: Parent QObject
        """
        self.plot_widget = plot_widget
        self.visible = True

        # Storage for event items
        self.event_lines = []  # InfiniteLine items
        self.event_labels = []  # TextItem items
        self.events = []  # EventData objects

        # Get plot item
        if hasattr(plot_widget, 'plotItem'):
            self.plot_item = plot_widget.plotItem
        else:
            self.plot_item = plot_widget

        logger.debug("EventOverlay initialized")

    def set_events(self, events: list[EventData]):
        """Display events on plot.

        Args:
            events: List of EventData objects
        """
        logger.info(f"EventOverlay.set_events() called with {len(events)} events")

        # Clear existing event markers
        self.clear()
        logger.debug(f"Cleared existing markers, now have {len(self.event_lines)} lines")

        # Store a COPY of events to avoid modifying the original list
        self.events = list(events)

        if not events:
            logger.debug("No events to display")
            return

        logger.debug(f"Creating markers for {len(events)} events")
        # Create markers for each event
        for event in events:
            # Create vertical line at event timestamp
            line = pg.InfiniteLine(
                pos=event.timestamp,
                angle=90,
                pen=pg.mkPen(color='#FF6B6B', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                movable=False,
                label=None
            )

            # Create text label
            label = pg.TextItem(
                text=event.label,
                color='#FF6B6B',
                anchor=(0.5, 1.0),  # Center horizontally, anchor at bottom
                border=pg.mkPen(color='#FF6B6B', width=1),
                fill=pg.mkBrush(color=(255, 255, 255, 200))  # Semi-transparent white background
            )

            # Position label at top of plot (will be updated when plot range changes)
            label.setPos(event.timestamp, 0)

            # Add to plot
            self.plot_item.addItem(line)
            self.plot_item.addItem(label)

            # Store references
            self.event_lines.append(line)
            self.event_labels.append(label)

        logger.debug(f"Added {len(self.event_lines)} lines and {len(self.event_labels)} labels to plot")

        # Update label positions based on current plot range
        self._update_label_positions()

        # Connect to plot range changes to update label positions
        if hasattr(self.plot_item, 'getViewBox'):
            view_box = self.plot_item.getViewBox()
            # Disconnect any existing connections to avoid duplicates
            try:
                view_box.sigRangeChanged.disconnect(self._on_range_changed)
            except (TypeError, RuntimeError):
                pass  # No previous connection
            view_box.sigRangeChanged.connect(self._on_range_changed)
            logger.debug("Connected sigRangeChanged for label position updates")

        logger.info(f"Displayed {len(events)} event markers (visible={self.visible})")

    def _update_label_positions(self):
        """Update event label Y positions to appear at top of visible plot range."""
        if not self.event_labels:
            return

        # Get current Y range
        view_box = self.plot_item.getViewBox()
        if view_box is None:
            return

        _, (y_min, y_max) = view_box.viewRange()

        # Position labels at 95% of Y range (near top)
        label_y = y_min + (y_max - y_min) * 0.95

        # Update all label positions
        for event, label in zip(self.events, self.event_labels):
            label.setPos(event.timestamp, label_y)

    def _on_range_changed(self, view_box, ranges):
        """Handle plot range change - update label positions.

        Args:
            view_box: ViewBox that changed
            ranges: New ranges
        """
        self._update_label_positions()

    def set_visible(self, visible: bool):
        """Show or hide all event markers.

        Args:
            visible: True to show, False to hide
        """
        self.visible = visible

        for line in self.event_lines:
            line.setVisible(visible)

        for label in self.event_labels:
            label.setVisible(visible)

        logger.debug(f"Event overlay visibility: {visible}")

    def toggle_visibility(self):
        """Toggle event marker visibility."""
        self.set_visible(not self.visible)

    def clear(self):
        """Remove all event markers from plot."""
        # Remove from plot
        for line in self.event_lines:
            self.plot_item.removeItem(line)

        for label in self.event_labels:
            self.plot_item.removeItem(label)

        # Clear storage
        self.event_lines.clear()
        self.event_labels.clear()
        self.events.clear()

        logger.debug("EventOverlay cleared")

    def get_num_events(self) -> int:
        """Get number of events currently displayed.

        Returns:
            Number of events
        """
        return len(self.events)

    def is_visible(self) -> bool:
        """Check if events are currently visible.

        Returns:
            True if visible, False if hidden
        """
        return self.visible
