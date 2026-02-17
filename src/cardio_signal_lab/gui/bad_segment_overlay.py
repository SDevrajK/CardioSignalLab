"""Bad segment overlay for PyQtGraph plots.

Renders bad segments as semi-transparent red shaded regions using
pg.LinearRegionItem. Supports toggling visibility and updating after
detection or manual marking.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from loguru import logger
from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from cardio_signal_lab.core import BadSegment, SignalData


# Orange shading for bad segments — clearly distinct from the red event lines
_FILL_COLOR = QColor(255, 140, 0, 55)   # RGBA: semi-transparent amber
_BORDER_COLOR = QColor(200, 100, 0, 130)


class BadSegmentOverlay:
    """Overlay that renders bad segments as shaded red regions on a plot.

    Usage:
        overlay = BadSegmentOverlay(plot_widget)
        overlay.set_bad_segments(signal.bad_segments, signal)
        overlay.clear()
    """

    def __init__(self, plot_widget):
        """Initialize overlay.

        Args:
            plot_widget: SignalPlotWidget (or any pg.PlotWidget) to attach to
        """
        self.plot_widget = plot_widget
        self._visible = True
        self._regions: list[pg.LinearRegionItem] = []

        if hasattr(plot_widget, "plotItem"):
            self.plot_item = plot_widget.plotItem
        else:
            self.plot_item = plot_widget

        logger.debug("BadSegmentOverlay initialized")

    def set_bad_segments(self, bad_segments: list[BadSegment], signal: SignalData):
        """Render bad segments as shaded red regions.

        Converts sample indices to time coordinates using signal.timestamps,
        then places a LinearRegionItem for each segment.

        Args:
            bad_segments: List of BadSegment objects
            signal: SignalData used to map indices to timestamps
        """
        self.clear()

        if not bad_segments:
            return

        timestamps = signal.timestamps

        for seg in bad_segments:
            start_idx = max(0, seg.start_idx)
            end_idx = min(len(timestamps) - 1, seg.end_idx)

            t_start = float(timestamps[start_idx])
            t_end = float(timestamps[end_idx])

            region = pg.LinearRegionItem(
                values=(t_start, t_end),
                orientation="vertical",
                brush=_FILL_COLOR,
                pen=pg.mkPen(color=_BORDER_COLOR, width=1),
                movable=False,
            )
            region.setVisible(self._visible)
            # Disable z-ordering interference with signal line
            region.setZValue(-10)

            self.plot_item.addItem(region)
            self._regions.append(region)

        logger.info(
            f"BadSegmentOverlay: rendered {len(bad_segments)} segment(s) "
            f"(visible={self._visible})"
        )

    def clear(self):
        """Remove all region items from the plot."""
        for region in self._regions:
            self.plot_item.removeItem(region)
        self._regions.clear()
        logger.debug("BadSegmentOverlay cleared")

    def set_visible(self, visible: bool):
        """Show or hide all shaded regions.

        Args:
            visible: True to show, False to hide
        """
        self._visible = visible
        for region in self._regions:
            region.setVisible(visible)
        logger.debug(f"BadSegmentOverlay visibility: {visible}")

    def toggle_visibility(self):
        """Toggle between visible and hidden."""
        self.set_visible(not self._visible)

    def is_visible(self) -> bool:
        """Return current visibility state."""
        return self._visible

    def num_segments(self) -> int:
        """Return number of segments currently displayed."""
        return len(self._regions)
