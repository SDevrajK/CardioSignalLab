"""Peak marker overlay for PyQtGraph plots.

Renders peak markers on signal plots with color coding for auto-detected vs
manually-added peaks. Supports efficient add/remove operations and click
events for peak selection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import Signal

from cardio_signal_lab.config import get_config

if TYPE_CHECKING:
    from cardio_signal_lab.core import PeakData, SignalData


class PeakOverlay:
    """Peak marker overlay for signal plots.

    Renders peaks as scatter plot markers with color coding:
    - Blue: Auto-detected peaks
    - Green: Manually-added peaks
    - Orange: Selected peak

    Signals:
        peak_clicked: Emitted when user clicks on a peak (peak_index)
        peak_hovered: Emitted when user hovers over a peak (peak_index)
    """

    def __init__(self, plot_widget, parent=None):
        """Initialize peak overlay.

        Args:
            plot_widget: SignalPlotWidget or PlotItem to attach markers to
            parent: Parent QObject for signals
        """
        self.plot_widget = plot_widget
        self.config = get_config()

        # Create scatter plot item for markers
        self.scatter = pg.ScatterPlotItem()
        self.scatter.setSize(10)
        self.scatter.setPxMode(True)  # Size in pixels, not data coordinates

        # Add to plot
        if hasattr(plot_widget, 'plotItem'):
            plot_widget.plotItem.addItem(self.scatter)
        else:
            plot_widget.addItem(self.scatter)

        # Data storage
        self.signal_data: SignalData | None = None
        self.peak_data: PeakData | None = None
        self.selected_index: int | None = None

        # Connect signals
        self.scatter.sigClicked.connect(self._on_peak_clicked)

        logger.debug("PeakOverlay initialized")

    def set_peaks(self, signal: SignalData, peaks: PeakData):
        """Display peaks on signal.

        Args:
            signal: SignalData containing signal values
            peaks: PeakData containing peak indices and sources
        """
        self.signal_data = signal
        self.peak_data = peaks

        if peaks.num_peaks == 0:
            # No peaks to display
            self.scatter.setData([], [])
            logger.debug("No peaks to display")
            return

        # Get peak times and amplitudes
        peak_times = signal.timestamps[peaks.indices]
        peak_amps = signal.samples[peaks.indices]

        # Get colors based on source (0=auto, 1=manual)
        colors = self._get_peak_colors(peaks.sources)

        # Update scatter plot
        self.scatter.setData(
            x=peak_times,
            y=peak_amps,
            brush=colors,
            symbol='o',
            data=np.arange(len(peaks.indices))  # Store indices for click handling
        )

        logger.info(f"Displayed {peaks.num_peaks} peaks ({peaks.num_auto} auto, {peaks.num_manual} manual)")

    def _get_peak_colors(self, sources: np.ndarray) -> list:
        """Get marker colors based on peak sources.

        Args:
            sources: Array of peak sources (0=auto, 1=manual)

        Returns:
            List of QColor or color strings
        """
        colors = []
        for source in sources:
            if source == 0:
                # Auto-detected peak
                colors.append(pg.mkBrush(self.config.gui.peak_color_auto))
            else:
                # Manually-added peak
                colors.append(pg.mkBrush(self.config.gui.peak_color_manual))

        return colors

    def add_peak(self, peak_index: int, source: int):
        """Add a single peak marker.

        Args:
            peak_index: Index in signal.samples where peak occurs
            source: Peak source (0=auto, 1=manual)
        """
        if self.signal_data is None or self.peak_data is None:
            logger.warning("Cannot add peak: no signal/peak data loaded")
            return

        # Add to peak data
        self.peak_data.indices = np.append(self.peak_data.indices, peak_index)
        self.peak_data.sources = np.append(self.peak_data.sources, source)

        # Refresh display
        self.set_peaks(self.signal_data, self.peak_data)

        logger.debug(f"Added peak at index {peak_index}, source={source}")

    def remove_peak(self, peak_index: int):
        """Remove peak marker by its index in the peak array.

        Args:
            peak_index: Index in peak_data.indices array (not signal sample index)
        """
        if self.peak_data is None or peak_index >= self.peak_data.num_peaks:
            logger.warning(f"Cannot remove peak: invalid index {peak_index}")
            return

        # Remove from peak data
        mask = np.ones(self.peak_data.num_peaks, dtype=bool)
        mask[peak_index] = False

        self.peak_data.indices = self.peak_data.indices[mask]
        self.peak_data.sources = self.peak_data.sources[mask]

        # Clear selection if removing selected peak
        if self.selected_index == peak_index:
            self.selected_index = None

        # Refresh display
        self.set_peaks(self.signal_data, self.peak_data)

        logger.debug(f"Removed peak at index {peak_index}")

    def select_peak(self, peak_index: int | None):
        """Select a peak (highlight it).

        Args:
            peak_index: Index in peak array to select, or None to clear selection
        """
        self.selected_index = peak_index

        if self.peak_data is None or self.signal_data is None:
            return

        if peak_index is None:
            # Clear selection - restore normal colors
            self.set_peaks(self.signal_data, self.peak_data)
            logger.debug("Peak selection cleared")
            return

        # Highlight selected peak
        peak_times = self.signal_data.timestamps[self.peak_data.indices]
        peak_amps = self.signal_data.samples[self.peak_data.indices]

        colors = self._get_peak_colors(self.peak_data.sources)
        # Override selected peak color
        colors[peak_index] = pg.mkBrush(self.config.gui.peak_color_selected)

        self.scatter.setData(
            x=peak_times,
            y=peak_amps,
            brush=colors,
            symbol='o',
            data=np.arange(len(self.peak_data.indices))
        )

        logger.debug(f"Selected peak {peak_index}")

    def _on_peak_clicked(self, scatter, points):
        """Handle click on peak marker.

        Args:
            scatter: ScatterPlotItem that was clicked
            points: List of clicked SpotItems
        """
        if not points:
            return

        # Get first clicked point
        point = points[0]
        peak_index = point.data()

        logger.info(f"Peak clicked: index {peak_index}")

        # For now, just select it
        # Peak correction handler will connect to this and handle deletion
        self.select_peak(peak_index)

    def clear(self):
        """Clear all peak markers."""
        self.scatter.setData([], [])
        self.signal_data = None
        self.peak_data = None
        self.selected_index = None

        logger.debug("PeakOverlay cleared")

    def get_num_peaks(self) -> int:
        """Get number of peaks currently displayed.

        Returns:
            Number of peaks
        """
        return self.peak_data.num_peaks if self.peak_data else 0

    def get_selected_index(self) -> int | None:
        """Get index of currently selected peak.

        Returns:
            Peak index or None if no selection
        """
        return self.selected_index
