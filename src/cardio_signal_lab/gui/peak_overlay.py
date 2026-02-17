"""Peak marker overlay for PyQtGraph plots.

Renders peak markers on signal plots with color coding based on classification.
Supports efficient add/remove operations and click events for peak selection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import Signal

from cardio_signal_lab.config import get_config
from cardio_signal_lab.core import PeakClassification

if TYPE_CHECKING:
    from cardio_signal_lab.core import PeakData, SignalData


class PeakOverlay:
    """Peak marker overlay for signal plots.

    Renders peaks as scatter plot markers with color coding by classification:
    - Blue: Auto-detected (AUTO)
    - Green: Manually-added (MANUAL)
    - Orange: Ectopic beat (ECTOPIC)
    - Red: Bad/artifact (BAD)
    - Yellow: Selected peak highlight

    Signals:
        peak_clicked: Emitted when user clicks on a peak (peak_index)
    """

    # Marker size bounds in pixels
    _SIZE_MIN = 4
    _SIZE_MAX = 12
    _SIZE_DEFAULT = 10

    def __init__(self, plot_widget, parent=None):
        """Initialize peak overlay.

        Args:
            plot_widget: SignalPlotWidget or PlotItem to attach markers to
            parent: Parent QObject for signals
        """
        self.plot_widget = plot_widget
        self.config = get_config()

        # Main scatter: all peaks rendered by classification color.
        # Only updated when peak data changes (add/delete/classify/reset).
        self.scatter = pg.ScatterPlotItem()
        self.scatter.setSize(self._SIZE_DEFAULT)
        self.scatter.setPxMode(True)

        # Selection scatter: a single point drawn on top to highlight the selected peak.
        # Updated on every navigation step — touching only one point is far cheaper than
        # re-rendering the full scatter on each arrow-key press.
        self._selected_scatter = pg.ScatterPlotItem()
        self._selected_scatter.setSize(self._SIZE_DEFAULT + 2)
        self._selected_scatter.setPxMode(True)

        # Add to plot (selection scatter on top)
        if hasattr(plot_widget, 'plotItem'):
            plot_widget.plotItem.addItem(self.scatter)
            plot_widget.plotItem.addItem(self._selected_scatter)
            self._viewbox = plot_widget.plotItem.getViewBox()
        else:
            plot_widget.addItem(self.scatter)
            plot_widget.addItem(self._selected_scatter)
            self._viewbox = plot_widget.getViewBox()

        # Data storage
        self.signal_data: SignalData | None = None
        self.peak_data: PeakData | None = None
        self.selected_index: int | None = None
        self._full_x_range: float = 0.0  # Signal duration for zoom-ratio sizing

        # Connect signals
        self.scatter.sigClicked.connect(self._on_peak_clicked)
        self._viewbox.sigRangeChanged.connect(self._on_range_changed)

        logger.debug("PeakOverlay initialized")

    def _on_range_changed(self, view_box, ranges):
        """Adjust marker size based on zoom level.

        Markers scale from _SIZE_MIN (fully zoomed out) to _SIZE_MAX (zoomed in)
        using a log10 relationship so the transition feels natural.
        """
        if self._full_x_range <= 0:
            return
        x_range, _ = ranges
        current_range = x_range[1] - x_range[0]
        if current_range <= 0:
            return
        ratio = self._full_x_range / current_range
        size = int(np.clip(
            self._SIZE_MIN + np.log10(max(ratio, 1.0)) * 4,
            self._SIZE_MIN,
            self._SIZE_MAX,
        ))
        self.scatter.setSize(size)
        self._selected_scatter.setSize(min(size + 2, self._SIZE_MAX))

    def set_peaks(self, signal: SignalData, peaks: PeakData):
        """Display peaks on signal.

        Args:
            signal: SignalData containing signal values
            peaks: PeakData containing peak indices and classifications
        """
        self.signal_data = signal
        self.peak_data = peaks
        self._full_x_range = float(signal.timestamps[-1] - signal.timestamps[0])

        # Set initial size based on current view
        (x_min, x_max), _ = self._viewbox.viewRange()
        self._on_range_changed(self._viewbox, ((x_min, x_max), (0, 1)))

        # Clear selection overlay — peak data is being replaced
        self._selected_scatter.setData([], [])
        self.selected_index = None

        if peaks.num_peaks == 0:
            self.scatter.setData([], [])
            logger.debug("No peaks to display")
            return

        peak_times = signal.timestamps[peaks.indices]
        peak_amps = signal.samples[peaks.indices]
        colors = self._get_peak_colors(peaks.classifications)

        self.scatter.setData(
            x=peak_times,
            y=peak_amps,
            brush=colors,
            symbol='o',
            data=np.arange(len(peaks.indices))
        )

        logger.info(
            f"Displayed {peaks.num_peaks} peaks "
            f"(auto={peaks.num_auto}, manual={peaks.num_manual}, "
            f"ectopic={peaks.num_ectopic}, bad={peaks.num_bad})"
        )

    def _get_peak_colors(self, classifications: np.ndarray) -> list:
        """Get marker colors based on peak classifications.

        Args:
            classifications: Array of PeakClassification values

        Returns:
            List of brush objects for PyQtGraph
        """
        colors = []
        for classification in classifications:
            if classification == PeakClassification.AUTO.value:
                colors.append(pg.mkBrush(self.config.gui.peak_color_auto))
            elif classification == PeakClassification.MANUAL.value:
                colors.append(pg.mkBrush(self.config.gui.peak_color_manual))
            elif classification == PeakClassification.ECTOPIC.value:
                colors.append(pg.mkBrush(self.config.gui.peak_color_ectopic))
            elif classification == PeakClassification.BAD.value:
                colors.append(pg.mkBrush(self.config.gui.peak_color_bad))
            else:
                # Unknown classification, use default
                colors.append(pg.mkBrush('gray'))

        return colors

    def add_peak(self, peak_index: int, classification: PeakClassification = PeakClassification.MANUAL):
        """Add a single peak marker.

        Args:
            peak_index: Index in signal.samples where peak occurs
            classification: Peak classification (default: MANUAL)
        """
        if self.signal_data is None or self.peak_data is None:
            logger.warning("Cannot add peak: no signal/peak data loaded")
            return

        # Add to peak data
        self.peak_data.indices = np.append(self.peak_data.indices, peak_index)
        self.peak_data.classifications = np.append(self.peak_data.classifications, classification.value)

        # Refresh display
        self.set_peaks(self.signal_data, self.peak_data)

        logger.debug(f"Added peak at index {peak_index}, classification={classification.name}")

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
        self.peak_data.classifications = self.peak_data.classifications[mask]

        # Clear selection if removing selected peak
        if self.selected_index == peak_index:
            self.selected_index = None

        # Refresh display
        self.set_peaks(self.signal_data, self.peak_data)

        logger.debug(f"Removed peak at index {peak_index}")

    def select_peak(self, peak_index: int | None):
        """Select a peak by highlighting it with the selected color.

        Only updates the lightweight _selected_scatter (one point) so that
        arrow-key navigation does not trigger a full re-render of all peaks.

        Args:
            peak_index: Index in peak array to select, or None to clear selection
        """
        self.selected_index = peak_index

        if peak_index is None:
            self._selected_scatter.setData([], [])
            logger.debug("Peak selection cleared")
            return

        if self.signal_data is None or self.peak_data is None:
            return

        sample = self.peak_data.indices[peak_index]
        t = float(self.signal_data.timestamps[sample])
        amp = float(self.signal_data.samples[sample])

        self._selected_scatter.setData(
            x=[t],
            y=[amp],
            brush=pg.mkBrush(self.config.gui.peak_color_selected),
            symbol='o',
            data=[peak_index],
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
        self._selected_scatter.setData([], [])
        self.signal_data = None
        self.peak_data = None
        self.selected_index = None
        self._full_x_range = 0.0

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
