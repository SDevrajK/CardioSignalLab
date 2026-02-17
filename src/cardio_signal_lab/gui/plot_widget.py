"""Reusable PyQtGraph plot widget for single signal visualization.

Wraps PyQtGraph PlotWidget with LOD rendering for efficient display of large
signals. Automatically updates displayed data when user zooms/pans.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import Signal

from cardio_signal_lab.config import get_config
from cardio_signal_lab.gui.lod_renderer import LODRenderer

if TYPE_CHECKING:
    from cardio_signal_lab.core import SignalData


class SignalPlotWidget(pg.PlotWidget):
    """PyQtGraph plot widget for single physiological signal.

    Features:
    - LOD rendering for smooth visualization of 10M+ point signals
    - Automatic data update on zoom/pan via ViewBox range changes
    - Configurable colors, grid, and axis labels
    - Auto-ranging for initial display

    Signals:
        range_changed: Emitted when visible range changes (x_min, x_max)
    """

    range_changed = Signal(float, float)  # x_min, x_max

    def __init__(self, parent=None):
        """Initialize signal plot widget.

        Args:
            parent: Parent QWidget
        """
        super().__init__(parent=parent)

        self.config = get_config()
        self.signal_data: SignalData | None = None
        self.lod_renderer: LODRenderer | None = None
        self.plot_curve: pg.PlotDataItem | None = None
        self._updating_range = False  # Flag to prevent recursive range updates

        # Configure plot appearance
        self._setup_plot()

        # Connect to ViewBox range changes
        self.plotItem.getViewBox().sigRangeChanged.connect(self._on_range_changed)

        logger.debug("SignalPlotWidget initialized")

    def _setup_plot(self):
        """Configure plot appearance and settings."""
        plot_item = self.plotItem

        # Set background color
        self.setBackground(self.config.gui.plot_background)

        # Configure grid
        plot_item.showGrid(x=True, y=True, alpha=self.config.gui.grid_alpha)

        # Set axis labels
        plot_item.setLabel("bottom", "Time", units="s")
        plot_item.setLabel("left", "Amplitude")

        # Do NOT enable auto-range - we'll set explicit ranges when data is loaded
        # Auto-range can cause issues with non-zero-based timestamps

        # Enable right-click menu with zoom options
        plot_item.setMenuEnabled(True)

        # Configure ViewBox for better zoom behavior
        view_box = plot_item.getViewBox()
        # Set to PanMode by default (left-drag = pan, wheel = zoom)
        # Right-click menu provides "Mouse Mode > 3 button" for rectangle zoom
        view_box.setMouseMode(view_box.PanMode)
        # Allow both axes to be adjusted by mouse
        view_box.setMouseEnabled(x=True, y=True)
        # Set aspect ratio to auto (independent x/y scaling)
        view_box.setAspectLocked(False)

        # Enable 3-button mouse mode for rectangle zoom
        # Middle-button drag (or Ctrl+Left drag on some systems) creates zoom rectangle
        view_box.enableAutoRange(enable=False)

        # Customize right-click menu to make rectangle zoom more accessible
        view_box.menu.clear()
        view_box.menu.addAction("View All", lambda: view_box.autoRange())
        view_box.menu.addSeparator()

        # Add mouse mode toggle
        pan_action = view_box.menu.addAction("Pan Mode (drag to pan)")
        pan_action.triggered.connect(lambda: view_box.setMouseMode(view_box.PanMode))

        rect_action = view_box.menu.addAction("Zoom Mode (drag to zoom)")
        rect_action.triggered.connect(lambda: view_box.setMouseMode(view_box.RectMode))

        view_box.menu.addSeparator()
        view_box.menu.addAction("Export View...", lambda: view_box.export())

    def set_signal(self, signal: SignalData):
        """Load signal data and create LOD renderer.

        Args:
            signal: SignalData to display
        """
        # Block range change signals during setup to prevent infinite loop
        self._updating_range = True
        try:
            self.signal_data = signal

            # Create LOD renderer
            self.lod_renderer = LODRenderer(signal.timestamps, signal.samples, num_levels=8)

            # Choose color based on signal type
            color = self._get_signal_color(signal.signal_type)

            # Get full range for initial display
            x_min, x_max, y_min, y_max = self.lod_renderer.get_full_range()

            # Get initial render data (use reasonable pixel width estimate)
            t_data, s_data = self.lod_renderer.get_render_data(x_min, x_max, 1000)

            # Create plot curve with data
            pen = pg.mkPen(color=color, width=1)
            if self.plot_curve is None:
                self.plot_curve = self.plotItem.plot(
                    t_data, s_data,
                    pen=pen
                )
            else:
                self.plot_curve.setPen(pen)
                self.plot_curve.setData(t_data, s_data)

            # Disable auto-range to prevent feedback loops
            self.plotItem.disableAutoRange()

            # Set initial view range
            view_box = self.plotItem.getViewBox()
            view_box.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0.05, update=True)

            # Update axis labels
            unit = signal.unit if signal.unit else "a.u."
            self.plotItem.setLabel("left", "Amplitude", units=unit)
            self.plotItem.setTitle(f"{signal.signal_type.value.upper()} - {signal.channel_name}")

            logger.info(
                f"Signal loaded: {signal.signal_type.value}, {signal.channel_name}, "
                f"{self.lod_renderer.num_samples} samples, {self.lod_renderer.duration:.2f}s"
            )
        finally:
            # Re-enable range change handling after setup is complete
            self._updating_range = False

    def _get_signal_color(self, signal_type) -> str:
        """Get color for signal type from config.

        Args:
            signal_type: SignalType enum value

        Returns:
            Color string (hex or name)
        """
        from cardio_signal_lab.core import SignalType

        color_map = {
            SignalType.ECG: self.config.gui.signal_color_ecg,
            SignalType.PPG: self.config.gui.signal_color_ppg,
            SignalType.EDA: self.config.gui.signal_color_eda,
        }

        return color_map.get(signal_type, "#000000")  # Default to black

    def wheelEvent(self, event):
        """Pan left/right on scroll wheel instead of zooming.

        Scroll up (positive delta) pans earlier in time; scroll down pans later.
        Pan step is 10% of the current visible range per wheel notch.
        """
        vb = self.plotItem.getViewBox()
        (x_min, x_max), _ = vb.viewRange()
        view_width = x_max - x_min

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        # 120 units = 1 standard wheel notch; negative delta pans right
        pan_amount = 0.1 * view_width * (-delta / 120.0)
        vb.setXRange(x_min + pan_amount, x_max + pan_amount, padding=0)
        event.accept()

    def _on_range_changed(self, view_box, ranges):
        """Handle ViewBox range change (zoom/pan).

        Args:
            view_box: ViewBox that changed
            ranges: Tuple of ((x_min, x_max), (y_min, y_max))
        """
        if self.lod_renderer is None or self.plot_curve is None:
            return

        # Prevent recursive updates
        if self._updating_range:
            return

        self._updating_range = True
        try:
            # Extract x range
            x_range, _ = ranges
            x_min, x_max = x_range

            # Update plot data for new visible range
            self._update_plot_data(x_min, x_max)

            # Emit signal
            self.range_changed.emit(x_min, x_max)
        finally:
            self._updating_range = False

    def _update_plot_data(self, x_min: float, x_max: float):
        """Update plot data for visible range using LOD renderer.

        Args:
            x_min: Left edge of visible range
            x_max: Right edge of visible range
        """
        if self.lod_renderer is None or self.plot_curve is None:
            return

        # Get plot widget width in pixels
        pixel_width = self.plotItem.getViewBox().width()
        if pixel_width <= 0:
            pixel_width = 800  # Default fallback

        # Get render data from LOD renderer
        t_data, s_data = self.lod_renderer.get_render_data(x_min, x_max, int(pixel_width))

        # Update plot curve
        if len(t_data) > 0:
            self.plot_curve.setData(t_data, s_data)
        else:
            # Empty data - clear plot
            self.plot_curve.setData([], [])

    def clear(self):
        """Clear plot and reset to empty state."""
        if self.plot_curve is not None:
            self.plot_curve.setData([], [])

        self.signal_data = None
        self.lod_renderer = None

        self.plotItem.setTitle("")

        logger.debug("SignalPlotWidget cleared")

    def reset_view(self):
        """Reset view to show full signal range."""
        if self.lod_renderer is None:
            return

        x_min, x_max, y_min, y_max = self.lod_renderer.get_full_range()
        self.plotItem.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0.05)

        logger.debug(f"View reset to full range: x=[{x_min:.2f}, {x_max:.2f}]")

    def get_visible_range(self) -> tuple[float, float, float, float]:
        """Get current visible range.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        view_box = self.plotItem.getViewBox()
        (x_min, x_max), (y_min, y_max) = view_box.viewRange()
        return x_min, x_max, y_min, y_max

    def test_plot(self):
        """Create a simple test plot to verify rendering works.

        Creates a sine wave from 0-10s for debugging visualization issues.
        """
        import numpy as np

        # Create simple test data
        t = np.linspace(0, 10, 1000)
        s = np.sin(2 * np.pi * t)

        logger.info(f"Test data: t={t[:5]}..., s={s[:5]}...")

        # Clear existing plot
        if self.plot_curve is not None:
            logger.info("Removing existing plot curve")
            self.plotItem.removeItem(self.plot_curve)
            self.plot_curve = None

        # Get ViewBox
        vb = self.plotItem.getViewBox()
        logger.info(f"ViewBox: {vb}, visible: {vb.isVisible()}, size: {vb.width()}x{vb.height()}")

        # Test 1: Try adding InfiniteLine directly to ViewBox
        logger.info("TEST 1: Adding InfiniteLine directly to ViewBox...")
        line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('green', width=5))
        vb.addItem(line)
        logger.info(f"InfiniteLine added to ViewBox: {line}, visible: {line.isVisible()}, scene: {line.scene()}")

        # Test 2: Try PlotDataItem directly to ViewBox
        logger.info("TEST 2: Creating PlotDataItem and adding to ViewBox...")
        curve = pg.PlotDataItem(
            t, s,
            pen=pg.mkPen(color='red', width=5),
            name='test_curve'
        )
        vb.addItem(curve)
        self.plot_curve = curve

        logger.info(f"PlotDataItem added to ViewBox: {curve}")
        logger.info(f"  visible: {curve.isVisible()}")
        logger.info(f"  in scene: {curve.scene() is not None}")
        logger.info(f"  data points: {len(curve.xData) if curve.xData is not None else 0}")

        # Test 3: Check all items in ViewBox
        all_items = vb.allChildren()
        logger.info(f"All items in ViewBox: {len(all_items)} items")
        for i, item in enumerate(all_items[:10]):  # Log first 10
            logger.info(f"  Item {i}: {type(item).__name__}, visible: {item.isVisible()}")

        # Set view to show test data
        self.plotItem.setRange(xRange=(0, 10), yRange=(-1.5, 1.5), padding=0)

        # Try to force update
        self.plotItem.update()
        self.update()

        # Check ViewBox state
        vb = self.plotItem.getViewBox()
        logger.info(f"ViewBox range after test: {vb.viewRange()}")
        logger.info(f"ViewBox size: {vb.width()}x{vb.height()}")

        logger.info("Test plot creation complete")
