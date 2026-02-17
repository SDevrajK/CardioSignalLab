"""Derived visualisation panel for the single-channel view.

Hosts one or more secondary plots (heart rate, EDA components) below the
main signal in a QVBoxLayout. All x-axes are linked to the main signal plot
so that pan/zoom is always synchronised.

Supported modes
---------------
- ``show_heart_rate``: single plot of instantaneous BPM + rolling average
- ``show_eda_components``: two stacked plots (tonic SCL / phasic SCR)

Usage
-----
    panel = DerivedPanel()
    panel.link_x_to(signal_plot_widget.plotItem)
    panel.show_heart_rate(times, bpm, rolling_bpm, signal_type="ecg")
    # ... later ...
    panel.clear()
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtWidgets import QVBoxLayout, QWidget


class DerivedPanel(QWidget):
    """Container for secondary derived plots linked to the main signal.

    All plots share the same x-axis as the main signal (set via
    ``link_x_to``). Calling any ``show_*`` method clears the previous
    content first.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._plots: list[pg.PlotWidget] = []
        self._main_plot_item = None  # PlotItem of the primary signal widget

        logger.debug("DerivedPanel initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link_x_to(self, plot_item):
        """Link all current (and future) derived plot x-axes to plot_item.

        Args:
            plot_item: pyqtgraph PlotItem of the primary signal plot
        """
        self._main_plot_item = plot_item
        for pw in self._plots:
            pw.plotItem.setXLink(plot_item)

    def show_heart_rate(
        self,
        times: np.ndarray,
        bpm: np.ndarray,
        rolling_bpm: np.ndarray,
        signal_type: str = "ecg",
    ):
        """Display instantaneous and rolling-average heart/pulse rate.

        Args:
            times: Midpoint timestamps for each inter-peak interval (s)
            bpm: Instantaneous BPM at each interval
            rolling_bpm: Rolling-average BPM
            signal_type: "ecg" or "ppg" — used for axis label only
        """
        self.clear()

        if len(times) == 0:
            logger.debug("DerivedPanel.show_heart_rate: no data to display")
            return

        rate_label = "Heart Rate" if signal_type == "ecg" else "Pulse Rate"
        color_instant = "#FF6B6B"
        color_rolling = "#FFFFFF"

        pw = self._make_plot(ylabel=rate_label, yunits="bpm")
        pw.plot(times, bpm, pen=pg.mkPen(color_instant, width=1), name="Instantaneous")
        pw.plot(times, rolling_bpm, pen=pg.mkPen(color_rolling, width=2), name="Rolling avg")

        self._add_plot(pw)
        logger.info(
            f"DerivedPanel: showing {rate_label}, "
            f"mean={bpm.mean():.1f} bpm, {len(bpm)} intervals"
        )

    def update_heart_rate(
        self,
        times: np.ndarray,
        bpm: np.ndarray,
        rolling_bpm: np.ndarray,
        signal_type: str = "ecg",
    ):
        """Refresh the HR plot without clearing/rebuilding the widget.

        Reuses the existing PlotWidget if the panel already shows HR (1 plot),
        otherwise falls back to a full rebuild via show_heart_rate.
        """
        if len(self._plots) == 1:
            pw = self._plots[0]
            pw.clear()
            if len(times) > 0:
                pw.plot(
                    times, bpm,
                    pen=pg.mkPen("#FF6B6B", width=1), name="Instantaneous"
                )
                pw.plot(
                    times, rolling_bpm,
                    pen=pg.mkPen("#FFFFFF", width=2), name="Rolling avg"
                )
        else:
            self.show_heart_rate(times, bpm, rolling_bpm, signal_type)

    def show_eda_components(
        self,
        timestamps: np.ndarray,
        tonic: np.ndarray,
        phasic: np.ndarray,
    ):
        """Display EDA tonic (SCL) and phasic (SCR) components as stacked plots.

        Args:
            timestamps: Time array matching the original EDA signal
            tonic: Tonic (SCL) component samples
            phasic: Phasic (SCR) component samples
        """
        self.clear()

        tonic_pw = self._make_plot(ylabel="SCL (Tonic)", yunits="uS")
        tonic_pw.plot(timestamps, tonic, pen=pg.mkPen("#4CAF50", width=1))

        phasic_pw = self._make_plot(ylabel="SCR (Phasic)", yunits="uS")
        phasic_pw.plot(timestamps, phasic, pen=pg.mkPen("#2196F3", width=1))

        self._add_plot(tonic_pw)
        self._add_plot(phasic_pw)
        logger.info("DerivedPanel: showing EDA tonic + phasic components")

    @property
    def has_content(self) -> bool:
        """True if at least one derived plot is currently shown."""
        return len(self._plots) > 0

    @property
    def num_plots(self) -> int:
        """Number of derived plots currently displayed."""
        return len(self._plots)

    def clear(self):
        """Remove all derived plots."""
        for pw in self._plots:
            self._layout.removeWidget(pw)
            pw.deleteLater()
        self._plots.clear()
        logger.debug("DerivedPanel cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_plot(self, ylabel: str = "", yunits: str = "") -> pg.PlotWidget:
        """Create a pre-configured PlotWidget."""
        pw = pg.PlotWidget()
        pw.setBackground("#1e1e1e")
        pw.plotItem.showGrid(x=True, y=True, alpha=0.3)
        pw.plotItem.setLabel("bottom", "Time", units="s")
        pw.plotItem.setLabel("left", ylabel, units=yunits)
        pw.plotItem.enableAutoRange(enable=True)
        view_box = pw.plotItem.getViewBox()
        view_box.setMouseMode(view_box.PanMode)
        view_box.setMouseEnabled(x=True, y=False)  # Only pan x; y auto-ranges
        return pw

    def _add_plot(self, pw: pg.PlotWidget):
        """Register a plot and link its x-axis to the main signal."""
        self._plots.append(pw)
        self._layout.addWidget(pw)
        if self._main_plot_item is not None:
            pw.plotItem.setXLink(self._main_plot_item)
