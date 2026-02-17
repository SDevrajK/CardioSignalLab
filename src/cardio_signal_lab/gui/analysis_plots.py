"""Analysis plot dialogs for ECG/PPG signals.

Provides two standalone plot dialogs:

- HeartbeatOverlayDialog: all detected heartbeats overlaid on a single
  plot (epoch view), useful for spotting morphology outliers.

- RRHistogramDialog: histogram of consecutive RR intervals with vertical
  lines marking the physiological (300/2000 ms) and MAD-based statistical
  filter boundaries used by the NN-interval exporter.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cardio_signal_lab.core.data_models import PeakData, SignalData
from cardio_signal_lab.processing.interval_analysis import (
    flag_physiological,
    flag_statistical,
)


# ---------------------------------------------------------------------------
# Default epoch windows (ms relative to peak)
# ---------------------------------------------------------------------------
_ECG_PRE_MS = 200.0
_ECG_POST_MS = 400.0
_PPG_PRE_MS = 300.0
_PPG_POST_MS = 600.0

# Default physiological filter bounds (same as interval_analysis defaults)
_PHYS_MIN_MS = 300.0
_PHYS_MAX_MS = 2000.0
_STAT_THRESHOLD = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rr_ms(signal: SignalData, peaks: PeakData) -> np.ndarray:
    """Return consecutive inter-peak intervals in ms, sorted by sample index."""
    if peaks.num_peaks < 2:
        return np.array([])
    sort_order = np.argsort(peaks.indices)
    sorted_times = signal.timestamps[peaks.indices[sort_order]]
    return np.diff(sorted_times) * 1000.0


def _extract_epochs(
    signal: SignalData,
    peaks: PeakData,
    pre_ms: float,
    post_ms: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract fixed-length epochs around each peak.

    Returns:
        time_axis: 1-D array of relative times in ms (same length for all epochs)
        epochs: list of 1-D amplitude arrays; epochs with insufficient data are
                silently excluded
    """
    sr = signal.sampling_rate
    pre_samples = int(round(pre_ms / 1000.0 * sr))
    post_samples = int(round(post_ms / 1000.0 * sr))
    n_epoch = pre_samples + post_samples
    n_samples = len(signal.samples)

    time_axis = np.linspace(-pre_ms, post_ms, n_epoch)
    epochs = []

    for idx in peaks.indices:
        start = idx - pre_samples
        end = idx + post_samples
        if start < 0 or end > n_samples:
            continue  # Skip edge peaks without enough context
        epochs.append(signal.samples[start:end].copy())

    return time_axis, epochs


# ---------------------------------------------------------------------------
# Heartbeat Overlay Dialog
# ---------------------------------------------------------------------------

class HeartbeatOverlayDialog(QDialog):
    """Show all detected heartbeats overlaid on a single epoch plot.

    Each beat is time-locked to its detected peak (t=0).  A median beat
    computed across all valid epochs is drawn as a bright white line on top.

    Args:
        signal: Current SignalData (must have .samples, .timestamps,
                .sampling_rate, .signal_type)
        peaks: Detected PeakData
        parent: Qt parent widget
    """

    def __init__(
        self,
        signal: SignalData,
        peaks: PeakData,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Heartbeat Overlay")
        self.resize(800, 500)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint
        )

        # Determine epoch window based on signal type
        sig_type_str = signal.signal_type.value.lower()
        if sig_type_str == "ppg":
            pre_ms, post_ms = _PPG_PRE_MS, _PPG_POST_MS
            x_label = "Time relative to pulse peak (ms)"
            title = f"Pulse Overlay  ({peaks.num_peaks} beats)"
        else:  # ECG or unknown
            pre_ms, post_ms = _ECG_PRE_MS, _ECG_POST_MS
            x_label = "Time relative to R-peak (ms)"
            title = f"Heartbeat Overlay  ({peaks.num_peaks} beats)"

        time_axis, epochs = _extract_epochs(signal, peaks, pre_ms, post_ms)
        n_used = len(epochs)

        # ---- Layout ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        if n_used == 0:
            layout.addWidget(QLabel(
                "Not enough data to extract epochs.\n"
                "Make sure peaks are not at the signal edges."
            ))
        else:
            pw = pg.PlotWidget(title=title)
            pw.setBackground("#1e1e1e")
            pw.plotItem.showGrid(x=True, y=True, alpha=0.3)
            pw.plotItem.setLabel("bottom", x_label)
            pw.plotItem.setLabel("left", "Amplitude")
            pw.plotItem.enableAutoRange(enable=True)

            # Individual beats: thin, semi-transparent
            alpha = max(15, min(80, int(200 / n_used)))
            beat_color = QColor(100, 180, 255, alpha)
            beat_pen = pg.mkPen(beat_color, width=1)
            for epoch in epochs:
                pw.plot(time_axis, epoch, pen=beat_pen)

            # Median beat: bold white
            median_beat = np.median(np.array(epochs), axis=0)
            pw.plot(
                time_axis, median_beat,
                pen=pg.mkPen("#FFFFFF", width=2),
                name="Median beat",
            )

            # Vertical line at t=0 (peak)
            peak_line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen("#FF6B6B", width=1, style=Qt.PenStyle.DashLine),
                label="peak",
                labelOpts={"color": "#FF6B6B", "position": 0.92},
            )
            pw.addItem(peak_line)

            layout.addWidget(pw)

            info = QLabel(
                f"{n_used} of {peaks.num_peaks} beats plotted  "
                f"(edge beats excluded)  |  window: -{pre_ms:.0f} to +{post_ms:.0f} ms"
            )
            info.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(info)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        logger.info(
            f"HeartbeatOverlayDialog: {n_used}/{peaks.num_peaks} epochs "
            f"plotted (pre={pre_ms:.0f}ms, post={post_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# RR Interval Histogram Dialog
# ---------------------------------------------------------------------------

class RRHistogramDialog(QDialog):
    """Histogram of consecutive RR intervals with filter boundary lines.

    Vertical lines drawn on the histogram:
    - Red dashed: physiological bounds (default 300 ms / 2000 ms)
    - Orange dashed: statistical bounds (median +/- threshold * MAD)

    Args:
        signal: Current SignalData
        peaks: Detected PeakData
        phys_min_ms: Physiological minimum interval (ms)
        phys_max_ms: Physiological maximum interval (ms)
        stat_threshold: MAD multiplier for statistical filter
        parent: Qt parent widget
    """

    def __init__(
        self,
        signal: SignalData,
        peaks: PeakData,
        phys_min_ms: float = _PHYS_MIN_MS,
        phys_max_ms: float = _PHYS_MAX_MS,
        stat_threshold: float = _STAT_THRESHOLD,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("RR Interval Histogram")
        self.resize(800, 520)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint
        )

        rr_ms = _compute_rr_ms(signal, peaks)
        n_intervals = len(rr_ms)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        if n_intervals == 0:
            layout.addWidget(QLabel(
                "Fewer than 2 peaks detected.\n"
                "No intervals to display."
            ))
        else:
            # Histogram bins: Freedman-Diaconis gives finer bins than "auto"
            counts, bin_edges = np.histogram(rr_ms, bins="fd")
            bin_width = float(bin_edges[1] - bin_edges[0])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            # Validity flags (for status summary)
            phys_valid = flag_physiological(
                rr_ms, min_ms=phys_min_ms, max_ms=phys_max_ms
            )
            stat_valid = flag_statistical(rr_ms, threshold=stat_threshold)

            # Statistical bounds
            median_rr = float(np.median(rr_ms))
            mad_rr = float(np.median(np.abs(rr_ms - median_rr)))
            stat_min = median_rr - stat_threshold * mad_rr
            stat_max = median_rr + stat_threshold * mad_rr

            n_phys_valid = int(np.sum(phys_valid))
            n_stat_valid = int(np.sum(stat_valid))
            n_both_valid = int(np.sum(phys_valid & stat_valid))

            sig_label = (
                "Pulse" if signal.signal_type.value.lower() == "ppg" else "Heart"
            )
            title = (
                f"RR Interval Distribution  "
                f"({n_intervals} intervals, mean={np.mean(rr_ms):.0f} ms, "
                f"median={median_rr:.0f} ms)"
            )

            pw = pg.PlotWidget(title=title)
            pw.setBackground("#1e1e1e")
            pw.plotItem.showGrid(x=True, y=True, alpha=0.3)
            pw.plotItem.setLabel("bottom", "RR Interval (ms)")
            pw.plotItem.setLabel("left", "Count")
            pw.plotItem.enableAutoRange(enable=True)

            # Histogram bars
            bar = pg.BarGraphItem(
                x=bin_centers,
                height=counts,
                width=bin_width * 0.9,
                brush=pg.mkBrush(QColor(100, 160, 255, 180)),
                pen=pg.mkPen(QColor(60, 110, 200, 255), width=1),
            )
            pw.addItem(bar)

            # Helper to add a labelled InfiniteLine
            def _add_vline(pos, color, label_text, label_pos=0.88):
                line = pg.InfiniteLine(
                    pos=pos,
                    angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                    label=label_text,
                    labelOpts={
                        "color": color,
                        "position": label_pos,
                        "rotateAxis": (1, 0),
                        "anchors": [(0, 1), (0, 1)],
                    },
                )
                pw.addItem(line)

            # Physiological bounds (red)
            _add_vline(phys_min_ms, "#FF4444", f"min={phys_min_ms:.0f}ms", 0.92)
            _add_vline(phys_max_ms, "#FF4444", f"max={phys_max_ms:.0f}ms", 0.85)

            # Statistical bounds (orange), only if MAD > 0
            if mad_rr > 1e-6:
                _add_vline(stat_min, "#FF9500", f"stat-min={stat_min:.0f}ms", 0.76)
                _add_vline(stat_max, "#FF9500", f"stat-max={stat_max:.0f}ms", 0.69)

            # Median line (white)
            _add_vline(median_rr, "#CCCCCC", f"median={median_rr:.0f}ms", 0.60)

            layout.addWidget(pw)

            # Legend / summary label
            phys_pct = 100.0 * n_phys_valid / n_intervals
            stat_pct = 100.0 * n_stat_valid / n_intervals
            both_pct = 100.0 * n_both_valid / n_intervals
            summary = (
                f"Total: {n_intervals}  |  "
                f"Physiologically valid (300-2000 ms): {n_phys_valid} ({phys_pct:.1f}%)  |  "
                f"Statistically valid (MAD x{stat_threshold}): {n_stat_valid} ({stat_pct:.1f}%)  |  "
                f"Both valid: {n_both_valid} ({both_pct:.1f}%)"
            )
            info = QLabel(summary)
            info.setStyleSheet("color: #aaa; font-size: 11px;")
            info.setWordWrap(True)
            layout.addWidget(info)

            legend = QLabel(
                "  -- Red dashed: physiological bounds (300/2000 ms)  "
                "  -- Orange dashed: MAD-based statistical bounds  "
                "  -- Grey dashed: median"
            )
            legend.setStyleSheet("color: #777; font-size: 10px;")
            layout.addWidget(legend)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        logger.info(
            f"RRHistogramDialog: {n_intervals} intervals, "
            f"phys=[{phys_min_ms},{phys_max_ms}]ms, "
            f"stat={stat_threshold}xMAD"
        )
