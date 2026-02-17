"""IMF selection dialog for interactive EEMD artifact removal.

Displays each IMF as a small waveform plot with a checkbox.  Auto-excluded
components are pre-unchecked.  The user can review, adjust, and confirm
which components to keep before signal reconstruction.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QScrollArea,
    QWidget,
    QDialogButtonBox,
)


class ImfSelectionDialog(QDialog):
    """Interactive IMF selection dialog for EEMD artifact removal.

    Shows one row per IMF:  [checkbox | waveform plot | frequency / energy stats]

    Unchecked IMFs are excluded from reconstruction.  Auto-classification
    from `auto_select_artifact_imfs` pre-populates the checkboxes.
    The residue (baseline trend) is always included in reconstruction.
    """

    _PLOT_HEIGHT = 110   # px per IMF row plot
    _STATS_WIDTH = 175   # px for the stats label column
    _CB_WIDTH    = 70    # px for checkbox column

    _COLOR_KEEP    = "#1f77b4"   # blue
    _COLOR_EXCLUDE = "#cc3333"   # red
    _BG_KEEP       = "white"
    _BG_EXCLUDE    = "#fff4f4"

    def __init__(
        self,
        imfs: np.ndarray,
        residue: np.ndarray,
        characteristics: list[dict],
        auto_excluded: list[int],
        sampling_rate: float,
        parent=None,
    ):
        """
        Args:
            imfs: Array [n_imfs x n_samples]
            residue: Residue signal [n_samples]
            characteristics: Output of analyze_imf_characteristics()
            auto_excluded: IMF indices auto-classified for exclusion
            sampling_rate: Signal sampling rate (Hz), used for time axis
            parent: Parent widget
        """
        super().__init__(parent)
        self.imfs = imfs
        self.residue = residue
        self.characteristics = characteristics
        self._auto_excluded = set(auto_excluded)
        self.sampling_rate = sampling_rate

        self._checkboxes: list[QCheckBox] = []
        self._plot_widgets: list[pg.PlotWidget] = []
        self._stats_labels: list[QLabel] = []

        self.setWindowTitle("EEMD Component Selection")
        self.setMinimumSize(860, 560)
        self.resize(1050, 720)

        self._build_ui()
        logger.debug(
            f"ImfSelectionDialog opened: {imfs.shape[0]} IMFs, "
            f"{len(auto_excluded)} auto-excluded"
        )

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # -- Instructions -------------------------------------------------
        n_imfs = self.imfs.shape[0]
        n_auto = len(self._auto_excluded)
        info = QLabel(
            f"{n_imfs} IMFs decomposed.  "
            f"{n_auto} auto-classified for exclusion (shown in red, unchecked).\n"
            "Check = KEEP in reconstruction.  The residue (baseline trend) is always included."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        # -- Toolbar buttons -----------------------------------------------
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Select All",    self._select_all),
            ("Deselect All",  self._deselect_all),
            ("Reset to Auto", self._reset_to_auto),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # -- Scrollable IMF rows -------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(3)
        vbox.setContentsMargins(4, 4, 4, 4)

        t_axis = np.arange(self.imfs.shape[1]) / self.sampling_rate

        for i, char in enumerate(self.characteristics):
            is_excluded = i in self._auto_excluded
            row = self._build_imf_row(i, char, t_axis, is_excluded)
            vbox.addWidget(row)

        vbox.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll, stretch=1)

        # -- OK / Cancel ---------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _build_imf_row(
        self,
        idx: int,
        char: dict,
        t_axis: np.ndarray,
        is_excluded: bool,
    ) -> QWidget:
        """Build one IMF row widget."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 1, 2, 1)

        # Checkbox
        cb = QCheckBox(f"IMF {idx}")
        cb.setChecked(not is_excluded)
        cb.setFixedWidth(self._CB_WIDTH)
        self._checkboxes.append(cb)
        layout.addWidget(cb)

        # Waveform plot
        plot = pg.PlotWidget()
        plot.setFixedHeight(self._PLOT_HEIGHT)
        plot.setSizePolicy(
            plot.sizePolicy().horizontalPolicy(),
            plot.sizePolicy().verticalPolicy(),
        )
        plot.plotItem.hideAxis("bottom")
        plot.plotItem.hideAxis("left")
        plot.plotItem.setMouseEnabled(x=False, y=False)
        plot.plotItem.hideButtons()
        plot.setBackground(self._BG_EXCLUDE if is_excluded else self._BG_KEEP)

        color = self._COLOR_EXCLUDE if is_excluded else self._COLOR_KEEP
        plot.plot(t_axis, self.imfs[idx], pen=pg.mkPen(color, width=1))
        self._plot_widgets.append(plot)
        layout.addWidget(plot, stretch=1)

        # Stats label
        freq       = char["peak_freq"]
        energy_pct = char["energy_pct"]
        status     = "AUTO-EXCLUDED" if is_excluded else "KEEP"
        stats = QLabel(
            f"Peak: {freq:.2f} Hz\n"
            f"Energy: {energy_pct:.1f}%\n"
            f"[{status}]"
        )
        stats.setFixedWidth(self._STATS_WIDTH)
        if is_excluded:
            stats.setStyleSheet("color: #cc3333;")
        self._stats_labels.append(stats)
        layout.addWidget(stats)

        # Update plot background when checkbox is toggled
        def _on_toggle(checked, pw=plot):
            pw.setBackground(self._BG_KEEP if checked else self._BG_EXCLUDE)

        cb.toggled.connect(_on_toggle)

        return row

    # ------------------------------------------------------------------ #
    # Toolbar actions                                                       #
    # ------------------------------------------------------------------ #

    def _select_all(self):
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self._checkboxes:
            cb.setChecked(False)

    def _reset_to_auto(self):
        for i, cb in enumerate(self._checkboxes):
            cb.setChecked(i not in self._auto_excluded)

    # ------------------------------------------------------------------ #
    # Result                                                               #
    # ------------------------------------------------------------------ #

    def get_excluded_imfs(self) -> list[int]:
        """Return indices of IMFs to exclude (unchecked boxes)."""
        return [i for i, cb in enumerate(self._checkboxes) if not cb.isChecked()]
