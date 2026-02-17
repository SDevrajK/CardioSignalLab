"""Spreadsheet-style event editor dialog.

Lets the user paste rows directly from Excel/LibreOffice into a three-column
table (time_s_raw | time_s_adjusted | label).  The adjusted column is
read-only and shows  raw - t0,  where t0 is the zero-reference offset set at
the top of the dialog.

Usage
-----
    dialog = EventEditorDialog(current_events, parent=self)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        new_events = dialog.get_events()   # uses adjusted timestamps
"""
from __future__ import annotations

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from cardio_signal_lab.core.data_models import EventData

# Lower-cased substrings that suggest a column is a timestamp
_TIME_HINTS = {"time", "timestamp", "onset", "t_s", "t(s)", "sec"}
# Lower-cased substrings that suggest a column is a label
_LABEL_HINTS = {"label", "event", "name", "description", "marker", "code", "condition"}


def _detect_columns(headers: list[str]) -> tuple[int, int]:
    """Return (time_col_index, label_col_index) from a list of header strings."""
    time_idx, label_idx = 0, 1

    for i, h in enumerate(headers):
        hl = h.lower().strip()
        if any(hint in hl for hint in _TIME_HINTS):
            time_idx = i
            break

    for i, h in enumerate(headers):
        hl = h.lower().strip()
        if any(hint in hl for hint in _LABEL_HINTS):
            label_idx = i
            break

    if time_idx == label_idx and len(headers) > 1:
        label_idx = (time_idx + 1) % len(headers)

    return time_idx, label_idx


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


class EventEditorDialog(QDialog):
    """Spreadsheet-style dialog for viewing and editing session events.

    Three columns:
    - time_s (raw)      Editable.  The timestamp as typed or pasted.
    - time_s (adjusted) Read-only. raw - t0 offset.  This is what is saved.
    - label             Editable.

    Setting t0 to the recording start time (in the original timestamp domain)
    zero-references pasted timestamps to align them with the signal.
    """

    COL_TIME_RAW = 0
    COL_TIME_ADJ = 1   # read-only
    COL_LABEL = 2

    def __init__(self, events: list[EventData], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Event Editor")
        self.resize(680, 500)

        layout = QVBoxLayout(self)

        # Instructions
        help_text = (
            "Edit events, or paste rows from Excel / LibreOffice (Ctrl+V).  "
            "Multi-column pastes auto-detect time and label columns from headers.  "
            "Use 'Swap Columns' when time and label are reversed.  "
            "Set t0 to subtract a zero-reference offset from all times."
        )
        lbl = QLabel(help_text)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # t0 offset row
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Subtract t0:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(4)
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setSuffix(" s")
        self.offset_spin.setToolTip(
            "Subtract this value from every raw time entry.\n"
            "Set to the recording start time to zero-reference\n"
            "timestamps pasted from the original recording."
        )
        offset_row.addWidget(self.offset_spin)
        note = QLabel("  adjusted = raw \u2212 t0  (saved column)")
        note.setStyleSheet("color: gray; font-style: italic;")
        offset_row.addWidget(note)
        offset_row.addStretch()
        layout.addLayout(offset_row)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["time_s (raw)", "time_s (adjusted)", "label"])
        self.table.horizontalHeader().setSectionResizeMode(
            self.COL_TIME_RAW, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            self.COL_TIME_ADJ, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            self.COL_LABEL, QHeaderView.ResizeMode.Stretch
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # Row management buttons
        btn_row = QHBoxLayout()

        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self._add_empty_row)
        btn_row.addWidget(add_btn)

        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self._delete_selected_rows)
        btn_row.addWidget(del_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(clear_btn)

        swap_btn = QPushButton("Swap Columns")
        swap_btn.setToolTip(
            "Swap the raw-time and label columns.\n"
            "Use this when pasted data has label in the first column."
        )
        swap_btn.clicked.connect(self._swap_columns)
        btn_row.addWidget(swap_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wire signals
        self.offset_spin.valueChanged.connect(self._refresh_adjusted_column)
        self.table.cellChanged.connect(self._on_cell_changed)

        # Populate from existing events
        self._populate(events)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_events(self) -> list[EventData]:
        """Parse the adjusted-time column into a sorted list of EventData.

        Rows with non-numeric or empty adjusted times are skipped with a warning.
        """
        events: list[EventData] = []
        skipped = 0

        for row in range(self.table.rowCount()):
            adj_item = self.table.item(row, self.COL_TIME_ADJ)
            label_item = self.table.item(row, self.COL_LABEL)

            adj_str = adj_item.text().strip() if adj_item else ""
            label_str = label_item.text().strip() if label_item else ""

            if not adj_str:
                skipped += 1
                continue

            try:
                t = float(adj_str)
            except ValueError:
                logger.warning(
                    f"Event row {row}: cannot parse adjusted time '{adj_str}' — skipped"
                )
                skipped += 1
                continue

            if not label_str:
                label_str = "event"

            events.append(EventData(timestamp=t, label=label_str))

        if skipped:
            logger.warning(f"Skipped {skipped} event rows with invalid timestamps")

        events.sort(key=lambda e: e.timestamp)
        return events

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if (
            event.key() == Qt.Key.Key_V
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self._paste_from_clipboard()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selected_rows()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def _populate(self, events: list[EventData]):
        self.table.setRowCount(0)
        for ev in sorted(events, key=lambda e: e.timestamp):
            self._append_row(str(ev.timestamp), ev.label)

    def _add_empty_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.blockSignals(True)
        self.table.setItem(row, self.COL_TIME_RAW, QTableWidgetItem(""))
        self._place_adjusted_item(row, "")
        self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(""))
        self.table.blockSignals(False)
        self.table.setCurrentCell(row, self.COL_TIME_RAW)

    def _delete_selected_rows(self):
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.table.removeRow(row)

    def _clear_all(self):
        reply = QMessageBox.question(
            self, "Clear All", "Remove all events from the table?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)

    def _swap_columns(self):
        """Swap the raw-time and label columns for all rows."""
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            raw_item = self.table.item(row, self.COL_TIME_RAW)
            label_item = self.table.item(row, self.COL_LABEL)
            raw_text = raw_item.text() if raw_item else ""
            label_text = label_item.text() if label_item else ""
            self.table.setItem(row, self.COL_TIME_RAW, QTableWidgetItem(label_text))
            self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(raw_text))
        self.table.blockSignals(False)
        self._refresh_adjusted_column()
        logger.info("Swapped time_s and label columns")

    def _append_row(self, raw_time_str: str, label_str: str):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.blockSignals(True)
        self.table.setItem(row, self.COL_TIME_RAW, QTableWidgetItem(raw_time_str.strip()))
        self._place_adjusted_item(row, raw_time_str.strip())
        self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(label_str.strip()))
        self.table.blockSignals(False)

    def _place_adjusted_item(self, row: int, raw_str: str):
        """Compute and place a read-only adjusted-time item."""
        try:
            adj_text = f"{float(raw_str) - self.offset_spin.value():.6g}"
        except ValueError:
            adj_text = ""
        adj_item = QTableWidgetItem(adj_text)
        adj_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(row, self.COL_TIME_ADJ, adj_item)

    # ------------------------------------------------------------------
    # Live update slots
    # ------------------------------------------------------------------

    def _on_cell_changed(self, row: int, col: int):
        """Recompute the adjusted cell whenever the raw-time cell is edited."""
        if col != self.COL_TIME_RAW:
            return
        raw_item = self.table.item(row, self.COL_TIME_RAW)
        raw_str = raw_item.text().strip() if raw_item else ""
        self.table.blockSignals(True)
        self._place_adjusted_item(row, raw_str)
        self.table.blockSignals(False)

    def _refresh_adjusted_column(self):
        """Recompute every adjusted cell (called when t0 offset changes)."""
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            raw_item = self.table.item(row, self.COL_TIME_RAW)
            raw_str = raw_item.text().strip() if raw_item else ""
            self._place_adjusted_item(row, raw_str)
        self.table.blockSignals(False)

    # ------------------------------------------------------------------
    # Clipboard paste
    # ------------------------------------------------------------------

    def _paste_from_clipboard(self):
        """Paste TSV/CSV data from the clipboard.

        Auto-detects delimiter and header row.  Appends below existing rows.
        """
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text.strip():
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return

        # Prefer tab (Excel default), fall back to comma
        delimiter = "\t" if "\t" in lines[0] else ","
        rows = [line.split(delimiter) for line in lines]

        # Header detection: first row is a header if no cell is numeric
        first_row = rows[0]
        has_header = not any(_is_numeric(cell.strip()) for cell in first_row)

        if has_header:
            headers = [c.strip() for c in first_row]
            time_col, label_col = _detect_columns(headers)
            data_rows = rows[1:]
            logger.debug(
                f"Paste: headers={headers}; "
                f"time_col={headers[time_col]}, label_col={headers[label_col]}"
            )
        else:
            time_col, label_col = 0, 1
            data_rows = rows
            logger.debug("Paste: no header detected; col 0=time, col 1=label")

        n_added = 0
        for cells in data_rows:
            time_str = cells[time_col].strip() if time_col < len(cells) else ""
            label_str = cells[label_col].strip() if label_col < len(cells) else ""

            if not time_str and not label_str:
                continue

            self._append_row(time_str, label_str)
            n_added += 1

        logger.info(f"Pasted {n_added} event rows from clipboard")
