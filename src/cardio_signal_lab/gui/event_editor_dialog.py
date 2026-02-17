"""Spreadsheet-style event editor dialog.

Lets the user paste rows directly from Excel/LibreOffice into a two-column
table (time_s | label).  When the clipboard contains a multi-column block
(e.g. participant_id | time_s | label), the dialog auto-detects which
column is the timestamp and which is the label.

Usage
-----
    dialog = EventEditorDialog(current_events, parent=self)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        new_events = dialog.get_events()
"""
from __future__ import annotations

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
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
    """Return (time_col_index, label_col_index) from a list of header strings.

    Matches against known substrings; falls back to column 0 for time and
    column 1 for label when no match is found.
    """
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

    # Avoid both pointing at the same column
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

    Features
    --------
    - Pre-populated with current session events
    - Ctrl+V pastes TSV from clipboard (Excel, LibreOffice, Numbers)
    - Multi-column paste: auto-detects time and label columns from headers
    - Add Row, Delete Row, Clear All buttons
    - On Accept: parses table into a list of EventData sorted by time
    """

    COL_TIME = 0
    COL_LABEL = 1

    def __init__(self, events: list[EventData], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Event Editor")
        self.resize(520, 480)

        layout = QVBoxLayout(self)

        # Instructions
        help_text = (
            "Edit events below, or paste rows directly from Excel / LibreOffice "
            "(Ctrl+V).  Multi-column pastes are supported — the time and label "
            "columns are detected automatically from header names."
        )
        lbl = QLabel(help_text)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # Table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["time_s", "label"])
        self.table.horizontalHeader().setSectionResizeMode(
            self.COL_TIME, QHeaderView.ResizeMode.ResizeToContents
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

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Populate from existing events
        self._populate(events)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_events(self) -> list[EventData]:
        """Parse the table into a sorted list of EventData.

        Rows with non-numeric or empty time fields are skipped with a warning.
        """
        events: list[EventData] = []
        skipped = 0

        for row in range(self.table.rowCount()):
            time_item = self.table.item(row, self.COL_TIME)
            label_item = self.table.item(row, self.COL_LABEL)

            time_str = time_item.text().strip() if time_item else ""
            label_str = label_item.text().strip() if label_item else ""

            if not time_str:
                skipped += 1
                continue

            try:
                t = float(time_str)
            except ValueError:
                logger.warning(f"Event row {row}: cannot parse time '{time_str}' — skipped")
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
        self.table.setItem(row, self.COL_TIME, QTableWidgetItem(""))
        self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(""))
        self.table.setCurrentCell(row, self.COL_TIME)

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

    def _append_row(self, time_str: str, label_str: str):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, self.COL_TIME, QTableWidgetItem(time_str.strip()))
        self.table.setItem(row, self.COL_LABEL, QTableWidgetItem(label_str.strip()))

    # ------------------------------------------------------------------
    # Clipboard paste
    # ------------------------------------------------------------------

    def _paste_from_clipboard(self):
        """Paste TSV data from the clipboard into the table.

        Handles both comma-separated and tab-separated clipboard content.
        Detects column headers automatically; extra columns are discarded.
        Appends below any existing rows.
        """
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text.strip():
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return

        # Detect delimiter: prefer tab (Excel default), fall back to comma
        delimiter = "\t" if "\t" in lines[0] else ","

        rows = [line.split(delimiter) for line in lines]

        # Check whether the first row is a header (all cells non-numeric)
        first_row = rows[0]
        has_header = not any(_is_numeric(cell.strip()) for cell in first_row)

        if has_header:
            headers = [c.strip() for c in first_row]
            time_col, label_col = _detect_columns(headers)
            data_rows = rows[1:]
            logger.debug(
                f"Paste: detected headers {headers}; "
                f"time_col={headers[time_col]}, label_col={headers[label_col]}"
            )
        else:
            # No header row — assume col 0 = time, col 1 = label
            time_col, label_col = 0, 1
            data_rows = rows
            logger.debug("Paste: no header row detected; using col 0=time, col 1=label")

        n_added = 0
        for cells in data_rows:
            # Guard against short rows
            time_str = cells[time_col].strip() if time_col < len(cells) else ""
            label_str = cells[label_col].strip() if label_col < len(cells) else ""

            if not time_str and not label_str:
                continue  # skip blank rows

            self._append_row(time_str, label_str)
            n_added += 1

        logger.info(f"Pasted {n_added} event rows from clipboard")
