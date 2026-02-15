"""Log panel widget with real-time loguru sink."""
from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QDockWidget
from PySide6.QtGui import QTextCursor, QColor, QTextCharFormat
from PySide6.QtCore import Qt, Signal, QObject


class LogSinkHandler:
    """Loguru sink handler that captures formatted messages and emits Qt signals."""

    def __init__(self, signal_emitter):
        self.signal_emitter = signal_emitter

    def write(self, message):
        """Called by loguru with the formatted message string.

        Expected format: "<lvl>LEVEL</lvl>|message\\n"
        """
        # Strip tags and split on |
        message = message.rstrip('\n')

        # Remove <lvl> and </lvl> tags
        message = message.replace('<lvl>', '').replace('</lvl>', '')

        # Split on first |
        parts = message.split('|', 1)
        if len(parts) == 2:
            level = parts[0].strip()
            text = parts[1].strip()
        else:
            level = "INFO"
            text = message.strip()

        self.signal_emitter.log_message.emit(level, text)


class LogHandler(QObject):
    """Thread-safe signal emitter for log messages."""

    log_message = Signal(str, str)  # (level, message)


class LogPanel(QDockWidget):
    """Dockable log panel with color-coded log messages.

    Displays real-time log messages from loguru with color coding:
    - DEBUG: gray
    - INFO: white
    - WARNING: yellow
    - ERROR: red
    - CRITICAL: bold red
    """

    def __init__(self, parent=None):
        super().__init__("Log", parent)

        # Create text edit (read-only)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.document().setMaximumBlockCount(1000)  # Limit to 1000 lines

        # Set monospace font
        font = self.text_edit.font()
        font.setFamily("Courier New")
        font.setPointSize(9)
        self.text_edit.setFont(font)

        # Dark background
        self.text_edit.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )

        self.setWidget(self.text_edit)

        # Configure dock widget
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        # Thread-safe log handler
        self.log_handler = LogHandler()
        self.log_handler.log_message.connect(self._append_log_message)

        # Level colors
        self.level_colors = {
            "TRACE": QColor("#808080"),
            "DEBUG": QColor("#808080"),
            "INFO": QColor("#d4d4d4"),
            "SUCCESS": QColor("#00ff00"),
            "WARNING": QColor("#ffff00"),
            "ERROR": QColor("#ff6666"),
            "CRITICAL": QColor("#ff0000"),
        }

    def _append_log_message(self, level: str, message: str):
        """Append log message to text edit (called from main thread via signal)."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Format level with color
        fmt = QTextCharFormat()
        color = self.level_colors.get(level, QColor("#d4d4d4"))
        fmt.setForeground(color)

        if level == "CRITICAL":
            fmt.setFontWeight(700)  # Bold

        # Insert timestamp prefix
        cursor.insertText(f"[{level:<8}] ", fmt)

        # Insert message
        fmt.setForeground(QColor("#d4d4d4"))
        fmt.setFontWeight(400)
        cursor.insertText(message + "\n", fmt)

        # Auto-scroll to bottom
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def clear(self):
        """Clear all log messages."""
        self.text_edit.clear()

    def get_loguru_sink(self):
        """Return a handler callable for loguru.add().

        Usage:
            logger.add(
                log_panel.get_loguru_sink(),
                format="<lvl>{level}</lvl>|{message}",
                level="INFO",
                colorize=False
            )
        """
        return LogSinkHandler(self.log_handler)
