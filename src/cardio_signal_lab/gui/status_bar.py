"""Status bar showing current mode and signal context."""
from PySide6.QtWidgets import QStatusBar
from loguru import logger

from cardio_signal_lab.signals import get_app_signals


class AppStatusBar(QStatusBar):
    """Status bar that displays current mode and signal context.

    Updates automatically in response to AppSignals:
    - mode_changed: Updates to show "Multi-Signal Mode" or "Single-Signal Mode"
    - file_loaded: Shows number of signals loaded
    - signal_selected: Shows selected signal type and channel

    Example displays:
    - "Multi-Signal Mode (3 signals loaded)"
    - "Single-Signal Mode: ECG (Channel 1)"
    """

    def __init__(self, parent=None):
        """Initialize the status bar.

        Args:
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)

        self.signals = get_app_signals()
        self.current_mode = "multi"
        self.num_signals = 0
        self.selected_signal_type = None
        self.selected_signal_name = None

        # Connect to AppSignals
        self.signals.mode_changed.connect(self._on_mode_changed)
        self.signals.file_loaded.connect(self._on_file_loaded)
        self.signals.signal_selected.connect(self._on_signal_selected)

        # Set initial message
        self._update_message()

        logger.debug("AppStatusBar initialized")

    def _update_message(self):
        """Update status bar message based on current state."""
        if self.current_mode == "multi":
            if self.num_signals > 0:
                message = f"Multi-Signal Mode ({self.num_signals} signals loaded)"
            else:
                message = "Multi-Signal Mode (no file loaded)"
        else:  # single mode
            if self.selected_signal_type and self.selected_signal_name:
                message = f"Single-Signal Mode: {self.selected_signal_type} ({self.selected_signal_name})"
            else:
                message = "Single-Signal Mode"

        self.showMessage(message)
        logger.debug(f"Status bar updated: {message}")

    def _on_mode_changed(self, mode: str):
        """Handle mode change signal.

        Args:
            mode: "multi" or "single"
        """
        self.current_mode = mode
        self._update_message()

    def _on_file_loaded(self, session):
        """Handle file loaded signal.

        Args:
            session: RecordingSession object
        """
        # Count signals in session
        self.num_signals = len(session.signals) if hasattr(session, "signals") else 0
        self._update_message()

    def _on_signal_selected(self, signal_data):
        """Handle signal selected signal.

        Args:
            signal_data: SignalData object
        """
        # Extract signal type and channel name
        if hasattr(signal_data, "signal_type"):
            self.selected_signal_type = signal_data.signal_type.name if hasattr(signal_data.signal_type, "name") else str(signal_data.signal_type)
        else:
            self.selected_signal_type = "Unknown"

        if hasattr(signal_data, "channel_name"):
            self.selected_signal_name = signal_data.channel_name
        else:
            self.selected_signal_name = "Unknown Channel"

        self._update_message()
