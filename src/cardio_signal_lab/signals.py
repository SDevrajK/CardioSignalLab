"""Application-wide signal/slot event bus for decoupling GUI components.

This module defines custom Qt signals that allow components to communicate
without tight coupling. All components can connect to these signals and emit
them as needed.
"""
from PySide6.QtCore import QObject, Signal

# Type imports for signal type hints (imported at runtime for slots)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cardio_signal_lab.core.data_models import PeakData, RecordingSession, SignalData


class AppSignals(QObject):
    """Central event bus with custom Qt signals for application-wide communication.

    This singleton class provides signals that decouple GUI components from
    processing logic and from each other. Components emit signals when state
    changes occur, and other components connect to these signals to react.

    Example:
        >>> signals = get_app_signals()
        >>> signals.file_loaded.connect(on_file_loaded)
        >>> signals.file_loaded.emit(session)
    """

    # File operations
    file_loaded = Signal(object)  # Emits: RecordingSession
    file_save_requested = Signal()
    file_export_requested = Signal()

    # Signal selection and mode changes
    signal_selected = Signal(object)  # Emits: SignalData
    signal_type_selected = Signal(object)  # Emits: SignalType
    mode_changed = Signal(str)  # Emits: "multi", "type", or "channel"

    # Peak editing operations
    peaks_updated = Signal(object)  # Emits: PeakData
    peak_selected = Signal(int)  # Emits: peak index
    peak_deselected = Signal()

    # Processing operations
    processing_started = Signal(str)  # Emits: operation name
    processing_progress = Signal(int)  # Emits: progress percentage (0-100)
    processing_finished = Signal()
    processing_cancelled = Signal()
    processing_error = Signal(str)  # Emits: error message

    # View operations
    view_zoom_changed = Signal(float, float)  # Emits: (x_min, x_max)
    view_reset_requested = Signal()

    def __init__(self):
        """Initialize the signal bus."""
        super().__init__()


# Global singleton instance
_app_signals: AppSignals | None = None


def get_app_signals() -> AppSignals:
    """Get the global AppSignals singleton.

    Returns:
        AppSignals instance for connecting/emitting signals

    Example:
        >>> from cardio_signal_lab.signals import get_app_signals
        >>> signals = get_app_signals()
        >>> signals.file_loaded.connect(my_handler)
    """
    global _app_signals
    if _app_signals is None:
        _app_signals = AppSignals()
    return _app_signals
