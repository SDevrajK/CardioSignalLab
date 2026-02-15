"""Background processing worker using QThread.

Runs long-running operations (EEMD, peak detection) in a background thread
to keep the GUI responsive. Emits progress and completion signals.
"""
from __future__ import annotations

from typing import Any, Callable

from loguru import logger
from PySide6.QtCore import QThread, Signal


class ProcessingWorker(QThread):
    """Background worker for signal processing operations.

    Runs a callable in a separate thread and emits signals for progress,
    completion, and errors.

    Signals:
        progress: Emitted with progress percentage (0-100)
        finished: Emitted with the result when processing completes
        error: Emitted with error message on failure
    """

    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        parent=None,
    ):
        """Initialize worker with a callable to run.

        Args:
            func: Function to execute in background
            args: Positional arguments for func
            kwargs: Keyword arguments for func
            parent: Parent QObject
        """
        super().__init__(parent=parent)
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self._cancelled = False

    def run(self):
        """Execute the processing function in background thread."""
        try:
            self.progress.emit(0)
            logger.info(f"Processing worker started: {self.func.__name__}")

            result = self.func(*self.args, **self.kwargs)

            if self._cancelled:
                logger.info("Processing was cancelled")
                return

            self.progress.emit(100)
            self.finished.emit(result)
            logger.info("Processing worker completed successfully")

        except Exception as e:
            logger.error(f"Processing worker failed: {e}")
            self.error.emit(str(e))

    def cancel(self):
        """Request cancellation of the processing."""
        self._cancelled = True
        logger.info("Processing cancellation requested")

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled
