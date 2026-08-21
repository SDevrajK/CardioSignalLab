"""CardioSignalLab - Main Entry Point

Desktop application for viewing, processing, and correcting physiological signals.
"""
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from loguru import logger

from cardio_signal_lab import __version__


def _user_log_dir() -> Path:
    """Per-user writable directory for log files (install dir may be read-only)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "CardioSignalLab" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "CardioSignalLab"
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "CardioSignalLab" / "logs"


_log_path = _user_log_dir() / "cardio_signal_lab.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.remove()  # Remove default handler
# In PyInstaller windowed builds (console=False), sys.stderr is None.
if sys.stderr is not None:
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
logger.add(
    _log_path,
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


def main():
    """Launch the CardioSignalLab application."""
    logger.info("Starting CardioSignalLab")

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("CardioSignalLab")
    app.setOrganizationName("HebertLab")
    app.setApplicationVersion(__version__)

    # Import main window here to ensure QApplication exists first
    from cardio_signal_lab.gui.main_window import MainWindow

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("CardioSignalLab window displayed")

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
