"""CardioSignalLab - Main Entry Point

Desktop application for viewing, processing, and correcting physiological signals.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "cardio_signal_lab.log",
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
    app.setApplicationVersion("0.1.0")

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
