"""CardioSignalLab application entry point.

Registered as the console script entry point in pyproject.toml.
"""
import sys

from loguru import logger
from PySide6.QtWidgets import QApplication


def main() -> None:
    """Launch the CardioSignalLab application."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
    )
    logger.add(
        "cardio_signal_lab.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )

    logger.info("Starting CardioSignalLab")

    app = QApplication(sys.argv)
    app.setApplicationName("CardioSignalLab")
    app.setOrganizationName("HebertLab")
    app.setApplicationVersion("0.1.0")

    from cardio_signal_lab.gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("CardioSignalLab window displayed")
    sys.exit(app.exec())
