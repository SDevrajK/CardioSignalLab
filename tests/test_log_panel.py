"""Tests for log panel widget."""
import pytest
from loguru import logger
from PySide6.QtCore import Qt

from cardio_signal_lab.gui.log_panel import LogPanel


@pytest.fixture
def log_panel(qtbot):
    """Create log panel fixture."""
    panel = LogPanel()
    qtbot.addWidget(panel)
    return panel


def test_log_panel_creation(log_panel):
    """Test log panel can be created."""
    assert log_panel is not None
    assert log_panel.text_edit is not None
    assert log_panel.text_edit.isReadOnly()


def test_log_panel_initial_state(log_panel):
    """Test log panel starts empty."""
    assert log_panel.text_edit.toPlainText() == ""


def test_log_panel_append_message(log_panel, qtbot):
    """Test appending log messages."""
    # Emit a log message
    log_panel.log_handler.log_message.emit("INFO", "Test message")

    # Wait for signal to be processed
    qtbot.wait(100)

    # Check text was added
    text = log_panel.text_edit.toPlainText()
    assert "INFO" in text
    assert "Test message" in text


def test_log_panel_multiple_levels(log_panel, qtbot):
    """Test different log levels appear correctly."""
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in levels:
        log_panel.log_handler.log_message.emit(level, f"Test {level}")
        qtbot.wait(50)

    text = log_panel.text_edit.toPlainText()
    for level in levels:
        assert level in text
        assert f"Test {level}" in text


def test_log_panel_clear(log_panel, qtbot):
    """Test clearing log panel."""
    log_panel.log_handler.log_message.emit("INFO", "Test message")
    qtbot.wait(100)

    assert log_panel.text_edit.toPlainText() != ""

    log_panel.clear()
    assert log_panel.text_edit.toPlainText() == ""


def test_log_panel_max_lines(log_panel, qtbot):
    """Test log panel respects max line count."""
    # The panel is set to max 1000 blocks
    assert log_panel.text_edit.document().maximumBlockCount() == 1000


def test_log_panel_loguru_sink(log_panel, qtbot):
    """Test loguru integration."""
    # Add log panel as a sink with custom format
    handler_id = logger.add(
        log_panel.get_loguru_sink(),
        format="<lvl>{level}</lvl>|{message}",
        level="INFO",
        colorize=False
    )

    # Log a message
    logger.info("Test loguru message")
    qtbot.wait(100)

    # Check it appears in the panel
    text = log_panel.text_edit.toPlainText()
    assert "Test loguru message" in text

    # Clean up
    logger.remove(handler_id)


def test_log_panel_dock_features(log_panel):
    """Test dock widget configuration."""
    # Check allowed areas
    allowed = log_panel.allowedAreas()
    assert allowed & Qt.DockWidgetArea.BottomDockWidgetArea
    assert allowed & Qt.DockWidgetArea.RightDockWidgetArea

    # Check features
    features = log_panel.features()
    assert features & log_panel.DockWidgetFeature.DockWidgetClosable
    assert features & log_panel.DockWidgetFeature.DockWidgetMovable
