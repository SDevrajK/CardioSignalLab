"""Processing history panel for CardioSignalLab.

Read-only dock widget showing the ordered sequence of processing steps
applied to the current signal. Updates after each pipeline operation.
"""
from __future__ import annotations

from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from loguru import logger


# Human-readable labels for pipeline operation names
_OPERATION_LABELS: dict[str, str] = {
    "bandpass": "Bandpass Filter",
    "highpass": "Highpass Filter",
    "lowpass": "Lowpass Filter",
    "notch": "Notch Filter",
    "baseline_correction": "Detrend (Polynomial)",
    "zero_reference": "DC Offset Removal",
    "segment": "Segment",
    "eemd_artifact_removal": "EEMD Artifact Removal",
    "ecg_clean": "NeuroKit2: ECG Clean",
    "ppg_clean": "NeuroKit2: PPG Clean",
    "eda_clean": "NeuroKit2: EDA Clean",
    "eda_decompose": "NeuroKit2: EDA Decompose",
    "detect_ecg_peaks": "Detect R-Peaks",
    "detect_ppg_peaks": "Detect Pulse Peaks",
    "detect_eda_features": "Detect SCR Peaks",
    # Structural operations (change signal length — cannot be pipeline-replayed)
    "crop": "Crop",
    "resample": "Resample",
}


def _format_step(idx: int, step) -> str:
    """Format a single pipeline step as a human-readable string.

    Args:
        idx: 1-based step index
        step: ProcessingStep with .operation (str) and .parameters (dict) attributes

    Returns:
        Formatted string like "1. Bandpass Filter: 0.5-40.0 Hz, order 4"
    """
    op = step.operation
    params = step.parameters
    label = _OPERATION_LABELS.get(op, op)

    param_parts: list[str] = []

    if op == "crop":
        param_parts = [
            f"{params.get('start', '?'):.3f}s - {params.get('end', '?'):.3f}s",
            f"{params.get('n_samples', '?')} samples",
        ]
    elif op == "resample":
        param_parts = [
            f"{params.get('original_sr', '?'):.1f} -> {params.get('target_sr', '?'):.1f} Hz",
            f"{params.get('n_samples', '?')} samples",
        ]
    elif op == "bandpass":
        param_parts = [
            f"{params.get('lowcut', '?')}-{params.get('highcut', '?')} Hz",
            f"order {params.get('order', '?')}",
        ]
    elif op in ("highpass", "lowpass"):
        param_parts = [
            f"{params.get('cutoff', '?')} Hz",
            f"order {params.get('order', '?')}",
        ]
    elif op == "notch":
        param_parts = [
            f"{params.get('freq', '?')} Hz",
            f"Q={params.get('quality_factor', '?')}",
        ]
    elif op == "baseline_correction":
        param_parts = [f"poly order {params.get('poly_order', '?')}"]
    elif op == "zero_reference":
        method = params.get("method", "mean")
        param_parts = [method]
        if method == "first_n":
            param_parts.append(f"n={params.get('n_samples', '?')}")
    elif op == "eda_decompose":
        param_parts = [params.get("component", "?"), params.get("method", "?")]
    elif op == "eemd_artifact_removal":
        param_parts = [
            f"ensemble={params.get('ensemble_size', '?')}",
            f"noise={params.get('noise_width', '?')}",
        ]

    if param_parts:
        return f"{idx}. {label}: {', '.join(str(p) for p in param_parts)}"
    return f"{idx}. {label}"


class ProcessingPanel(QDockWidget):
    """Read-only dock panel showing the processing pipeline applied to the current signal.

    Shows an ordered list of steps like:
        1. Bandpass Filter: 0.5-40.0 Hz, order 4
        2. Notch Filter: 60.0 Hz, Q=30.0
        3. NeuroKit2: ECG Clean

    Call update_steps() after each pipeline operation to refresh the display.
    Call clear() when switching channels.
    """

    def __init__(self, parent=None):
        super().__init__("Processing Steps", parent)
        self.setObjectName("ProcessingPanel")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._empty_label = QLabel("No processing applied.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._empty_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self._list)
        self._list.hide()

        self.setWidget(container)
        logger.debug("ProcessingPanel initialized")

    def update_steps(self, steps: list[dict]):
        """Refresh the panel with the current pipeline steps.

        Args:
            steps: List of pipeline step dicts, each with "operation" and "params" keys
        """
        self._list.clear()

        if not steps:
            self._empty_label.show()
            self._list.hide()
            return

        self._empty_label.hide()
        self._list.show()

        for i, step in enumerate(steps, start=1):
            text = _format_step(i, step)
            self._list.addItem(QListWidgetItem(text))

        logger.debug(f"ProcessingPanel updated: {len(steps)} steps")

    def update_combined(self, structural_steps: list, pipeline_steps: list):
        """Show structural ops (crop/resample) then filter pipeline steps.

        Structural steps are shown in orange/italic to indicate they are
        permanent and cannot be undone by Reset Processing.  A divider
        separates structural from subsequent filter steps.

        Args:
            structural_steps: List of structural ProcessingStep objects
                              (crop, resample — applied directly to raw signal)
            pipeline_steps:   List of ProcessingStep objects from the pipeline
        """
        self._list.clear()

        has_structural = bool(structural_steps)
        has_pipeline   = bool(pipeline_steps)

        if not has_structural and not has_pipeline:
            self._empty_label.show()
            self._list.hide()
            return

        self._empty_label.hide()
        self._list.show()

        idx = 1

        for step in structural_steps:
            text = _format_step(idx, step) + "  [permanent]"
            item = QListWidgetItem(text)
            item.setForeground(QColor("#b05a00"))   # dark orange
            item.setToolTip(
                "Structural operation: changes signal length/timestamps.\n"
                "Cannot be reverted by Reset Processing."
            )
            self._list.addItem(item)
            idx += 1

        if has_structural and has_pipeline:
            sep = QListWidgetItem("--- filters applied after structural ops ---")
            sep.setForeground(QColor("#999999"))
            sep.setFlags(sep.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._list.addItem(sep)

        for step in pipeline_steps:
            text = _format_step(idx, step)
            self._list.addItem(QListWidgetItem(text))
            idx += 1

        logger.debug(
            f"ProcessingPanel updated: {len(structural_steps)} structural, "
            f"{len(pipeline_steps)} pipeline steps"
        )

    def clear(self):
        """Clear all steps (e.g. when switching to a new channel)."""
        self._list.clear()
        self._list.hide()
        self._empty_label.show()
