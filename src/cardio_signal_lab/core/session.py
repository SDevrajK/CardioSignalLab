"""Session save/load for resuming work.

Saves and restores:
- Source file path
- Processing pipeline
- Peak corrections
- View state (optional)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from cardio_signal_lab.core.data_models import PeakData


def save_session(
    source_file: Path | str,
    pipeline_data: dict[str, Any],
    peaks: PeakData | None,
    output_path: Path | str,
    view_state: dict[str, Any] | None = None,
) -> Path:
    """Save session to JSON file for resuming work.

    Args:
        source_file: Path to original data file
        pipeline_data: Serialized ProcessingPipeline
        peaks: PeakData with corrections
        output_path: Output session file path (.csl.json)
        view_state: Optional view state (zoom range, selected signal)

    Returns:
        Path to created session file
    """
    output_path = Path(output_path)
    source_file = Path(source_file)

    # Build session data
    session_data = {
        "source_file": str(source_file.absolute()),
        "processing_pipeline": pipeline_data,
        "peaks": None,
        "view_state": view_state or {},
    }

    # Add peaks if present
    if peaks is not None and peaks.num_peaks > 0:
        session_data["peaks"] = {
            "indices": peaks.indices.tolist(),
            "classifications": peaks.classifications.tolist(),
        }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(session_data, f, indent=2)

    logger.info(f"Saved session to {output_path}")
    return output_path


def load_session(session_path: Path | str) -> dict[str, Any]:
    """Load session from JSON file.

    Returns:
        Dict with keys: source_file, processing_pipeline, peaks, view_state
    """
    session_path = Path(session_path)

    with open(session_path, "r") as f:
        session_data = json.load(f)

    logger.info(f"Loaded session from {session_path}")
    return session_data
