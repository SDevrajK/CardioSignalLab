"""Export processed signals and annotations to various formats.

Supports CSV, NPY, and annotation exports with processing parameter sidecars
for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cardio_signal_lab.core.data_models import PeakData, ProcessingStep, SignalData


def export_csv(
    signal: SignalData,
    peaks: PeakData | None,
    output_path: Path | str,
    include_peaks: bool = True,
) -> Path:
    """Export signal and peaks to CSV.

    CSV format:
    - time_s: Timestamp in seconds
    - signal: Signal amplitude
    - peak: 1 if peak at this sample, 0 otherwise (if include_peaks=True)
    - peak_classification: Classification value (if include_peaks=True)

    Args:
        signal: SignalData to export
        peaks: PeakData with peak annotations (optional)
        output_path: Output CSV file path
        include_peaks: Whether to include peak columns

    Returns:
        Path to created CSV file
    """
    output_path = Path(output_path)

    # Build DataFrame
    df = pd.DataFrame({
        "time_s": signal.timestamps,
        "signal": signal.samples,
    })

    if include_peaks and peaks is not None and peaks.num_peaks > 0:
        # Create peak marker column
        peak_marker = np.zeros(len(signal.samples), dtype=int)
        peak_marker[peaks.indices] = 1

        # Create classification column
        peak_classification = np.full(len(signal.samples), -1, dtype=int)
        peak_classification[peaks.indices] = peaks.classifications

        df["peak"] = peak_marker
        df["peak_classification"] = peak_classification

    # Save
    df.to_csv(output_path, index=False)

    logger.info(f"Exported signal to CSV: {output_path} ({len(df)} rows)")
    return output_path


def export_npy(
    signal: SignalData,
    peaks: PeakData | None,
    output_path: Path | str,
) -> Path:
    """Export signal and peaks as NumPy arrays.

    Creates:
    - {output_path}_signal.npy: Shape (N, 2) with [timestamps, samples]
    - {output_path}_peaks.npy: Peak indices (if peaks provided)
    - {output_path}_classifications.npy: Peak classifications (if peaks provided)

    Args:
        signal: SignalData to export
        peaks: PeakData (optional)
        output_path: Base output path (without extension)

    Returns:
        Path to signal .npy file
    """
    output_path = Path(output_path)
    stem = output_path.stem

    # Save signal
    signal_data = np.column_stack([signal.timestamps, signal.samples])
    signal_file = output_path.parent / f"{stem}_signal.npy"
    np.save(signal_file, signal_data)

    # Save peaks if provided
    if peaks is not None and peaks.num_peaks > 0:
        peaks_file = output_path.parent / f"{stem}_peaks.npy"
        np.save(peaks_file, peaks.indices)

        class_file = output_path.parent / f"{stem}_classifications.npy"
        np.save(class_file, peaks.classifications)

        logger.info(f"Exported signal + peaks to NPY: {signal_file}, {peaks_file}, {class_file}")
    else:
        logger.info(f"Exported signal to NPY: {signal_file}")

    return signal_file


def export_annotations(
    signal: SignalData,
    peaks: PeakData,
    output_path: Path | str,
) -> Path:
    """Export peak annotations as CSV with timestamps and classifications.

    CSV format:
    - peak_index: Sample index
    - time_s: Timestamp in seconds
    - amplitude: Signal value at peak
    - classification: Classification value (0=AUTO, 1=MANUAL, 2=ECTOPIC, 3=BAD)

    Args:
        signal: SignalData
        peaks: PeakData with annotations
        output_path: Output annotation CSV path

    Returns:
        Path to created annotation file
    """
    output_path = Path(output_path)

    if peaks.num_peaks == 0:
        logger.warning("No peaks to export")
        # Create empty file
        pd.DataFrame(columns=["peak_index", "time_s", "amplitude", "classification"]).to_csv(
            output_path, index=False
        )
        return output_path

    # Build annotation DataFrame
    df = pd.DataFrame({
        "peak_index": peaks.indices,
        "time_s": signal.timestamps[peaks.indices],
        "amplitude": signal.samples[peaks.indices],
        "classification": peaks.classifications,
    })

    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(df)} peak annotations to {output_path}")
    return output_path


def save_processing_parameters(
    pipeline_steps: list[ProcessingStep],
    signal_type: str,
    sampling_rate: float,
    output_path: Path | str,
) -> Path:
    """Save processing parameters as JSON sidecar for reproducibility.

    Args:
        pipeline_steps: List of ProcessingStep from pipeline
        signal_type: Signal type (ecg, ppg, eda)
        sampling_rate: Sampling rate in Hz
        output_path: Output JSON path

    Returns:
        Path to created JSON file
    """
    output_path = Path(output_path)

    # Serialize pipeline steps
    steps_data = [
        {
            "operation": step.operation,
            "parameters": step.parameters,
            "timestamp": step.timestamp,
        }
        for step in pipeline_steps
    ]

    data = {
        "signal_type": signal_type,
        "sampling_rate": sampling_rate,
        "processing_pipeline": steps_data,
        "software": "CardioSignalLab MVP",
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved processing parameters to {output_path}")
    return output_path
