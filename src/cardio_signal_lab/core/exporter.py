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
        # Peak marker and classification columns (sample-aligned)
        peak_marker = np.zeros(len(signal.samples), dtype=int)
        peak_marker[peaks.indices] = 1

        peak_classification = np.full(len(signal.samples), -1, dtype=int)
        peak_classification[peaks.indices] = peaks.classifications

        # N-N interval column: placed at each peak row, NaN elsewhere.
        # N-N[i] = time from peak i to peak i+1 (ms); NaN at last peak and non-peak rows.
        nn_ms_col = np.full(len(signal.samples), np.nan)
        if peaks.num_peaks > 1:
            peak_times = signal.timestamps[peaks.indices]
            nn_ms = np.diff(peak_times) * 1000.0
            nn_ms_col[peaks.indices[:-1]] = nn_ms

        df["peak"] = peak_marker
        df["peak_classification"] = peak_classification
        df["nn_interval_ms"] = nn_ms_col

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

        # N-N intervals (ms): sequential inter-peak differences, NaN for last peak
        peak_times = signal.timestamps[peaks.indices]
        nn_ms = np.full(len(peak_times), np.nan)
        if len(peak_times) > 1:
            nn_ms[:-1] = np.diff(peak_times) * 1000.0
        nn_file = output_path.parent / f"{stem}_nn_intervals_ms.npy"
        np.save(nn_file, nn_ms)

        logger.info(
            f"Exported signal + peaks to NPY: {signal_file}, {peaks_file}, "
            f"{class_file}, {nn_file}"
        )
    else:
        logger.info(f"Exported signal to NPY: {signal_file}")

    return signal_file


_CLASSIFICATION_LABELS = {0: "AUTO", 1: "MANUAL", 2: "ECTOPIC", 3: "BAD"}


def export_annotations(
    signal: SignalData,
    peaks: PeakData,
    output_path: Path | str,
) -> Path:
    """Export peak annotations as CSV with N-N intervals and classifications.

    CSV columns:
    - peak_index: Sample index in the signal array
    - time_s: Peak timestamp in seconds
    - amplitude: Signal amplitude at peak
    - classification: Numeric code (0=AUTO, 1=MANUAL, 2=ECTOPIC, 3=BAD)
    - classification_label: Human-readable label
    - nn_interval_ms: Interval to the *next* peak in milliseconds (NaN for
      the last peak). Computed from raw sequential timestamps regardless of
      classification so researchers can apply their own filtering criteria.

    Args:
        signal: SignalData (provides timestamps and samples)
        peaks: PeakData — peaks must be sorted by index (guaranteed by PeakEditor)
        output_path: Output annotation CSV path

    Returns:
        Path to created annotation file
    """
    output_path = Path(output_path)

    _EMPTY_COLS = [
        "peak_index", "time_s", "amplitude",
        "classification", "classification_label", "nn_interval_ms",
    ]

    if peaks.num_peaks == 0:
        logger.warning("No peaks to export")
        pd.DataFrame(columns=_EMPTY_COLS).to_csv(output_path, index=False)
        return output_path

    peak_times = signal.timestamps[peaks.indices]

    # N-N intervals: difference to next peak in ms; NaN for the final peak
    nn_ms = np.full(len(peak_times), np.nan)
    if len(peak_times) > 1:
        nn_ms[:-1] = np.diff(peak_times) * 1000.0

    classification_labels = [
        _CLASSIFICATION_LABELS.get(int(c), str(c)) for c in peaks.classifications
    ]

    df = pd.DataFrame({
        "peak_index": peaks.indices,
        "time_s": peak_times,
        "amplitude": signal.samples[peaks.indices],
        "classification": peaks.classifications,
        "classification_label": classification_labels,
        "nn_interval_ms": nn_ms,
    })

    df.to_csv(output_path, index=False)

    n_valid = int(np.sum(~np.isnan(nn_ms)))
    logger.info(
        f"Exported {len(df)} peaks to {output_path} "
        f"({n_valid} N-N intervals; mean={np.nanmean(nn_ms):.1f} ms)"
    )
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
