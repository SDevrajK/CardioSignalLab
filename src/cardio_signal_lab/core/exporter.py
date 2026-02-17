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

from cardio_signal_lab.core.data_models import (
    EventData,
    PeakClassification,
    PeakData,
    ProcessingStep,
    SignalData,
)


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


def export_intervals(
    signal: SignalData,
    peaks: PeakData,
    output_path: Path | str,
    *,
    mode: str = "rr",
    events: list[EventData] | None = None,
    min_rr_ms: float = 300.0,
    max_rr_ms: float = 2000.0,
    stat_threshold: float = 3.0,
) -> Path:
    """Export per-interval RR or NN interval table as CSV.

    Each row represents one consecutive inter-peak interval.

    RR mode columns:
        interval_index, rr_interval_ms, peak1_time_s, peak2_time_s,
        peak1_index, peak2_index, heart_rate_bpm,
        peak1_classification, peak2_classification,
        physiologically_valid, statistically_valid
        [, event_label]  -- if events are provided

    NN mode: same as RR but rows are pre-filtered to only those where
        physiologically_valid AND statistically_valid AND neither peak is
        ECTOPIC or BAD.  The validity columns are omitted (all rows would
        be True).  The interval column is renamed nn_interval_ms.

    Args:
        signal: SignalData providing timestamps
        peaks: PeakData with peak indices and classifications
        output_path: Output CSV path
        mode: "rr" (keep all intervals + validity flags) or
              "nn" (filter to clean intervals only)
        events: Optional list of EventData; if provided, each interval is
                annotated with the label of the most recent event preceding
                its midpoint
        min_rr_ms: Physiological minimum interval in ms (default 300)
        max_rr_ms: Physiological maximum interval in ms (default 2000)
        stat_threshold: MAD multiplier for the statistical outlier test
                        (default 3.0)

    Returns:
        Path to created CSV file

    Raises:
        ValueError: If mode is not "rr" or "nn"
    """
    from cardio_signal_lab.processing.interval_analysis import (
        flag_physiological,
        flag_statistical,
    )

    if mode not in ("rr", "nn"):
        raise ValueError(f"mode must be 'rr' or 'nn', got {mode!r}")

    output_path = Path(output_path)

    if peaks.num_peaks < 2:
        logger.warning("Fewer than 2 peaks — cannot compute intervals; writing empty file")
        cols = ["interval_index", "rr_interval_ms" if mode == "rr" else "nn_interval_ms"]
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        return output_path

    # Sort peaks by sample index (PeakEditor guarantees this, but be safe)
    sort_order = np.argsort(peaks.indices)
    sorted_indices = peaks.indices[sort_order]
    sorted_classifs = peaks.classifications[sort_order]

    peak_times = signal.timestamps[sorted_indices]

    # Build per-interval arrays (N-1 intervals for N peaks)
    n_intervals = len(sorted_indices) - 1
    interval_idx = np.arange(n_intervals)
    rr_ms = np.diff(peak_times) * 1000.0
    p1_times = peak_times[:-1]
    p2_times = peak_times[1:]
    p1_indices = sorted_indices[:-1]
    p2_indices = sorted_indices[1:]
    hr_bpm = 60_000.0 / rr_ms
    p1_labels = [_CLASSIFICATION_LABELS.get(int(c), str(c)) for c in sorted_classifs[:-1]]
    p2_labels = [_CLASSIFICATION_LABELS.get(int(c), str(c)) for c in sorted_classifs[1:]]

    # Validity flags
    phys_valid = flag_physiological(rr_ms, min_ms=min_rr_ms, max_ms=max_rr_ms)
    stat_valid = flag_statistical(rr_ms, threshold=stat_threshold)

    # Classification-based validity: interval is "normal" only if neither
    # boundary peak is ectopic or bad
    _BAD_CLASSES = {PeakClassification.ECTOPIC.value, PeakClassification.BAD.value}
    class_valid = np.array([
        (int(c1) not in _BAD_CLASSES) and (int(c2) not in _BAD_CLASSES)
        for c1, c2 in zip(sorted_classifs[:-1], sorted_classifs[1:])
    ])

    df = pd.DataFrame({
        "interval_index": interval_idx,
        "rr_interval_ms": rr_ms,
        "peak1_time_s": p1_times,
        "peak2_time_s": p2_times,
        "peak1_index": p1_indices,
        "peak2_index": p2_indices,
        "heart_rate_bpm": hr_bpm,
        "peak1_classification": p1_labels,
        "peak2_classification": p2_labels,
        "physiologically_valid": phys_valid,
        "statistically_valid": stat_valid,
    })

    # Optional event annotation: label of the most recent event before each
    # interval midpoint (or "before_first_event" if none has started yet)
    if events:
        event_times = np.array([e.timestamp for e in events])
        event_labels_arr = [e.label for e in events]
        midpoints = (p1_times + p2_times) / 2.0

        interval_event_labels = []
        for mid in midpoints:
            preceding = np.where(event_times <= mid)[0]
            if len(preceding) == 0:
                interval_event_labels.append("before_first_event")
            else:
                interval_event_labels.append(event_labels_arr[preceding[-1]])
        df["event_label"] = interval_event_labels

    if mode == "rr":
        df.to_csv(output_path, index=False)
        n_phys = int(np.sum(phys_valid))
        n_stat = int(np.sum(stat_valid))
        logger.info(
            f"Exported {n_intervals} RR intervals to {output_path} "
            f"(physiologically valid: {n_phys}, statistically valid: {n_stat})"
        )
    else:
        # NN mode: keep only fully valid intervals
        nn_mask = phys_valid & stat_valid & class_valid
        df_nn = df[nn_mask].copy()
        df_nn = df_nn.rename(columns={"rr_interval_ms": "nn_interval_ms"})
        df_nn = df_nn.drop(columns=["physiologically_valid", "statistically_valid"])
        df_nn = df_nn.reset_index(drop=True)
        df_nn["interval_index"] = df_nn.index
        df_nn.to_csv(output_path, index=False)
        logger.info(
            f"Exported {len(df_nn)}/{n_intervals} NN intervals to {output_path} "
            f"({n_intervals - len(df_nn)} excluded)"
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
