"""Derived signal computations (heart rate, EDA components).

Functions that produce secondary time-series from processed signals and
detected peaks. These are used exclusively for visualization; they do not
modify the primary signal and are not recorded in the processing pipeline.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cardio_signal_lab.core import PeakData, SignalData

from cardio_signal_lab.core.data_models import PeakClassification

# Only AUTO and MANUAL peaks are normal beats; ECTOPIC and BAD are excluded
# from heart rate calculation (treated like deleted peaks).
_VALID_CLASSIFICATIONS = {PeakClassification.AUTO.value, PeakClassification.MANUAL.value}


def compute_heart_rate(
    signal: SignalData,
    peaks: PeakData,
    rolling_window: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute instantaneous heart/pulse rate from inter-peak intervals.

    Only AUTO and MANUAL peaks contribute to the calculation. ECTOPIC and BAD
    peaks are excluded — they are treated like deleted peaks so that artifact
    beats do not distort the displayed heart rate or rolling average.

    Args:
        signal: SignalData providing timestamps for peak positions
        peaks: PeakData with detected peak indices
        rolling_window: Number of beats for rolling average (default 10)

    Returns:
        Tuple of (times, bpm, rolling_bpm):
          - times: midpoint timestamps between consecutive valid peaks (N-1,)
          - bpm: instantaneous rate at each interval in beats per minute (N-1,)
          - rolling_bpm: rolling-average BPM over `rolling_window` beats (N-1,)
        All arrays are empty if fewer than 2 valid peaks are available.
    """
    if peaks.num_peaks < 2:
        logger.debug("compute_heart_rate: fewer than 2 peaks, returning empty arrays")
        return np.array([]), np.array([]), np.array([])

    # Filter to AUTO and MANUAL peaks only
    valid_mask = np.isin(peaks.classifications, list(_VALID_CLASSIFICATIONS))
    valid_indices = peaks.indices[valid_mask]
    n_excluded = peaks.num_peaks - int(valid_mask.sum())
    if n_excluded > 0:
        logger.debug(f"compute_heart_rate: excluded {n_excluded} ECTOPIC/BAD peaks")

    if len(valid_indices) < 2:
        logger.debug("compute_heart_rate: fewer than 2 valid peaks after filtering")
        return np.array([]), np.array([]), np.array([])

    peak_times = signal.timestamps[valid_indices]
    rr = np.diff(peak_times)  # inter-peak intervals in seconds

    # Guard against zero or negative intervals (e.g. duplicate indices)
    valid = rr > 0
    if not np.all(valid):
        n_bad = np.sum(~valid)
        logger.warning(f"compute_heart_rate: {n_bad} non-positive RR intervals removed")
        rr = rr[valid]
        # Recompute times from valid pairs only
        valid_idx = np.where(valid)[0]
        peak_times = peak_times[np.concatenate([[valid_idx[0]], valid_idx + 1])]
        peak_times = np.unique(peak_times)  # de-duplicate

    if len(rr) == 0:
        return np.array([]), np.array([]), np.array([])

    bpm = 60.0 / rr
    mid_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    # Rolling average with a centred window (mode='same' pads edges)
    w = min(rolling_window, len(bpm))
    kernel = np.ones(w) / w
    rolling_bpm = np.convolve(bpm, kernel, mode="same")

    logger.debug(
        f"compute_heart_rate: {len(bpm)} intervals, "
        f"mean={bpm.mean():.1f} bpm, rolling_window={w}"
    )
    return mid_times, bpm, rolling_bpm
