"""Bad segment detection and interpolation for physiological signals.

Detects two classes of artifacts:
1. Amplitude artifacts - large transients detected via rolling MAD threshold
   (e.g., motion pop, electrode displacement, hardware glitches)
2. Timestamp gaps - missing data detected via jumps in the timestamp sequence

Both return sample-index ranges. Detected segments can be repaired with
cubic spline interpolation using anchor points from clean signal on each side.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline

from cardio_signal_lab.core.data_models import BadSegment
from cardio_signal_lab.processing.pipeline import register_operation


def _merge_overlapping(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent index ranges.

    Args:
        segments: List of (start_idx, end_idx) tuples (may be unsorted)

    Returns:
        Sorted, merged list of non-overlapping ranges
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged: list[tuple[int, int]] = [sorted_segs[0]]

    for start, end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            # Overlapping or adjacent: extend
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def detect_amplitude_artifacts(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    mad_threshold: float = 4.0,
    window_s: float = 10.0,
    dilation_s: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect large-amplitude transient artifacts using a rolling MAD threshold.

    Computes rolling median and MAD over a window_s-second window and flags
    samples where |x - median| > mad_threshold * MAD. The flagged region is
    then dilated by dilation_s seconds on each side to capture ringing edges.

    Suitable for detecting hardware pop/motion artifacts in PPG/ECG that span
    multiple samples (unlike single-sample spike detectors).

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        mad_threshold: Flag threshold as multiples of rolling MAD (default 4.0)
        window_s: Rolling window duration in seconds (default 10.0)
        dilation_s: Dilation applied to each side of flagged region in seconds (default 0.3)

    Returns:
        List of (start_idx, end_idx) index pairs for bad segments (merged)
    """
    n = len(samples)
    if n < 4:
        return []

    half_win = max(1, int(window_s * sampling_rate / 2))
    dilation = max(1, int(dilation_s * sampling_rate))

    # Compute rolling median and MAD via a simple sliding window.
    # For large signals this is O(n * window) but window_s is typically 10s
    # at common physiological sampling rates (50-256 Hz) so it stays fast.
    flagged = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        window = samples[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad < 1e-10:
            # Flat region: any non-zero deviation counts as artifact
            if abs(samples[i] - med) > 1e-10:
                flagged[i] = True
        elif abs(samples[i] - med) > mad_threshold * mad:
            flagged[i] = True

    # Dilate flagged region
    flagged_indices = np.where(flagged)[0]
    if len(flagged_indices) == 0:
        return []

    dilated = np.zeros(n, dtype=bool)
    for idx in flagged_indices:
        lo = max(0, idx - dilation)
        hi = min(n, idx + dilation + 1)
        dilated[lo:hi] = True

    # Convert boolean mask to contiguous (start, end) ranges
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start = 0
    for i, val in enumerate(dilated):
        if val and not in_segment:
            in_segment = True
            seg_start = i
        elif not val and in_segment:
            in_segment = False
            segments.append((seg_start, i - 1))
    if in_segment:
        segments.append((seg_start, n - 1))

    result = _merge_overlapping(segments)
    logger.debug(
        f"Amplitude artifact detection: {len(result)} segment(s), "
        f"MAD threshold={mad_threshold}, window={window_s}s, dilation={dilation_s}s"
    )
    return result


def detect_timestamp_gaps(
    timestamps: np.ndarray,
    sampling_rate: float,
    *,
    gap_multiplier: float = 2.0,
) -> list[tuple[int, int]]:
    """Detect missing data by finding timestamp jumps larger than expected.

    A gap at index i means the device produced no samples in the interval
    [timestamps[i], timestamps[i+1]]. The returned segment spans [i, i+1]
    to mark both the last sample before and the first sample after the gap.

    Args:
        timestamps: Timestamp array in seconds (monotonically increasing)
        sampling_rate: Expected sampling rate in Hz
        gap_multiplier: Minimum gap size as a multiple of the expected interval (default 2.0)

    Returns:
        List of (start_idx, end_idx) index pairs for gap locations
    """
    if len(timestamps) < 2:
        return []

    expected_interval = 1.0 / sampling_rate
    diffs = np.diff(timestamps)
    gap_threshold = expected_interval * gap_multiplier
    gap_at = np.where(diffs > gap_threshold)[0]  # Index of sample just before each gap

    segments = [(int(i), int(i + 1)) for i in gap_at]

    logger.debug(
        f"Timestamp gap detection: {len(segments)} gap(s), "
        f"threshold={gap_multiplier}x expected interval ({expected_interval * 1000:.2f} ms)"
    )
    return segments


def detect_bad_segments(
    samples: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float,
    *,
    mad_threshold: float = 4.0,
    window_s: float = 10.0,
    dilation_s: float = 0.3,
    gap_multiplier: float = 2.0,
) -> list[BadSegment]:
    """Detect bad segments from both amplitude artifacts and timestamp gaps.

    Combines amplitude-based and gap-based detection, merges overlapping
    ranges, and returns typed BadSegment objects.

    Args:
        samples: Signal samples (1D array)
        timestamps: Timestamp array (same length as samples)
        sampling_rate: Sampling rate in Hz
        mad_threshold: MAD threshold multiplier for amplitude detection
        window_s: Rolling window for amplitude detection in seconds
        dilation_s: Dilation in seconds applied to each flagged amplitude region
        gap_multiplier: Minimum gap size as multiple of expected interval

    Returns:
        List of BadSegment objects (merged, sorted by start index)
    """
    amplitude_segs = detect_amplitude_artifacts(
        samples, sampling_rate,
        mad_threshold=mad_threshold, window_s=window_s, dilation_s=dilation_s
    )
    gap_segs = detect_timestamp_gaps(
        timestamps, sampling_rate, gap_multiplier=gap_multiplier
    )

    # Tag by source before merging
    tagged: list[tuple[int, int, str]] = (
        [(s, e, "amplitude") for s, e in amplitude_segs]
        + [(s, e, "gap") for s, e in gap_segs]
    )
    tagged.sort(key=lambda x: x[0])

    # Merge overlapping ranges, preserving sources
    if not tagged:
        logger.info("Bad segment detection: no segments found")
        return []

    # Build merged list (merge ranges that overlap; keep both sources if merged)
    merged: list[tuple[int, int, str]] = [tagged[0]]
    for start, end, source in tagged[1:]:
        prev_start, prev_end, prev_source = merged[-1]
        if start <= prev_end + 1:
            combined_source = prev_source if prev_source == source else "amplitude+gap"
            merged[-1] = (prev_start, max(prev_end, end), combined_source)
        else:
            merged.append((start, end, source))

    result = [
        BadSegment(start_idx=s, end_idx=e, source=src)
        for s, e, src in merged
    ]

    n_amplitude = sum(1 for s, e, src in tagged if src == "amplitude")
    n_gap = sum(1 for s, e, src in tagged if src == "gap")
    logger.info(
        f"Bad segment detection complete: {len(result)} segment(s) "
        f"({n_amplitude} amplitude, {n_gap} gap)"
    )
    return result


def interpolate_bad_segments(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    segments: list[list[int]],
    anchor_s: float = 2.0,
) -> np.ndarray:
    """Repair bad segments with cubic spline interpolation.

    For each bad segment, selects anchor points from clean signal on both
    sides (up to anchor_s seconds worth of samples, capped at 5 full PPG/ECG
    cycles) and fits a cubic spline through them. The spline is evaluated
    over the bad segment indices to produce a physiologically plausible
    replacement.

    This is the standard approach for multi-sample transient artifact repair
    described in Orphanidou (2015) and Elgendi (2016).

    Args:
        samples: Signal samples (1D array) — pipeline convention
        sampling_rate: Sampling rate in Hz — pipeline convention
        segments: List of [start_idx, end_idx] pairs (JSON-serializable from pipeline params)
        anchor_s: Anchor region duration in seconds on each side of the bad segment (default 2.0)

    Returns:
        Signal with bad segments replaced by cubic spline interpolation
    """
    if not segments:
        return samples.copy()

    n = len(samples)
    result = samples.copy()
    anchor_n = max(4, int(anchor_s * sampling_rate))

    for seg in segments:
        start_idx, end_idx = int(seg[0]), int(seg[1])
        start_idx = max(0, start_idx)
        end_idx = min(n - 1, end_idx)

        if end_idx < start_idx:
            continue

        # Gather clean anchor indices from left side
        left_end = start_idx - 1
        left_start = max(0, left_end - anchor_n + 1)
        left_indices = np.arange(left_start, left_end + 1)

        # Gather clean anchor indices from right side
        right_start = end_idx + 1
        right_end = min(n - 1, right_start + anchor_n - 1)
        right_indices = np.arange(right_start, right_end + 1)

        has_left = len(left_indices) >= 2
        has_right = len(right_indices) >= 2
        bad_indices = np.arange(start_idx, end_idx + 1)

        if not has_left and not has_right:
            logger.warning(
                f"Bad segment [{start_idx}, {end_idx}]: no clean data on either side, skipping"
            )
            continue

        if not has_left:
            # Left edge of signal: no anchor on the left.
            # Cubic extrapolation backward is unreliable, so fill with the
            # mean of the right anchor region (stable baseline estimate).
            result[bad_indices] = np.mean(result[right_indices])
            logger.debug(
                f"Left-edge bad segment [{start_idx}, {end_idx}]: "
                f"filled with right-anchor mean ({end_idx - start_idx + 1} samples)"
            )
            continue

        if not has_right:
            # Right edge of signal: no anchor on the right.
            result[bad_indices] = np.mean(result[left_indices])
            logger.debug(
                f"Right-edge bad segment [{start_idx}, {end_idx}]: "
                f"filled with left-anchor mean ({end_idx - start_idx + 1} samples)"
            )
            continue

        # Both sides available: cubic spline through anchor points on each side
        anchor_indices = np.concatenate([left_indices, right_indices])
        anchor_values = result[anchor_indices]

        try:
            # extrapolate=False: no extrapolation past anchor range (both sides present)
            cs = CubicSpline(anchor_indices, anchor_values, extrapolate=False)
            result[bad_indices] = cs(bad_indices)
        except Exception as exc:
            logger.warning(f"Bad segment [{start_idx}, {end_idx}]: interpolation failed ({exc}), skipping")
            continue

        logger.debug(f"Interpolated bad segment [{start_idx}, {end_idx}] ({end_idx - start_idx + 1} samples)")

    n_segs = len(segments)
    logger.info(f"Bad segment interpolation complete: {n_segs} segment(s) repaired")
    return result


# --- Pipeline operation wrapper ---

def _pipeline_interpolate_bad_segments(samples, sampling_rate, **params):
    return interpolate_bad_segments(samples, sampling_rate, **params)


register_operation("interpolate_bad_segments", _pipeline_interpolate_bad_segments)
