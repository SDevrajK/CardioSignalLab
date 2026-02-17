"""R-R and N-N interval validity filtering for HRV analysis.

Provides two independent validity checks:

1. Physiological validity — rejects intervals outside the plausible human
   heart-rate range.  Default: 300-2000 ms (30-200 bpm).

2. Statistical validity — rejects intervals that are statistical outliers
   relative to the surrounding recording using a global MAD-based criterion.
   |x - median| > threshold * MAD  (default threshold = 3.0).
   MAD is preferred over SD because it is robust to the outliers we are
   trying to detect.

Terminology note:
- RR intervals: all consecutive inter-peak intervals, including ectopic beats
  and artifact intervals.  Validity flags are attached so the researcher can
  apply their own exclusion criteria.
- NN intervals: only intervals between two Normal (non-ectopic, non-artifact)
  beats where both the physiological and statistical criteria are satisfied.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


def flag_physiological(
    intervals_ms: np.ndarray,
    *,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
) -> np.ndarray:
    """Flag intervals within the physiologically plausible range.

    Args:
        intervals_ms: Inter-peak intervals in milliseconds (1D array)
        min_ms: Minimum valid interval in ms (default 300 ms = 200 bpm)
        max_ms: Maximum valid interval in ms (default 2000 ms = 30 bpm)

    Returns:
        Boolean array, True = physiologically valid
    """
    if len(intervals_ms) == 0:
        return np.array([], dtype=bool)

    valid = (intervals_ms >= min_ms) & (intervals_ms <= max_ms)

    n_invalid = int(np.sum(~valid))
    if n_invalid:
        logger.debug(
            f"Physiological filter: {n_invalid}/{len(intervals_ms)} intervals "
            f"outside [{min_ms:.0f}, {max_ms:.0f}] ms"
        )
    return valid


def flag_statistical(
    intervals_ms: np.ndarray,
    *,
    threshold: float = 3.0,
) -> np.ndarray:
    """Flag intervals that are not statistical outliers (MAD-based).

    Uses the global median absolute deviation (MAD) as a robust spread
    estimator.  An interval is flagged as invalid when:

        |x - median| > threshold * MAD

    MAD is preferred over SD for this purpose because it is not inflated by
    the outliers being tested.  A threshold of 3.0 is the conventional choice
    in the HRV literature (Clifford 2002, Malik 1996).

    Note: the full interval series (including physiologically invalid ones) is
    used to compute median/MAD so that the statistics are not biased by
    pre-filtering.  Apply physiological filtering independently.

    Args:
        intervals_ms: Inter-peak intervals in milliseconds (1D array)
        threshold: Number of MADs from the median for the rejection boundary
                   (default 3.0)

    Returns:
        Boolean array, True = statistically valid (not an outlier)
    """
    if len(intervals_ms) == 0:
        return np.array([], dtype=bool)

    median = np.median(intervals_ms)
    mad = np.median(np.abs(intervals_ms - median))

    if mad < 1e-6:
        # Perfectly flat series: every value is "valid" by this criterion
        logger.debug("Statistical filter: MAD is near-zero; all intervals marked valid")
        return np.ones(len(intervals_ms), dtype=bool)

    valid = np.abs(intervals_ms - median) <= threshold * mad

    n_invalid = int(np.sum(~valid))
    if n_invalid:
        logger.debug(
            f"Statistical filter (MAD x{threshold}): {n_invalid}/{len(intervals_ms)} "
            f"outliers  [median={median:.1f} ms, MAD={mad:.1f} ms, "
            f"bounds=({median - threshold * mad:.1f}, {median + threshold * mad:.1f}) ms]"
        )
    return valid
