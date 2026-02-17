"""Import helpers for replacing or supplementing data loaded from XDF/CSV.

Two importers are provided:

load_events_csv(path)
    Replace XDF event markers with events from an external CSV.
    Auto-detects timestamp and label columns by common name patterns.

load_peaks_binary_csv(path, signal_length)
    Load pre-corrected peak positions from a binary (0/1) CSV column,
    as produced by the EKG_Peak_Corrector and similar tools.
    Column name is expected to contain "peak" (case-insensitive).
    All imported peaks are classified MANUAL.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cardio_signal_lab.core.data_models import EventData, PeakClassification, PeakData


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

# Column name substrings (lower-case) tried in order for timestamp detection
_TIME_CANDIDATES = ["time_s", "time", "timestamp", "onset", "t"]
# Column name substrings tried in order for label detection
_LABEL_CANDIDATES = ["label", "event", "name", "description", "marker", "code"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column whose lower-cased name contains any candidate."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_low, col_orig in cols_lower.items():
            if cand in col_low:
                return col_orig
    return None


def load_events_csv(path: Path | str) -> list[EventData]:
    """Load event markers from a CSV file.

    Expected columns (detected by name, case-insensitive):
    - Timestamp: any column whose name contains 'time', 'timestamp', or 'onset'
    - Label:     any column whose name contains 'label', 'event', or 'name'

    If auto-detection fails, column 0 is used as timestamp and column 1 as label.

    Args:
        path: Path to the CSV file

    Returns:
        List of EventData sorted by timestamp

    Raises:
        ValueError: If the file has fewer than 2 columns or no numeric timestamp column
        FileNotFoundError: If path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"Events CSV is empty: {path}")
        return []

    # Detect timestamp column
    time_col = _find_column(df, _TIME_CANDIDATES)
    if time_col is None:
        # Fall back to first numeric column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric column found for timestamps in {path.name}")
        time_col = numeric_cols[0]
        logger.warning(f"Timestamp column not identified by name; using '{time_col}'")
    else:
        logger.debug(f"Using '{time_col}' as timestamp column")

    # Detect label column
    label_col = _find_column(df, _LABEL_CANDIDATES)
    if label_col is None:
        # Fall back to first non-numeric column, then first column
        str_cols = df.select_dtypes(exclude="number").columns.tolist()
        label_col = str_cols[0] if str_cols else df.columns[1] if len(df.columns) > 1 else time_col
        logger.warning(f"Label column not identified by name; using '{label_col}'")
    else:
        logger.debug(f"Using '{label_col}' as label column")

    events = []
    for _, row in df.iterrows():
        try:
            t = float(row[time_col])
            label = str(row[label_col]).strip()
            if label == "" or label == "nan":
                label = "event"
            events.append(EventData(timestamp=t, label=label))
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping malformed event row: {row.to_dict()} ({e})")

    events.sort(key=lambda ev: ev.timestamp)
    logger.info(
        f"Loaded {len(events)} events from {path.name} "
        f"(time_col='{time_col}', label_col='{label_col}')"
    )
    return events


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------

def load_peaks_binary_csv(path: Path | str, signal_length: int) -> PeakData:
    """Load corrected peaks from a binary (0/1) CSV column.

    The file must contain a column whose name includes 'peak' (case-insensitive),
    e.g. 'PPG_Peaks', 'ECG_Peaks', 'peaks'.  Each row corresponds to one signal
    sample: 1 = peak, 0 = no peak.

    All imported peaks are classified MANUAL (shown green, included in heart
    rate calculation).  If the file contains a 'classification' or 'label' column
    that matches PeakClassification names (AUTO/MANUAL/ECTOPIC/BAD), those values
    are used instead.

    Args:
        path: Path to the CSV file
        signal_length: Expected number of samples (used for bounds checking)

    Returns:
        PeakData with sorted indices and MANUAL classifications

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If no peaks column is found or peak indices are out of range
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peaks file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"Peaks CSV is empty: {path}")
        return PeakData(indices=np.array([], dtype=int), classifications=np.array([], dtype=int))

    # Find the binary peaks column
    peak_col = None
    for col in df.columns:
        if "peak" in col.lower():
            peak_col = col
            break

    if peak_col is None:
        # Last resort: try first column if it contains only 0s and 1s
        first_col = df.columns[0]
        unique_vals = set(df[first_col].dropna().unique())
        if unique_vals <= {0, 1, 0.0, 1.0}:
            peak_col = first_col
            logger.warning(
                f"No column with 'peak' in name found; using '{peak_col}' "
                f"(contains only 0/1 values)"
            )
        else:
            raise ValueError(
                f"No peaks column found in {path.name}. "
                f"Expected a column whose name contains 'peak' (e.g. 'PPG_Peaks')."
            )

    logger.debug(f"Using '{peak_col}' as peaks column from {path.name}")

    binary = df[peak_col].to_numpy()
    indices = np.where(binary == 1)[0].astype(int)

    # Bounds check
    out_of_range = indices[indices >= signal_length]
    if len(out_of_range) > 0:
        logger.warning(
            f"{len(out_of_range)} peak indices >= signal_length ({signal_length}); "
            f"they will be dropped"
        )
        indices = indices[indices < signal_length]

    # Classification: MANUAL by default; honour a classification column if present
    classification_col = _find_column(df, ["classification", "class", "label"])
    _NAME_TO_VAL = {c.name: c.value for c in PeakClassification}

    if classification_col is not None and classification_col != peak_col:
        # Extract classification for peak rows only
        cls_series = df[classification_col].iloc[indices]
        classifications = np.full(len(indices), PeakClassification.MANUAL.value, dtype=int)
        for i, val in enumerate(cls_series):
            try:
                v = int(val)
                if v in {c.value for c in PeakClassification}:
                    classifications[i] = v
                    continue
            except (ValueError, TypeError):
                pass
            # Try string name
            name = str(val).strip().upper()
            if name in _NAME_TO_VAL:
                classifications[i] = _NAME_TO_VAL[name]
        logger.debug(f"Read classifications from '{classification_col}' column")
    else:
        classifications = np.full(len(indices), PeakClassification.MANUAL.value, dtype=int)

    logger.info(
        f"Loaded {len(indices)} peaks from {path.name} "
        f"(column='{peak_col}', all MANUAL)"
    )
    return PeakData(indices=indices, classifications=classifications)
