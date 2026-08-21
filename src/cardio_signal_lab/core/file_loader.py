"""File loading with Protocol-based architecture for extensibility.

Supports XDF, CSV, MATLAB .mat, and EDF/EDF+/BDF files with automatic signal
type detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import pyxdf
from loguru import logger

from cardio_signal_lab.core.data_models import EventData, RecordingSession, SignalData, SignalType


def detect_signal_type_from_name(name: str) -> SignalType:
    """Auto-detect signal type from a channel or column name.

    Args:
        name: Channel name or column name

    Returns:
        Detected SignalType
    """
    name_upper = name.upper()

    if "ECG" in name_upper:
        return SignalType.ECG
    elif "GSR" in name_upper or "EDA" in name_upper or "ELECTRODERMAL" in name_upper:
        return SignalType.EDA
    elif "PPG" in name_upper or "BVP" in name_upper or "PLETH" in name_upper:
        return SignalType.PPG
    elif "ADC A13" in name_upper or "INTERNAL ADC" in name_upper:
        # Shimmer device uses Internal ADC A13 for PPG
        return SignalType.PPG
    else:
        return SignalType.UNKNOWN


class FileLoader(Protocol):
    """Protocol for file loaders.

    Implementing classes must provide can_load() and load() methods.
    """

    def can_load(self, path: Path) -> bool:
        """Check if this loader can handle the given file.

        Args:
            path: Path to file

        Returns:
            True if this loader can handle the file
        """
        ...

    def load(self, path: Path) -> RecordingSession:
        """Load file and return RecordingSession.

        Args:
            path: Path to file

        Returns:
            RecordingSession with loaded signals

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        ...


class XdfLoader:
    """Loader for XDF (Extensible Data Format) files.

    Loads physiological signals from LSL recordings with automatic
    signal type detection based on stream metadata.
    """

    def __init__(self, apply_lsl_alignment: bool = False):
        """Initialize XDF loader.

        Args:
            apply_lsl_alignment: Use LSL timestamps for alignment (default: False, uses device timestamps)
        """
        self.apply_lsl_alignment = apply_lsl_alignment
        self.skipped_streams = []  # Track streams that failed to load

    def can_load(self, path: Path) -> bool:
        """Check if file is XDF format."""
        return path.suffix.lower() == ".xdf"

    def load(self, path: Path) -> RecordingSession:
        """Load XDF file.

        Args:
            path: Path to XDF file

        Returns:
            RecordingSession with loaded signals

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If XDF format is invalid
        """
        # Reset skipped streams list for this load
        self.skipped_streams = []

        # Layered validation: type → exists → format
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception:
                raise TypeError(f"path must be a Path object or string, got {type(path).__name__}")

        if not path.exists():
            raise FileNotFoundError(f"XDF file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".xdf":
            raise ValueError(f"Invalid file extension, expected .xdf, got {path.suffix}")

        logger.info(f"Loading XDF file: {path}")

        # Load XDF with selective stream loading
        streams, header = self._load_xdf_streams(path)

        if not streams:
            raise ValueError(f"No streams found in XDF file: {path}")

        # Determine LSL t0 reference from the first physiological signal stream.
        # All subsequent streams and event markers are zero-referenced to this value
        # so that everything shares the same time axis.
        lsl_t0_reference: float | None = None
        for stream in streams:
            if not isinstance(stream["time_series"], list):  # physiological stream
                lsl_t0_reference = float(stream["time_stamps"][0])
                logger.debug(f"LSL t0 reference: {lsl_t0_reference:.6f} s (absolute LSL clock)")
                break

        if lsl_t0_reference is None:
            raise ValueError(f"No physiological signal streams found in XDF file: {path}")

        # Extract signals from streams (skip marker streams)
        signals = []
        for stream in streams:
            # Skip event/marker streams (they have time_series as list, not array)
            if isinstance(stream["time_series"], list):
                continue
            signal_list = self._extract_signals_from_stream(stream, lsl_t0_reference)
            signals.extend(signal_list)

        if not signals:
            raise ValueError(f"No physiological signals found in XDF file: {path}")

        logger.info(f"Loaded {len(signals)} signals from XDF file")

        # Extract events from marker streams, zero-referenced to the same LSL t0
        events = self._extract_events_from_streams(streams, lsl_t0_reference)

        return RecordingSession(
            source_path=path,
            signals=signals,
            events=events,
            lsl_t0_reference=lsl_t0_reference,
        )

    def _load_xdf_streams(self, path: Path) -> tuple:
        """Load XDF streams with selective loading.

        Returns:
            Tuple of (streams, header)

        Raises:
            ValueError: If file has structural corruption (length mismatch)
        """
        # Define selective stream queries for physiological signals and events
        select_queries = [
            {"name": "ECG"},
            {"type": "ECG"},
            {"name": "PPG"},
            {"type": "PPG"},
            {"name": "GSR"},
            {"type": "GSR"},
            {"name": "EDA"},
            {"type": "EDA"},
            {"type": "shimmer"},  # Shimmer device streams
            {"type": "Markers"},  # Event marker streams
            {"type": "markers"},
            {"type": "Events"},
            {"type": "events"},
        ]

        try:
            logger.debug("Attempting selective stream loading")
            streams, header = pyxdf.load_xdf(str(path), select_streams=select_queries)

            if not streams:
                logger.warning("No streams matched queries, loading all streams")
                streams, header = pyxdf.load_xdf(str(path))
        except Exception as e:
            # Check for struct.error which indicates file truncation/length mismatch
            if "unpack requires a buffer" in str(e):
                logger.error(f"File has structural corruption (length mismatch): {e}")
                raise ValueError(
                    f"XDF file is corrupted with a length mismatch between data samples and timestamps. "
                    f"This cannot be automatically fixed. "
                    f"Please re-save the file in MATLAB using the standard load/save procedure. "
                    f"This will properly align all data streams. "
                    f"(Technical: {e})"
                )

            logger.warning(f"Selective loading failed: {e}, loading all streams")
            try:
                streams, header = pyxdf.load_xdf(str(path))
            except Exception as e2:
                if "unpack requires a buffer" in str(e2):
                    logger.error(f"File has structural corruption (length mismatch): {e2}")
                    raise ValueError(
                        f"XDF file is corrupted with a length mismatch between data samples and timestamps. "
                        f"This cannot be automatically fixed. "
                        f"Please re-save the file in MATLAB using the standard load/save procedure. "
                        f"This will properly align all data streams. "
                        f"(Technical: {e2})"
                    )
                raise

        logger.info(f"Loaded {len(streams)} streams from XDF")
        return streams, header

    def _extract_signals_from_stream(self, stream: dict, lsl_t0_reference: float) -> list[SignalData]:
        """Extract SignalData objects from an XDF stream.

        4-tier fallback strategy for corrupted timestamps:
        1. Use LSL timestamps if clean
        2. If LSL corrupted but fixable, repair and use
        3. If not fixable, try device timestamps
        4. If device timestamps also corrupted, skip the signal

        Args:
            stream: XDF stream dict
            lsl_t0_reference: First LSL timestamp of the primary signal stream.
                Used to zero-reference all timestamps to a common time axis.

        Returns:
            List of SignalData objects (one per channel)
        """
        signals = []

        # Extract stream metadata
        stream_name = stream["info"]["name"][0] if isinstance(stream["info"]["name"], list) else stream["info"]["name"]
        sampling_rate = float(stream["info"]["nominal_srate"][0])

        # Validate sampling rate
        if not self._validate_sampling_rate(sampling_rate, stream_name):
            logger.error(f"Skipping stream '{stream_name}' due to invalid sampling rate")
            self.skipped_streams.append(stream_name)
            return []

        # Detect signal type from stream metadata (used as fallback)
        stream_signal_type = self._detect_signal_type(stream["info"])

        # Extract channel names and time series
        channel_names = self._extract_channel_names(stream["info"])
        time_series = stream["time_series"]
        lsl_timestamps = stream["time_stamps"].astype(np.float64)

        # Handle timestamp column (first column in Shimmer streams)
        has_device_timestamps = False
        if "shimmer" in stream_name.lower() or time_series.shape[1] == len(channel_names):
            # First column is device timestamp
            has_device_timestamps = True
            start_col = 1
            channel_names = channel_names[1:]  # Skip timestamp column
            device_timestamps_raw = time_series[:, 0].astype(np.float64)
        else:
            start_col = 0
            device_timestamps_raw = None

        # TIER 1: Try LSL timestamps first
        lsl_timestamps_zeroed = lsl_timestamps - lsl_t0_reference
        lsl_corruption_info = self._assess_timestamp_corruption(lsl_timestamps_zeroed, stream_name, "LSL")
        valid_mask = None

        if lsl_corruption_info["is_clean"]:
            logger.info(f"Using LSL timestamps for '{stream_name}' (clean)")
            timestamps = lsl_timestamps_zeroed
            timestamp_source = "LSL"
        # TIER 2: Try to repair LSL timestamps if fixable
        elif lsl_corruption_info["is_fixable"]:
            logger.warning(f"LSL timestamps corrupted but fixable for '{stream_name}': {lsl_corruption_info['backward_jumps']} backward jumps")
            result = self._repair_lsl_timestamps(lsl_timestamps_zeroed, stream_name)
            if result is not None:
                timestamps, valid_mask = result
                timestamp_source = "LSL (repaired)"
                logger.info(f"Successfully repaired LSL timestamps for '{stream_name}'")
            else:
                timestamps = None
        else:
            timestamps = None

        # TIER 3: If LSL failed, try device timestamps
        if timestamps is None and has_device_timestamps:
            device_timestamps_s = device_timestamps_raw / 1000.0  # Convert ms to seconds
            device_corruption_info = self._assess_timestamp_corruption(device_timestamps_s, stream_name, "Device")

            if device_corruption_info["is_clean"]:
                logger.info(f"Falling back to device timestamps for '{stream_name}' (clean)")
                timestamps = device_timestamps_s - device_timestamps_s[0]  # Zero-reference
                timestamp_source = "Device"
            elif device_corruption_info["is_fixable"]:
                logger.warning(f"Device timestamps corrupted but fixable for '{stream_name}'")
                result = self._repair_lsl_timestamps(device_timestamps_s - device_timestamps_s[0], stream_name)
                if result is not None:
                    timestamps, valid_mask = result
                    timestamp_source = "Device (repaired)"
                    logger.info(f"Successfully repaired device timestamps for '{stream_name}'")
            else:
                logger.error(f"Both LSL and device timestamps severely corrupted for '{stream_name}' ({device_corruption_info['backward_jumps']} backward jumps) - skipping all channels")

        # TIER 4: If all timestamp sources failed, skip this stream
        if timestamps is None:
            logger.error(f"Failed to obtain usable timestamps for stream '{stream_name}' - skipping")
            self.skipped_streams.append(stream_name)
            return []

        # Create SignalData for each channel
        for i, channel_name in enumerate(channel_names):
            col_idx = start_col + i
            if col_idx < time_series.shape[1]:
                samples = time_series[:, col_idx].astype(np.float64)

                # Filter samples if timestamps were repaired
                if valid_mask is not None:
                    samples = samples[valid_mask]

                # Skip channels with all zeros or NaNs
                if np.all(samples == 0) or np.all(np.isnan(samples)):
                    logger.debug(f"Skipping empty channel: {channel_name}")
                    continue

                # Validate signal values
                if not self._validate_signal_values(samples, channel_name):
                    logger.warning(f"Signal '{channel_name}' failed validation, but including in session")

                # Per-channel signal type detection
                channel_type = detect_signal_type_from_name(channel_name)
                if channel_type == SignalType.UNKNOWN:
                    channel_type = stream_signal_type

                # Filter LSL timestamps if applicable
                lsl_ts_filtered = lsl_timestamps if valid_mask is None else lsl_timestamps[valid_mask]

                signal = SignalData(
                    samples=samples,
                    sampling_rate=sampling_rate,
                    timestamps=timestamps,
                    channel_name=channel_name,
                    signal_type=channel_type,
                    lsl_timestamps=lsl_ts_filtered,
                )
                signals.append(signal)

        logger.debug(f"Extracted {len(signals)} signals from stream '{stream_name}' (timestamps: {timestamp_source})")
        return signals

    def _extract_channel_names(self, stream_info: dict) -> list[str]:
        """Extract channel names from stream info.

        Args:
            stream_info: XDF stream info dict

        Returns:
            List of channel names
        """
        try:
            channels = stream_info["desc"][0]["channels"][0]["channel"]
            if isinstance(channels, list):
                names = []
                for ch in channels:
                    if isinstance(ch["label"], list):
                        names.append(ch["label"][0])
                    else:
                        names.append(ch["label"])
                return names
            else:
                # Single channel
                return [channels["label"][0] if isinstance(channels["label"], list) else channels["label"]]
        except (KeyError, IndexError, TypeError):
            # Fallback: generate channel names
            n_channels = stream_info["channel_count"][0]
            return [f"Channel_{i+1}" for i in range(int(n_channels))]

    def _detect_signal_type(self, stream_info: dict) -> SignalType:
        """Detect signal type from stream metadata.

        Checks both 'name' and 'type' fields in stream info for signal classification.

        Args:
            stream_info: XDF stream info dict

        Returns:
            Detected SignalType (ECG, PPG, EDA, or UNKNOWN)
        """
        # Extract name and type fields
        stream_name = stream_info["name"][0] if isinstance(stream_info["name"], list) else stream_info["name"]
        stream_type = ""
        if "type" in stream_info:
            stream_type = stream_info["type"][0] if isinstance(stream_info["type"], list) else stream_info["type"]

        # Combine for checking (check both name and type)
        combined = f"{stream_name} {stream_type}".upper()

        if "ECG" in combined:
            return SignalType.ECG
        elif "PPG" in combined or "GSR" in combined:
            # Note: Some devices label PPG as GSR
            return SignalType.PPG
        elif "EDA" in combined or "ELECTRODERMAL" in combined:
            return SignalType.EDA
        else:
            logger.warning(f"Unknown signal type for stream: name='{stream_name}', type='{stream_type}'")
            return SignalType.UNKNOWN

    def _extract_events_from_streams(self, streams: list, lsl_t0_reference: float) -> list[EventData]:
        """Extract event markers from XDF streams.

        Looks for streams with type "Markers" or similar event streams.
        Event streams typically have nominal_srate=0 and time_series as list of markers.

        All event timestamps are zero-referenced using lsl_t0_reference (the first LSL
        timestamp of the physiological signal stream), so events and signals share the
        same time axis.

        Args:
            streams: List of XDF streams
            lsl_t0_reference: First LSL timestamp of the primary physiological signal stream.

        Returns:
            List of EventData objects
        """
        events = []

        for stream in streams:
            info = stream["info"]
            stream_type = ""
            if "type" in info:
                stream_type = info["type"][0] if isinstance(info["type"], list) else info["type"]

            # Check if this is an event/marker stream
            if stream_type.lower() not in ["markers", "marker", "events", "event"]:
                continue

            # Event streams have time_series as list of markers
            time_series = stream["time_series"]
            if not isinstance(time_series, list):
                continue

            time_stamps = stream["time_stamps"]
            stream_name = info["name"][0] if isinstance(info["name"], list) else info["name"]

            logger.info(f"Found event stream: {stream_name} with {len(time_series)} events")

            # Zero-reference using the physiological signal's first LSL timestamp.
            # Do NOT use the event stream's own first timestamp — that would misalign events.
            time_stamps = np.array(time_stamps, dtype=np.float64)
            time_stamps = time_stamps - lsl_t0_reference
            logger.debug(f"Event timestamps zero-referenced to signal LSL t0 (range: {time_stamps[0]:.3f}s to {time_stamps[-1]:.3f}s)")

            # Extract events
            for i, (marker, timestamp) in enumerate(zip(time_series, time_stamps)):
                # Marker is typically a list with one string element
                if isinstance(marker, list):
                    label = marker[0] if len(marker) > 0 else "unknown"
                else:
                    label = str(marker)

                event = EventData(
                    timestamp=float(timestamp),
                    label=label,
                    duration=None,  # XDF markers typically don't include duration
                    metadata={"stream": stream_name}
                )
                events.append(event)

            logger.info(f"Loaded {len(events)} events from stream '{stream_name}'")

        return events

    def _validate_sampling_rate(self, sampling_rate: float, stream_name: str) -> bool:
        """Validate sampling rate is reasonable.

        Args:
            sampling_rate: Sampling rate in Hz
            stream_name: Name of stream for error messages

        Returns:
            True if valid, False if suspicious (logs warning)
        """
        if sampling_rate <= 0:
            logger.error(f"Invalid sampling rate for '{stream_name}': {sampling_rate} Hz (must be positive)")
            return False

        # Reasonable range for physiological signals
        if sampling_rate < 16 or sampling_rate > 2000:
            logger.warning(
                f"Unusual sampling rate for '{stream_name}': {sampling_rate} Hz "
                f"(expected 16-2000 Hz for physiological signals)"
            )

        return True

    def _validate_timestamps(self, timestamps: np.ndarray, channel_name: str) -> bool:
        """Validate timestamps are strictly increasing.

        Args:
            timestamps: Timestamp array
            channel_name: Channel name for error messages

        Returns:
            True if valid, False if not monotonic
        """
        if len(timestamps) == 0:
            return True

        # Check for strictly increasing (allow some floating-point tolerance)
        time_diffs = np.diff(timestamps)
        if np.any(time_diffs <= 0):
            non_monotonic_indices = np.where(time_diffs <= 0)[0]
            logger.warning(
                f"Timestamps not strictly increasing for '{channel_name}': "
                f"{len(non_monotonic_indices)} non-monotonic jumps detected "
                f"(first at index {non_monotonic_indices[0]})"
            )
            return False

        return True

    def _validate_signal_values(self, samples: np.ndarray, channel_name: str) -> bool:
        """Validate signal values are reasonable.

        Args:
            samples: Signal samples
            channel_name: Channel name for error messages

        Returns:
            True if valid, False if problematic (logs warning)
        """
        if len(samples) == 0:
            logger.warning(f"Empty signal for '{channel_name}'")
            return False

        # Check for excessive NaN values
        nan_count = np.sum(np.isnan(samples))
        if nan_count > 0:
            nan_pct = 100.0 * nan_count / len(samples)
            if nan_pct > 50:
                logger.warning(
                    f"Signal '{channel_name}' has {nan_pct:.1f}% NaN values ({nan_count}/{len(samples)})"
                )
                return False
            elif nan_pct > 10:
                logger.warning(
                    f"Signal '{channel_name}' has {nan_pct:.1f}% NaN values ({nan_count}/{len(samples)}), "
                    f"consider checking data quality"
                )

        # Check for flat signal (constant value)
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) > 0:
            if np.std(valid_samples) == 0:
                logger.warning(
                    f"Signal '{channel_name}' is constant (value={valid_samples[0]:.3f}), "
                    f"may indicate sensor disconnection"
                )
                return False

            # Check for extreme values (>1000 standard deviations from mean)
            mean = np.mean(valid_samples)
            std = np.std(valid_samples)
            if std > 0:
                extreme_mask = np.abs(valid_samples - mean) > 1000 * std
                extreme_count = np.sum(extreme_mask)
                if extreme_count > 0:
                    logger.warning(
                        f"Signal '{channel_name}' has {extreme_count} extreme outliers "
                        f"(>1000σ from mean={mean:.2f}, σ={std:.2f})"
                    )

        return True

    def _assess_timestamp_corruption(self, timestamps: np.ndarray, stream_name: str, source: str) -> dict:
        """Assess severity of timestamp corruption.

        Args:
            timestamps: Timestamp array to assess
            stream_name: Stream name for logging
            source: "LSL" or "Device" for logging context

        Returns:
            Dictionary with:
            - is_clean: True if no corruption
            - is_fixable: True if corruption is minor and can be repaired
            - backward_jumps: Number of backward jumps
            - max_backward_jump: Magnitude of worst backward jump
        """
        if len(timestamps) < 2:
            return {"is_clean": True, "is_fixable": False, "backward_jumps": 0, "max_backward_jump": 0}

        diffs = np.diff(timestamps)
        backward_mask = diffs <= 0
        backward_count = np.sum(backward_mask)

        if backward_count == 0:
            return {"is_clean": True, "is_fixable": False, "backward_jumps": 0, "max_backward_jump": 0}

        # Assess if fixable
        # Fixable if: few jumps (<100) and not too frequent (< 5% of samples)
        backward_pct = 100.0 * backward_count / len(timestamps)
        max_backward = np.min(diffs[backward_mask])

        is_fixable = backward_count < 100 and backward_pct < 5.0

        logger.debug(
            f"{source} timestamps for '{stream_name}': "
            f"{backward_count} backward jumps ({backward_pct:.2f}%), "
            f"worst: {max_backward:.6f}s, fixable: {is_fixable}"
        )

        return {
            "is_clean": False,
            "is_fixable": is_fixable,
            "backward_jumps": backward_count,
            "max_backward_jump": max_backward,
        }

    def _repair_lsl_timestamps(self, timestamps: np.ndarray, stream_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Repair corrupted timestamps by removing out-of-order samples.

        Simple strategy: remove samples with backward/flat jumps, then verify
        the repair actually worked.

        Args:
            timestamps: Corrupted timestamp array
            stream_name: Stream name for logging

        Returns:
            Tuple of (repaired_timestamps, valid_mask), or None if repair failed
        """
        if len(timestamps) < 2:
            return timestamps, np.ones(len(timestamps), dtype=bool)

        diffs = np.diff(timestamps)
        # Keep samples where diff > 0
        valid_mask = np.concatenate([[True], diffs > 0])

        if np.sum(valid_mask) < 10:
            logger.error(f"Repair would remove too many samples ({np.sum(~valid_mask)}/{len(timestamps)}) for '{stream_name}'")
            return None

        # Extract valid timestamps
        repaired = timestamps[valid_mask]

        # VERIFY the repair actually fixed the problem
        repaired_diffs = np.diff(repaired)
        repaired_backward_count = np.sum(repaired_diffs <= 0)

        if repaired_backward_count > 0:
            logger.error(
                f"Repair failed for '{stream_name}': "
                f"still {repaired_backward_count} backward jumps after removing {len(timestamps) - len(repaired)} samples. "
                f"Corruption too extensive."
            )
            return None

        removed_count = len(timestamps) - len(repaired)
        logger.info(f"Successfully repaired timestamps: removed {removed_count} out-of-order samples from '{stream_name}'")

        return repaired, valid_mask


class CsvLoader:
    """Loader for CSV files with time and signal columns.

    Supports two CSV formats:
    1. Standard: First column = time, remaining = signals
    2. Shimmer: 3 header rows (names, calibration, units) + data

    Sampling rate is auto-calculated from timestamps.
    Signal type can be specified via init parameter or auto-detected from column names.
    """

    def __init__(self, signal_type: SignalType = SignalType.UNKNOWN, auto_detect_type: bool = True):
        """Initialize CSV loader.

        Args:
            signal_type: Signal type for all columns (default: UNKNOWN).
            auto_detect_type: Auto-detect signal type from column names (default: True).
        """
        self.signal_type = signal_type
        self.auto_detect_type = auto_detect_type

    def can_load(self, path: Path) -> bool:
        """Check if file is CSV format."""
        return path.suffix.lower() == ".csv"

    def load(self, path: Path) -> RecordingSession:
        """Load CSV file.

        Args:
            path: Path to CSV file

        Returns:
            RecordingSession with loaded signals

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        # Layered validation: type → exists → format
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception:
                raise TypeError(f"path must be a Path object or string, got {type(path).__name__}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Invalid file extension, expected .csv, got {path.suffix}")

        logger.info(f"Loading CSV file: {path}")

        # Detect CSV format (standard vs Shimmer)
        is_shimmer = self._is_shimmer_format(path)

        if is_shimmer:
            logger.info("Detected Shimmer CSV format")
            return self._load_shimmer_csv(path)
        else:
            logger.info("Detected standard CSV format")
            return self._load_standard_csv(path)

    def _is_shimmer_format(self, path: Path) -> bool:
        """Check if CSV is in Shimmer format.

        Shimmer CSVs have 3 header rows:
        - Row 1: Column names
        - Row 2: Calibration status (contains "CAL")
        - Row 3: Units

        Args:
            path: Path to CSV file

        Returns:
            True if Shimmer format detected
        """
        try:
            # Read first 3 rows without parsing
            with open(path, 'r') as f:
                lines = [f.readline().strip() for _ in range(3)]

            # Check if row 2 contains "CAL"
            if len(lines) >= 2 and "CAL" in lines[1]:
                return True
        except Exception:
            pass

        return False

    def _load_shimmer_csv(self, path: Path) -> RecordingSession:
        """Load Shimmer-format CSV.

        Format:
        - Row 1: Column headers
        - Row 2: Calibration status ("CAL" = calibrated physiological, "RAW" = device metadata)
        - Row 3: Units
        - Row 4+: Data

        Only CAL columns are loaded as signals. RAW columns (lead-off detection,
        status bytes such as EXG*_STA) are skipped — they are device metadata,
        not physiological measurements.

        Args:
            path: Path to CSV file

        Returns:
            RecordingSession with loaded signals
        """
        # Read the three header rows so we can use calibration status for filtering
        with open(path, "r") as f:
            column_names = f.readline().strip().split(",")
            cal_status = f.readline().strip().split(",")  # "CAL" or "RAW" per column

        # Load CSV skipping first 3 rows (header metadata)
        df = pd.read_csv(path, skiprows=3, header=None)

        if len(df) < 2:
            raise ValueError("CSV must have at least 2 data rows")

        # Assign column names to data
        df.columns = column_names[:len(df.columns)]

        # Build set of columns to keep: timestamp column + all CAL columns.
        # Pad cal_status with "RAW" if it is shorter than column_names (safety).
        cal_status_padded = list(cal_status) + ["RAW"] * (len(column_names) - len(cal_status))
        cal_columns = {
            column_names[i]
            for i, status in enumerate(cal_status_padded)
            if status.strip().upper() == "CAL"
        }
        raw_skipped = [
            column_names[i]
            for i, status in enumerate(cal_status_padded[1:], start=1)
            if status.strip().upper() != "CAL"
        ]
        if raw_skipped:
            logger.info(
                f"Shimmer CSV: skipping {len(raw_skipped)} RAW column(s) "
                f"(lead-off / status): {raw_skipped}"
            )

        # Drop rows where the EXG device status columns indicate abnormal state.
        #
        # Shimmer EXG recordings interleave multiple packet types in the same CSV.
        # Rows from non-ECG packets (e.g. lead-off detection bursts, Bluetooth
        # handshake packets) have EXG status bytes that differ from the dominant
        # recording state. These rows contain garbage ADC values and can have
        # timestamps far outside the main recording window, creating spurious gaps.
        #
        # Strategy: keep only rows whose EXG status columns match the mode (most
        # common value). This preserves normal recording rows regardless of the
        # exact status code used by a specific Shimmer firmware version.
        exg_status_cols = [
            column_names[i]
            for i, status in enumerate(cal_status_padded)
            if status.strip().upper() == "RAW" and "sta" in column_names[i].lower()
        ]
        present_status = [c for c in exg_status_cols if c in df.columns]
        if present_status:
            # Build a composite status key and keep only rows matching the mode
            status_key = df[present_status].astype(str).agg(",".join, axis=1)
            mode_key = status_key.mode().iloc[0]
            status_mask = status_key != mode_key
            n_dropped = int(status_mask.sum())
            if n_dropped:
                df = df[~status_mask].reset_index(drop=True)
                logger.info(
                    f"Shimmer CSV: dropped {n_dropped} row(s) with non-standard EXG "
                    f"status (lead-off / handshake packets) — {len(df)} rows remain"
                )

        # First column is time (in milliseconds for Shimmer)
        time_col = df.columns[0]
        timestamps_ms = df[time_col].values.astype(np.float64)

        # Convert milliseconds to seconds
        timestamps = timestamps_ms / 1000.0

        # Align to start at time 0
        timestamps = timestamps - timestamps[0]

        # Validate timestamps are monotonically increasing
        time_diffs = np.diff(timestamps)
        if not np.all(time_diffs > 0):
            non_monotonic_indices = np.where(time_diffs <= 0)[0]
            raise ValueError(
                f"Timestamps must be strictly increasing. Found {len(non_monotonic_indices)} "
                f"non-monotonic jumps (first at row {non_monotonic_indices[0] + 4})"
            )

        # Calculate sampling rate
        mean_interval = np.mean(time_diffs)
        sampling_rate = 1.0 / mean_interval

        # Validate sampling rate
        if sampling_rate <= 0:
            raise ValueError(f"Invalid sampling rate: {sampling_rate} Hz (must be positive)")

        if sampling_rate < 16 or sampling_rate > 2000:
            logger.warning(
                f"Unusual sampling rate: {sampling_rate:.2f} Hz "
                f"(expected 16-2000 Hz for physiological signals)"
            )

        logger.info(f"Detected sampling rate: {sampling_rate:.2f} Hz (timestamps aligned to t=0)")

        # Create signals from CAL columns only (skip timestamp column and all RAW columns)
        signals = []
        for col_name in df.columns[1:]:
            # Skip RAW columns (lead-off detection, status bytes, etc.)
            if col_name not in cal_columns:
                continue

            samples = df[col_name].values.astype(np.float64)

            # Skip empty columns
            if np.all(np.isnan(samples)) or np.all(samples == 0):
                logger.debug(f"Skipping empty column: {col_name}")
                continue

            # Detect signal type from column name
            if self.auto_detect_type:
                detected_type = self._detect_signal_type_from_name(col_name)
            else:
                detected_type = self.signal_type

            signal = SignalData(
                samples=samples,
                sampling_rate=sampling_rate,
                timestamps=timestamps,
                channel_name=col_name,
                signal_type=detected_type,
            )
            signals.append(signal)

        if not signals:
            raise ValueError("No valid signal columns found in CSV")

        logger.info(f"Loaded {len(signals)} signals from Shimmer CSV")

        # Load events if available
        events = []
        event_path = self._find_event_file(path)
        if event_path:
            try:
                if event_path.suffix == ".json":
                    events = self._load_events_from_json(event_path)

                    # If JSON loaded 0 events, try CSV fallback
                    if len(events) == 0:
                        csv_fallback = event_path.with_suffix(".csv")
                        if csv_fallback.exists():
                            logger.info(f"JSON loaded 0 events, attempting CSV fallback: {csv_fallback.name}")
                            try:
                                events = self._load_events_from_csv(csv_fallback)
                            except Exception as csv_e:
                                logger.warning(f"CSV fallback also failed: {csv_e}")
                else:
                    events = self._load_events_from_csv(event_path)
            except Exception as e:
                logger.warning(f"Failed to load events from {event_path}: {e}")

                # Try fallback to CSV if JSON failed with exception
                if event_path.suffix == ".json":
                    csv_fallback = event_path.with_suffix(".csv")
                    if csv_fallback.exists():
                        try:
                            logger.info(f"Attempting CSV fallback after error: {csv_fallback.name}")
                            events = self._load_events_from_csv(csv_fallback)
                        except Exception as csv_e:
                            logger.warning(f"CSV fallback also failed: {csv_e}")

        return RecordingSession(source_path=path, signals=signals, events=events)

    def _load_standard_csv(self, path: Path) -> RecordingSession:
        """Load standard CSV format.

        Format:
        - Row 1: Column headers
        - Row 2+: Data (first column = time, rest = signals)

        Args:
            path: Path to CSV file

        Returns:
            RecordingSession with loaded signals
        """
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

        if len(df) < 2:
            raise ValueError("CSV must have at least 2 rows")

        # Expect first column to be time
        if df.columns[0].lower() not in ["time", "timestamp", "t", "time stamp"]:
            logger.warning(f"First column '{df.columns[0]}' doesn't look like time, assuming it is anyway")

        timestamps = df.iloc[:, 0].values.astype(np.float64)

        # Check if timestamps are in milliseconds (large values)
        if np.mean(timestamps) > 1000:
            logger.info("Timestamps appear to be in milliseconds, converting to seconds")
            timestamps = timestamps / 1000.0

        # Align to start at time 0
        timestamps = timestamps - timestamps[0]

        # Validate timestamps
        time_diffs = np.diff(timestamps)
        if not np.all(time_diffs > 0):
            non_monotonic_indices = np.where(time_diffs <= 0)[0]
            raise ValueError(
                f"Timestamps must be strictly increasing. Found {len(non_monotonic_indices)} "
                f"non-monotonic jumps (first at row {non_monotonic_indices[0] + 2})"
            )

        # Calculate sampling rate from timestamps
        mean_interval = np.mean(time_diffs)
        sampling_rate = 1.0 / mean_interval

        # Validate sampling rate
        if sampling_rate <= 0:
            raise ValueError(f"Invalid sampling rate: {sampling_rate} Hz (must be positive)")

        if sampling_rate < 16 or sampling_rate > 2000:
            logger.warning(
                f"Unusual sampling rate: {sampling_rate:.2f} Hz "
                f"(expected 16-2000 Hz for physiological signals)"
            )

        logger.info(f"Detected sampling rate: {sampling_rate:.2f} Hz (timestamps aligned to t=0)")

        # Create signals from remaining columns
        signals = []
        for col_name in df.columns[1:]:
            samples = df[col_name].values.astype(np.float64)

            # Skip empty columns
            if np.all(np.isnan(samples)) or np.all(samples == 0):
                logger.debug(f"Skipping empty column: {col_name}")
                continue

            # Detect signal type from column name
            if self.auto_detect_type:
                detected_type = self._detect_signal_type_from_name(col_name)
            else:
                detected_type = self.signal_type

            signal = SignalData(
                samples=samples,
                sampling_rate=sampling_rate,
                timestamps=timestamps,
                channel_name=col_name,
                signal_type=detected_type,
            )
            signals.append(signal)

        if not signals:
            raise ValueError("No valid signal columns found in CSV")

        logger.info(f"Loaded {len(signals)} signals from CSV")

        return RecordingSession(source_path=path, signals=signals)

    def _detect_signal_type_from_name(self, column_name: str) -> SignalType:
        """Auto-detect signal type from column name.

        Args:
            column_name: Name of the signal column

        Returns:
            Detected SignalType
        """
        detected = detect_signal_type_from_name(column_name)
        if detected == SignalType.UNKNOWN:
            return self.signal_type
        return detected

    def _find_event_file(self, signal_path: Path) -> Path | None:
        """Find companion event file for Shimmer CSV.

        Event files are in same directory with same name but starting with
        "events_" instead of "shimmer_". Can be .json or .csv format.

        Args:
            signal_path: Path to Shimmer signal CSV file

        Returns:
            Path to event file if found, None otherwise
        """
        # Check if this is a Shimmer file
        if not signal_path.stem.startswith("shimmer_"):
            return None

        # Replace "shimmer_" with "events_" in filename
        event_stem = signal_path.stem.replace("shimmer_", "events_", 1)
        event_dir = signal_path.parent

        # Check for JSON first (preferred format)
        json_path = event_dir / f"{event_stem}.json"
        if json_path.exists():
            return json_path

        # Fall back to CSV
        csv_path = event_dir / f"{event_stem}.csv"
        if csv_path.exists():
            return csv_path

        return None

    def _load_events_from_json(self, event_path: Path) -> list[EventData]:
        """Load events from JSON format.

        Events use time_since_connected_ms which aligns with signals starting at t=0.

        Args:
            event_path: Path to JSON event file

        Returns:
            List of EventData objects
        """
        with open(event_path, 'r') as f:
            data = json.load(f)

        events = []
        for event in data.get("events", []):
            # Use time_since_connected_ms if available, else calculate from timestamp_unix_ms
            if "time_since_connected_ms" in event:
                timestamp_ms = event["time_since_connected_ms"]
            elif "timestamp_unix_ms" in event and "synchronization" in data:
                # Calculate relative time from unix timestamp - try nested dict access safely
                try:
                    anchor_unix_ms = data["synchronization"]["synchronizer_config"]["anchor_unix_ms"]
                    timestamp_ms = event["timestamp_unix_ms"] - anchor_unix_ms
                except (KeyError, TypeError):
                    logger.warning(f"Event has timestamp_unix_ms but no valid synchronization anchor")
                    continue
            else:
                logger.warning(f"Event missing timestamp information: {event}")
                continue

            # Convert to seconds (already relative to start)
            timestamp_s = timestamp_ms / 1000.0

            event_data = EventData(
                timestamp=timestamp_s,
                label=event.get("event_type", "unknown"),
                duration=event.get("duration"),
                metadata=event.get("metadata", {})
            )
            events.append(event_data)

        logger.info(f"Loaded {len(events)} events from JSON: {event_path.name}")
        return events

    def _load_events_from_csv(self, event_path: Path) -> list[EventData]:
        """Load events from CSV format.

        Events use time_since_connected_ms which aligns with signals starting at t=0.

        Args:
            event_path: Path to CSV event file

        Returns:
            List of EventData objects
        """
        # Read CSV, skip comment lines
        df = pd.read_csv(event_path, comment='#')

        events = []
        for _, row in df.iterrows():
            # Get timestamp in milliseconds
            timestamp_ms = row.get("time_since_connected_ms")
            if timestamp_ms is None or pd.isna(timestamp_ms):
                continue

            # Convert to seconds
            timestamp_s = timestamp_ms / 1000.0

            # Parse metadata if present
            metadata = {}
            if "metadata" in row and not pd.isna(row["metadata"]):
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse event metadata: {row['metadata']}")

            event_data = EventData(
                timestamp=timestamp_s,
                label=row.get("event_type", "unknown"),
                duration=None,  # CSV format doesn't include duration
                metadata=metadata
            )
            events.append(event_data)

        logger.info(f"Loaded {len(events)} events from CSV: {event_path.name}")
        return events


class MatLoader:
    """Loader for MATLAB .mat files containing corrected XDF streams.

    This loader is specifically for .mat files that were created by saving
    corrected XDF streams from MATLAB after manual fixing in the debugger:
        1. Load XDF in MATLAB with xdf()
        2. Wait for error, then fix corrupted array in debugger
        3. Save with save('xdf_data.mat', 'streams', 'header')
        4. Load the .mat file here in CardioSignalLab

    The .mat file must contain 'streams' variable (and optionally 'header').
    """

    def __init__(self, apply_lsl_alignment: bool = False):
        """Initialize MAT loader.

        Args:
            apply_lsl_alignment: Use LSL timestamps for alignment (default: False, uses device timestamps)
        """
        self.apply_lsl_alignment = apply_lsl_alignment
        self.skipped_streams = []  # Track streams that failed to load

    def can_load(self, path: Path) -> bool:
        """Check if file is MAT format."""
        return path.suffix.lower() == ".mat"

    def load(self, path: Path) -> RecordingSession:
        """Load MAT file containing XDF streams.

        Args:
            path: Path to MAT file

        Returns:
            RecordingSession with loaded signals

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If MAT format is invalid or doesn't contain streams
        """
        # Reset skipped streams list for this load
        self.skipped_streams = []

        # Layered validation: type → exists → format
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception:
                raise TypeError(f"path must be a Path object or string, got {type(path).__name__}")

        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".mat":
            raise ValueError(f"Invalid file extension, expected .mat, got {path.suffix}")

        logger.info(f"Loading MAT file: {path}")

        # Load MAT file
        try:
            import scipy.io
            mat_data = scipy.io.loadmat(str(path), squeeze_me=True)
        except ImportError:
            raise ValueError("scipy not installed. Install with: pip install scipy")
        except Exception as e:
            raise ValueError(f"Failed to load MAT file: {e}")

        # Extract streams and header
        if 'streams' not in mat_data:
            raise ValueError("MAT file must contain 'streams' variable")

        streams_mat = mat_data['streams']
        header = mat_data.get('header', None)

        # Convert MATLAB structure to pyxdf format
        streams = self._matlab_to_xdf_structure(streams_mat)

        if not streams:
            raise ValueError(f"No streams found in MAT file: {path}")

        # Determine LSL t0 reference from the first physiological signal stream
        lsl_t0_reference: float | None = None
        for stream in streams:
            timestamps = stream.get('time_stamps')
            if timestamps is not None:
                # Handle both arrays and scalars (scipy squeeze_me=True may return scalars)
                try:
                    ts_array = np.asarray(timestamps)
                    if ts_array.size > 0:
                        lsl_t0_reference = float(ts_array.flat[0])
                        logger.debug(f"LSL t0 reference: {lsl_t0_reference:.6f} s (absolute LSL clock)")
                        break
                except (TypeError, ValueError):
                    continue

        if lsl_t0_reference is None:
            raise ValueError(f"No valid timestamp streams found in MAT file: {path}")

        # Extract signals from streams (skip marker/event streams)
        signals = []
        for i, stream in enumerate(streams):
            # Skip marker/event streams (time_series is list or array of strings/objects)
            ts = stream.get("time_series")
            is_marker_stream = False

            if isinstance(ts, list):
                is_marker_stream = True
            elif isinstance(ts, np.ndarray):
                # Check if it's a string or object array (markers)
                if ts.dtype.kind in ['U', 'S']:  # Unicode or byte string
                    is_marker_stream = True
                elif ts.dtype.kind == 'O' and len(ts) > 0:  # Object array - check first element
                    try:
                        if isinstance(ts.flat[0], (str, bytes)):
                            is_marker_stream = True
                    except (IndexError, TypeError):
                        pass

            if is_marker_stream:
                logger.debug(f"Stream {i}: Skipping marker stream (string data)")
                continue

            signal_list = self._extract_signals_from_stream(stream, lsl_t0_reference)
            signals.extend(signal_list)

        if not signals:
            raise ValueError(f"No physiological signals found in MAT file: {path}")

        logger.info(f"Loaded {len(signals)} signals from MAT file")

        # Extract events from marker streams if present
        events = self._extract_events_from_streams(streams, lsl_t0_reference)

        return RecordingSession(
            source_path=path,
            signals=signals,
            events=events,
            lsl_t0_reference=lsl_t0_reference,
        )

    def _matlab_to_xdf_structure(self, streams_mat) -> list:
        """Convert MATLAB streams structure to pyxdf format.

        MATLAB saves streams as: (info_struct, time_series, time_stamps) or similar tuples.
        scipy.io.loadmat may return these as numpy record arrays or tuples.

        Args:
            streams_mat: MATLAB streams structure (from scipy.io.loadmat)

        Returns:
            List of stream dicts in pyxdf format
        """
        # Ensure streams_mat is iterable
        try:
            if hasattr(streams_mat, 'shape') and len(streams_mat.shape) > 0:
                streams_list = list(streams_mat)
            elif isinstance(streams_mat, (list, tuple)):
                streams_list = list(streams_mat)
            else:
                streams_list = [streams_mat]
        except Exception:
            streams_list = [streams_mat]

        streams = []
        for i, stream_mat in enumerate(streams_list):
            try:
                # Handle tuple format: (info_struct, optional_chunk_info, time_series, time_stamps)
                # or dict format: {'info': ..., 'time_series': ..., 'time_stamps': ...}

                if isinstance(stream_mat, tuple):
                    # MATLAB saved as tuple format
                    logger.debug(f"Stream {i}: Processing tuple format with {len(stream_mat)} elements")

                    if len(stream_mat) >= 3:
                        # Last two elements are typically time_series and time_stamps
                        info_data = stream_mat[0]
                        # Find time_series and time_stamps (usually last two elements)
                        time_series_data = stream_mat[-2] if len(stream_mat) >= 2 else None
                        time_stamps_data = stream_mat[-1] if len(stream_mat) >= 1 else None
                    else:
                        info_data = stream_mat[0] if len(stream_mat) > 0 else None
                        time_series_data = stream_mat[1] if len(stream_mat) > 1 else None
                        time_stamps_data = stream_mat[2] if len(stream_mat) > 2 else None

                elif isinstance(stream_mat, np.ndarray) and stream_mat.dtype.names:
                    # numpy record array with named fields
                    logger.debug(f"Stream {i}: Processing numpy record array with fields {stream_mat.dtype.names}")

                    # Extract fields from the record array
                    # Handle the case where records might be stored as arrays of records
                    if stream_mat.ndim == 0:
                        # Scalar record (0-d array)
                        stream_mat = np.atleast_1d(stream_mat)[0]

                    # Extract each field
                    try:
                        info_data = stream_mat['info'] if 'info' in stream_mat.dtype.names else stream_mat
                        time_series_data = stream_mat['time_series'] if 'time_series' in stream_mat.dtype.names else None
                        time_stamps_data = stream_mat['time_stamps'] if 'time_stamps' in stream_mat.dtype.names else None
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not extract fields from record array: {e}")
                        info_data = stream_mat
                        time_series_data = None
                        time_stamps_data = None

                elif isinstance(stream_mat, dict):
                    # Dict format
                    info_data = stream_mat.get('info')
                    time_series_data = stream_mat.get('time_series')
                    time_stamps_data = stream_mat.get('time_stamps')

                else:
                    logger.warning(f"Stream {i}: Unknown format {type(stream_mat)}")
                    continue

                stream = {
                    'info': self._convert_info_struct(info_data) if info_data is not None else {},
                    'time_series': np.asarray(time_series_data) if time_series_data is not None else np.array([]),
                    'time_stamps': np.asarray(time_stamps_data) if time_stamps_data is not None else np.array([])
                }
                streams.append(stream)
            except Exception as e:
                logger.warning(f"Error processing stream {i}: {e}, skipping")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        return streams

    def _convert_info_struct(self, info_mat) -> dict:
        """Convert MATLAB info structure to Python dict.

        Args:
            info_mat: MATLAB info structure (numpy record array or dict)

        Returns:
            Python dict
        """
        info = {}

        if isinstance(info_mat, dict):
            # Dict-like structure
            for key, value in info_mat.items():
                # Convert MATLAB cell arrays and nested structures to Python
                if hasattr(value, 'size'):  # numpy array
                    if value.size == 1:
                        info[key] = value.item()
                    else:
                        info[key] = value.tolist()
                elif isinstance(value, dict):
                    info[key] = self._convert_info_struct(value)
                else:
                    info[key] = value

        elif isinstance(info_mat, np.ndarray) and info_mat.dtype.names:
            # Structured array (record array)
            # Handle both regular and 0-d arrays
            if info_mat.ndim == 0:
                info_mat = np.atleast_1d(info_mat)[0]

            for field_name in info_mat.dtype.names:
                try:
                    value = info_mat[field_name]

                    # Recursively convert nested structures
                    if isinstance(value, np.ndarray):
                        if value.dtype.names:  # Nested structured array
                            info[field_name] = self._convert_info_struct(value)
                        elif value.size == 1:
                            info[field_name] = value.item()
                        else:
                            info[field_name] = value.tolist()
                    elif isinstance(value, dict):
                        info[field_name] = self._convert_info_struct(value)
                    else:
                        info[field_name] = value
                except (IndexError, TypeError):
                    continue

        else:
            # Try to convert as-is
            try:
                if hasattr(info_mat, 'size') and info_mat.size == 1:
                    info = info_mat.item()
                else:
                    info = info_mat
            except Exception:
                info = info_mat

        return info

    def _extract_signals_from_stream(self, stream: dict, lsl_t0_reference: float) -> list[SignalData]:
        """Extract SignalData objects from a stream.

        Handles both single-channel and multi-channel streams.
        Uses the same 4-tier fallback strategy as XdfLoader for timestamp validation.

        Args:
            stream: Stream dict
            lsl_t0_reference: First LSL timestamp reference

        Returns:
            List of SignalData objects
        """
        # Skip marker streams
        if isinstance(stream.get("time_series"), list):
            return []

        info = stream.get("info", {})
        stream_name = info.get("name", "Unknown")
        if isinstance(stream_name, list):
            stream_name = stream_name[0] if stream_name else "Unknown"

        # Extract signal data
        samples_raw = stream.get("time_series", np.array([]))
        timestamps = stream.get("time_stamps", np.array([]))

        if len(samples_raw) == 0 or len(timestamps) == 0:
            logger.warning(f"Skipping stream '{stream_name}': empty data")
            return []

        # Handle multi-channel data (2D arrays)
        # If samples_raw is 2D, each row is a separate channel
        if isinstance(samples_raw, np.ndarray) and samples_raw.ndim == 2:
            # Multi-channel: (channels, samples)
            channels = samples_raw.shape[0]
            logger.info(f"Stream '{stream_name}' has {channels} channels")

            # Get channel names from info if available
            # For Shimmer GSR devices: typically [timestamps_placeholder, GSR, Internal_ADC_A13]
            if stream_name.lower() == "gsr" and channels == 3:
                channel_names = ["timestamps", "GSR", "Internal ADC A13"]
            else:
                channel_names = self._get_channel_names(info, channels, stream_name)

            signals = []
            for ch_idx in range(channels):
                # Skip 'timestamps' placeholder channel (first channel in Shimmer)
                if channel_names[ch_idx].lower() == 'timestamps':
                    logger.debug(f"Skipping timestamps placeholder channel")
                    continue

                samples = samples_raw[ch_idx, :]

                # Validate and repair timestamps (4-tier strategy)
                ch_samples, ch_timestamps = self._validate_and_repair_timestamps(
                    samples, timestamps, f"{stream_name}_ch{ch_idx + 1}"
                )

                if ch_samples is None:
                    self.skipped_streams.append(f"{stream_name}_ch{ch_idx + 1}")
                    continue

                # Zero-reference timestamps
                ch_timestamps = ch_timestamps - lsl_t0_reference

                # Calculate sampling rate
                if len(ch_timestamps) > 1:
                    time_diffs = np.diff(ch_timestamps)
                    sampling_rate = 1.0 / np.mean(time_diffs)
                else:
                    sampling_rate = 1.0

                # Validate sampling rate
                self._validate_sampling_rate(sampling_rate, channel_names[ch_idx])

                # Detect signal type from channel name
                detected_type = detect_signal_type_from_name(channel_names[ch_idx])

                signal = SignalData(
                    samples=ch_samples,
                    sampling_rate=sampling_rate,
                    timestamps=ch_timestamps,
                    channel_name=channel_names[ch_idx],
                    signal_type=detected_type,
                )
                signals.append(signal)

            return signals

        else:
            # Single channel
            samples = np.asarray(samples_raw)
            if samples.ndim != 1:
                samples = samples.flatten()

            # Validate and repair timestamps (4-tier strategy)
            samples, timestamps = self._validate_and_repair_timestamps(
                samples, timestamps, stream_name
            )

            if samples is None:
                self.skipped_streams.append(stream_name)
                return []

            # Zero-reference timestamps
            timestamps = timestamps - lsl_t0_reference

            # Calculate sampling rate
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                sampling_rate = 1.0 / np.mean(time_diffs)
            else:
                sampling_rate = 1.0

            # Validate sampling rate
            self._validate_sampling_rate(sampling_rate, stream_name)

            # Detect signal type
            detected_type = self._detect_signal_type_from_stream_info(info)

            signal = SignalData(
                samples=samples,
                sampling_rate=sampling_rate,
                timestamps=timestamps,
                channel_name=stream_name,
                signal_type=detected_type,
            )

            return [signal]

    def _validate_and_repair_timestamps(
        self, samples: np.ndarray, timestamps: np.ndarray, stream_name: str
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """4-tier fallback strategy for timestamp validation.

        1. Use timestamps if clean
        2. If corrupted but fixable, repair
        3. If not fixable, skip the stream
        4. (No device timestamps in MAT files)

        Args:
            samples: Signal samples
            timestamps: Timestamp array
            stream_name: Stream name for logging

        Returns:
            (samples, timestamps) if valid, (None, None) if skipped
        """
        # Ensure arrays are numeric
        samples = np.asarray(samples, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)

        # Validate timestamps
        assessment = self._assess_timestamp_corruption(timestamps, stream_name, "MAT")

        if assessment["is_clean"]:
            logger.info(f"Stream '{stream_name}': timestamps are clean")
            return samples, timestamps

        # Try to repair if fixable
        if assessment["is_fixable"]:
            logger.info(
                f"Stream '{stream_name}': corrupted but fixable "
                f"({assessment['backward_jumps']} backward jumps)"
            )
            repaired = self._repair_lsl_timestamps(timestamps, stream_name)
            if repaired is not None:
                repaired_timestamps, valid_mask = repaired
                # Also trim samples to match
                repaired_samples = samples[valid_mask]
                return repaired_samples, repaired_timestamps

        # Can't repair
        logger.error(
            f"Stream '{stream_name}': corrupted and cannot be repaired "
            f"({assessment['backward_jumps']} backward jumps, "
            f"{100.0 * assessment['backward_jumps'] / len(timestamps):.1f}%)"
        )
        return None, None

    def _get_channel_names(self, stream_info: dict, num_channels: int, stream_name: str) -> list[str]:
        """Get channel names from stream info or generate them.

        Args:
            stream_info: Stream info dict (numpy record array)
            num_channels: Number of channels
            stream_name: Default stream name

        Returns:
            List of channel names
        """
        channel_names = []

        # Try to extract channel names from Shimmer-style desc/channels metadata
        try:
            if isinstance(stream_info, np.ndarray) and stream_info.dtype.names:
                stream_info_scalar = np.atleast_1d(stream_info)[0]

                if 'desc' in stream_info_scalar.dtype.names:
                    desc = stream_info_scalar['desc']

                    if isinstance(desc, np.ndarray) and desc.dtype.names:
                        desc_scalar = np.atleast_1d(desc)[0]

                        if 'channels' in desc_scalar.dtype.names:
                            channels_wrapper = desc_scalar['channels']

                            # Unwrap the nested array structure
                            if isinstance(channels_wrapper, np.ndarray):
                                if channels_wrapper.ndim == 0:
                                    channels_wrapper = np.atleast_1d(channels_wrapper)

                                # Try to iterate through channels
                                if channels_wrapper.size > 0:
                                    channels_data = channels_wrapper.flat[0]

                                    if isinstance(channels_data, np.ndarray):
                                        # Each channel is a record with 'label' and 'unit'
                                        for ch_item in channels_data:
                                            try:
                                                if isinstance(ch_item, np.ndarray):
                                                    ch_scalar = np.atleast_1d(ch_item)[0]
                                                    if hasattr(ch_scalar, 'dtype') and 'label' in ch_scalar.dtype.names:
                                                        label = ch_scalar['label']
                                                        if isinstance(label, np.ndarray):
                                                            label = label.flat[0]
                                                        label_str = str(label)
                                                        # Skip 'timestamps' placeholder
                                                        if label_str.lower() != 'timestamps':
                                                            channel_names.append(label_str)
                                            except (IndexError, TypeError, AttributeError):
                                                continue
        except Exception as e:
            logger.debug(f"Could not extract channel names from desc: {e}")

        # Fill in any missing channel names
        while len(channel_names) < num_channels:
            ch_idx = len(channel_names) + 1
            channel_names.append(f"{stream_name}_ch{ch_idx}")

        return channel_names[:num_channels]

    def _detect_signal_type_from_stream_info(self, stream_info: dict) -> SignalType:
        """Detect signal type from stream metadata.

        Args:
            stream_info: Stream info dict

        Returns:
            Detected SignalType
        """
        stream_name = stream_info.get("name", "")
        if isinstance(stream_name, list):
            stream_name = stream_name[0] if stream_name else ""

        stream_type = ""
        if "type" in stream_info:
            st = stream_info["type"]
            stream_type = st[0] if isinstance(st, list) else st

        combined = f"{stream_name} {stream_type}".upper()

        if "ECG" in combined:
            return SignalType.ECG
        elif "PPG" in combined or "GSR" in combined:
            return SignalType.PPG
        elif "EDA" in combined or "ELECTRODERMAL" in combined:
            return SignalType.EDA
        else:
            return SignalType.UNKNOWN

    def _extract_events_from_streams(self, streams: list, lsl_t0_reference: float) -> list[EventData]:
        """Extract events from marker streams.

        Args:
            streams: List of streams
            lsl_t0_reference: LSL t0 reference

        Returns:
            List of EventData objects
        """
        events = []

        for stream in streams:
            info = stream.get("info", {})
            stream_type = ""
            if "type" in info:
                st = info["type"]
                stream_type = st[0] if isinstance(st, list) else st

            # Skip non-marker streams
            if stream_type.lower() not in ["markers", "marker", "events", "event"]:
                continue

            time_series = stream.get("time_series")
            if not isinstance(time_series, list):
                continue

            time_stamps = stream.get("time_stamps", np.array([]))
            stream_name = info.get("name", "Unknown")
            if isinstance(stream_name, list):
                stream_name = stream_name[0] if stream_name else "Unknown"

            logger.info(f"Found event stream: {stream_name} with {len(time_series)} events")

            # Zero-reference using LSL t0
            time_stamps = np.array(time_stamps, dtype=np.float64)
            time_stamps = time_stamps - lsl_t0_reference

            # Extract events
            for marker, timestamp in zip(time_series, time_stamps):
                if isinstance(marker, list):
                    label = marker[0] if marker else "unknown"
                else:
                    label = str(marker)

                event = EventData(
                    timestamp=float(timestamp),
                    label=label,
                    duration=None,
                    metadata={"stream": stream_name}
                )
                events.append(event)

        return events

    def _assess_timestamp_corruption(self, timestamps: np.ndarray, stream_name: str, source: str) -> dict:
        """Assess severity of timestamp corruption.

        Args:
            timestamps: Timestamp array to assess
            stream_name: Stream name for logging
            source: "MAT" for logging context

        Returns:
            Dictionary with corruption assessment
        """
        if len(timestamps) < 2:
            return {"is_clean": True, "is_fixable": False, "backward_jumps": 0, "max_backward_jump": 0}

        diffs = np.diff(timestamps)
        backward_mask = diffs <= 0
        backward_count = np.sum(backward_mask)

        if backward_count == 0:
            return {"is_clean": True, "is_fixable": False, "backward_jumps": 0, "max_backward_jump": 0}

        # Assess if fixable
        backward_pct = 100.0 * backward_count / len(timestamps)
        max_backward = np.min(diffs[backward_mask])

        is_fixable = backward_count < 100 and backward_pct < 5.0

        logger.debug(
            f"{source} timestamps for '{stream_name}': "
            f"{backward_count} backward jumps ({backward_pct:.2f}%), "
            f"worst: {max_backward:.6f}s, fixable: {is_fixable}"
        )

        return {
            "is_clean": False,
            "is_fixable": is_fixable,
            "backward_jumps": backward_count,
            "max_backward_jump": max_backward,
        }

    def _repair_lsl_timestamps(self, timestamps: np.ndarray, stream_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Repair corrupted timestamps by removing out-of-order samples.

        Args:
            timestamps: Corrupted timestamp array
            stream_name: Stream name for logging

        Returns:
            (repaired_timestamps, valid_mask), or None if repair failed
        """
        if len(timestamps) < 2:
            return timestamps, np.ones(len(timestamps), dtype=bool)

        diffs = np.diff(timestamps)
        valid_mask = np.concatenate([[True], diffs > 0])

        if np.sum(valid_mask) < 10:
            logger.error(f"Repair would remove too many samples ({np.sum(~valid_mask)}/{len(timestamps)}) for '{stream_name}'")
            return None

        repaired = timestamps[valid_mask]

        # Verify repair worked
        repaired_diffs = np.diff(repaired)
        repaired_backward_count = np.sum(repaired_diffs <= 0)

        if repaired_backward_count > 0:
            logger.error(
                f"Repair failed for '{stream_name}': "
                f"still {repaired_backward_count} backward jumps after removing {len(timestamps) - len(repaired)} samples"
            )
            return None

        removed_count = len(timestamps) - len(repaired)
        logger.info(f"Successfully repaired timestamps: removed {removed_count} out-of-order samples from '{stream_name}'")

        return repaired, valid_mask

    def _validate_sampling_rate(self, sampling_rate: float, stream_name: str) -> bool:
        """Validate sampling rate is reasonable.

        Args:
            sampling_rate: Sampling rate in Hz
            stream_name: Name of stream for error messages

        Returns:
            True if valid
        """
        if sampling_rate <= 0:
            logger.error(f"Invalid sampling rate for '{stream_name}': {sampling_rate} Hz (must be positive)")
            return False

        if sampling_rate < 16 or sampling_rate > 2000:
            logger.warning(
                f"Unusual sampling rate for '{stream_name}': {sampling_rate} Hz "
                f"(expected 16-2000 Hz for physiological signals)"
            )

        return True


class EdfLoader:
    """Loader for EDF/EDF+ and BDF (Biosemi 24-bit) files via edfio.

    Loads physiological signals with per-channel sampling rates and automatic
    signal type detection from channel labels.  EDF+ annotations are mapped to
    EventData.  Timestamps are generated uniformly from each channel's sample
    count and sampling frequency, so every channel is zero-referenced to the
    recording start and shares a common time axis.

    EDF/EDF+ files are typically polysomnographic recordings with many
    non-cardiac channels (EEG, EOG, EMG, respiration, SpO2, ...).  By default
    only channels whose label maps to a recognized physiological signal type
    (ECG, PPG, EDA) are kept; everything else is skipped and reported in
    ``filtered_channels``.  This deliberately drops derived interval channels
    (e.g. "RRI", "RR", "IBI"): RR intervals are event/interval data, not a
    continuous signal, and are recomputed from the ECG via the program's own
    peak-detection and interval-analysis pipeline instead of being trusted from
    an idiosyncratic channel.
    """

    def __init__(self, include_unknown: bool = False):
        """Initialize EDF loader.

        Args:
            include_unknown: If True, also load channels whose label does not
                map to a recognized signal type (default: False, keep only
                ECG/PPG/EDA).
        """
        self.include_unknown = include_unknown
        self.skipped_streams = []  # Track channels that failed to load (zero-rate, too short)
        self.filtered_channels = []  # Track channels dropped by label-based filtering

    def can_load(self, path: Path) -> bool:
        """Check if file is EDF or BDF format."""
        return path.suffix.lower() in (".edf", ".bdf")

    def load(self, path: Path) -> RecordingSession:
        """Load EDF/EDF+ or BDF file.

        Args:
            path: Path to EDF/BDF file

        Returns:
            RecordingSession with loaded signals

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self.skipped_streams = []
        self.filtered_channels = []

        # Layered validation: type -> exists -> format
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception:
                raise TypeError(f"path must be a Path object or string, got {type(path).__name__}")

        if not path.exists():
            raise FileNotFoundError(f"EDF/BDF file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() not in (".edf", ".bdf"):
            raise ValueError(f"Invalid file extension, expected .edf or .bdf, got {path.suffix}")

        logger.info(f"Loading EDF/BDF file: {path}")

        try:
            import edfio
        except ImportError:
            raise ValueError("edfio not installed. Install with: pip install edfio")

        # BDF files use the 24-bit Biosemi variant; branch on extension
        try:
            if path.suffix.lower() == ".bdf":
                edf = edfio.read_bdf(path)
            else:
                edf = edfio.read_edf(path)
        except Exception as e:
            raise ValueError(f"Failed to load EDF/BDF file: {e}")

        # Extract signals (edfio excludes EDF+ annotation channels from .signals)
        signals = []
        for signal in edf.signals:
            channel_name = signal.label
            sampling_rate = float(signal.sampling_frequency)

            # Status/annotation-adjacent channels can carry a zero sampling rate
            if sampling_rate <= 0:
                logger.warning(f"Skipping channel '{channel_name}' (sampling rate {sampling_rate} Hz)")
                self.skipped_streams.append(channel_name)
                continue

            # Label-based filtering: keep only recognized physiological signals
            # by default.  EDF/EDF+ has no standard for RR-interval channels
            # (Kemp & Olivan, 2003); derived intervals appear only in
            # vendor-specific channels, so they are dropped here and recomputed
            # from the ECG via the peak-detection + interval-analysis pipeline.
            signal_type = detect_signal_type_from_name(channel_name)
            if signal_type == SignalType.UNKNOWN and not self.include_unknown:
                logger.info(
                    f"Skipping channel '{channel_name}' (unrecognized signal type; "
                    f"not ECG/PPG/EDA). Derived interval channels such as RRI are "
                    f"recomputed from the ECG signal."
                )
                self.filtered_channels.append(channel_name)
                continue

            # Copy: edfio may return read-only/lazily-loaded arrays; SignalData owns writable data
            samples = np.array(signal.data, dtype=np.float64)

            if len(samples) < 2:
                logger.warning(f"Skipping channel '{channel_name}' (fewer than 2 samples)")
                self.skipped_streams.append(channel_name)
                continue

            if np.all(samples == 0) or np.all(np.isnan(samples)):
                logger.debug(f"Skipping empty channel: {channel_name}")
                continue

            # Uniform timestamps zero-referenced to recording start
            timestamps = np.arange(len(samples), dtype=np.float64) / sampling_rate

            if sampling_rate < 16 or sampling_rate > 2000:
                logger.warning(
                    f"Unusual sampling rate for '{channel_name}': {sampling_rate} Hz "
                    f"(expected 16-2000 Hz for physiological signals)"
                )

            signals.append(
                SignalData(
                    samples=samples,
                    sampling_rate=sampling_rate,
                    timestamps=timestamps,
                    channel_name=channel_name,
                    signal_type=signal_type,
                    unit=getattr(signal, "physical_dimension", "") or "",
                )
            )

        if not signals:
            raise ValueError(f"No physiological signals found in EDF/BDF file: {path}")

        if self.filtered_channels:
            logger.info(
                f"Filtered out {len(self.filtered_channels)} unrecognized channel(s): "
                f"{self.filtered_channels}"
            )

        logger.info(f"Loaded {len(signals)} signals from EDF/BDF file")

        # EDF+ annotations -> events (onsets are already relative to recording start)
        events = self._extract_annotations(edf)

        return RecordingSession(source_path=path, signals=signals, events=events)

    def _extract_annotations(self, edf) -> list[EventData]:
        """Extract EDF+ annotations as EventData.

        Args:
            edf: edfio Edf (or Bdf) object

        Returns:
            List of EventData objects
        """
        events = []
        try:
            annotations = edf.annotations
        except Exception as e:
            logger.warning(f"Failed to read EDF annotations: {e}")
            return events

        for ann in annotations:
            duration = float(ann.duration) if ann.duration is not None and ann.duration > 0 else None
            events.append(
                EventData(
                    timestamp=float(ann.onset),
                    label=str(ann.text),
                    duration=duration,
                )
            )

        if events:
            logger.info(f"Loaded {len(events)} annotations from EDF/BDF file")
        return events


def get_loader(path: Path) -> FileLoader:
    """Get appropriate file loader for the given path.

    Args:
        path: Path to file

    Returns:
        FileLoader instance

    Raises:
        ValueError: If no loader can handle the file

    Example:
        >>> loader = get_loader(Path("data.xdf"))
        >>> session = loader.load(Path("data.xdf"))
    """
    path = Path(path)

    # Try each loader (order matters - try specific formats before generic ones)
    loaders = [XdfLoader(), MatLoader(), EdfLoader(), CsvLoader()]

    for loader in loaders:
        if loader.can_load(path):
            return loader

    raise ValueError(f"No loader available for file: {path}")
