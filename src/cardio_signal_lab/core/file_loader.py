"""File loading with Protocol-based architecture for extensibility.

Supports XDF and CSV files with automatic signal type detection.
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

        # Extract signals from streams (skip marker streams)
        signals = []
        for stream in streams:
            # Skip event/marker streams (they have time_series as list, not array)
            if isinstance(stream["time_series"], list):
                continue
            signal_list = self._extract_signals_from_stream(stream)
            signals.extend(signal_list)

        if not signals:
            raise ValueError(f"No physiological signals found in XDF file: {path}")

        logger.info(f"Loaded {len(signals)} signals from XDF file")

        # Extract events from marker streams
        events = self._extract_events_from_streams(streams)

        return RecordingSession(source_path=path, signals=signals, events=events)

    def _load_xdf_streams(self, path: Path) -> tuple:
        """Load XDF streams with selective loading.

        Returns:
            Tuple of (streams, header)
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
            logger.warning(f"Selective loading failed: {e}, loading all streams")
            streams, header = pyxdf.load_xdf(str(path))

        logger.info(f"Loaded {len(streams)} streams from XDF")
        return streams, header

    def _extract_signals_from_stream(self, stream: dict) -> list[SignalData]:
        """Extract SignalData objects from an XDF stream.

        Args:
            stream: XDF stream dict

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
            return []

        # Detect signal type from stream metadata (used as fallback)
        stream_signal_type = self._detect_signal_type(stream["info"])

        # Extract timestamps (use device timestamps if available, otherwise LSL)
        timestamps = self._extract_timestamps(stream)

        # Validate timestamps
        if not self._validate_timestamps(timestamps, stream_name):
            logger.warning(f"Timestamps not monotonic for '{stream_name}', but continuing load")

        # Extract channel names
        channel_names = self._extract_channel_names(stream["info"])

        # Extract time series data
        time_series = stream["time_series"]

        # Handle timestamp column (first column in Shimmer streams)
        if "shimmer" in stream_name.lower() or time_series.shape[1] == len(channel_names):
            # First column is device timestamp
            start_col = 1
            channel_names = channel_names[1:]  # Skip timestamp column
        else:
            start_col = 0

        # Create SignalData for each channel
        for i, channel_name in enumerate(channel_names):
            col_idx = start_col + i
            if col_idx < time_series.shape[1]:
                samples = time_series[:, col_idx].astype(np.float64)

                # Skip channels with all zeros or NaNs
                if np.all(samples == 0) or np.all(np.isnan(samples)):
                    logger.debug(f"Skipping empty channel: {channel_name}")
                    continue

                # Validate signal values
                if not self._validate_signal_values(samples, channel_name):
                    logger.warning(f"Signal '{channel_name}' failed validation, but including in session")
                    # Continue anyway - validation warnings are informational

                # Per-channel signal type detection (e.g., GSR vs PPG from same stream)
                channel_type = detect_signal_type_from_name(channel_name)
                if channel_type == SignalType.UNKNOWN:
                    channel_type = stream_signal_type

                signal = SignalData(
                    samples=samples,
                    sampling_rate=sampling_rate,
                    timestamps=timestamps,
                    channel_name=channel_name,
                    signal_type=channel_type,
                )
                signals.append(signal)

        logger.debug(f"Extracted {len(signals)} signals from stream '{stream_name}'")
        return signals

    def _extract_timestamps(self, stream: dict) -> np.ndarray:
        """Extract timestamps from stream.

        Uses device timestamps if available (first column of time_series),
        otherwise falls back to LSL timestamps. Aligns to start at t=0.

        Args:
            stream: XDF stream dict

        Returns:
            1D array of timestamps in seconds, starting at 0
        """
        # Check if first column looks like device timestamps (in milliseconds)
        time_series = stream["time_series"]
        if time_series.shape[1] > 1:
            first_col = time_series[:, 0]
            # Device timestamps are typically large values in milliseconds
            if np.mean(first_col) > 1000:
                timestamps = first_col / 1000.0  # Convert ms to seconds
                timestamps = timestamps - timestamps[0]  # Align to t=0
                logger.debug("Using device timestamps from first column (aligned to t=0)")
                return timestamps.astype(np.float64)

        # Fall back to LSL timestamps
        timestamps = stream["time_stamps"].astype(np.float64)
        timestamps = timestamps - timestamps[0]  # Align to t=0
        logger.debug("Using LSL timestamps (aligned to t=0)")
        return timestamps

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

    def _extract_events_from_streams(self, streams: list) -> list[EventData]:
        """Extract event markers from XDF streams.

        Looks for streams with type "Markers" or similar event streams.
        Event streams typically have nominal_srate=0 and time_series as list of markers.

        Args:
            streams: List of XDF streams

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

            # Convert timestamps to numpy array and align to t=0
            time_stamps = np.array(time_stamps, dtype=np.float64)

            # XDF LSL timestamps are always in seconds (absolute epoch time)
            # Just align to t=0 (same as signals)
            time_stamps = time_stamps - time_stamps[0]
            logger.debug(f"Aligned event timestamps to t=0 (range: {time_stamps[0]:.2f}s to {time_stamps[-1]:.2f}s)")

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
        - Row 2: Calibration status
        - Row 3: Units
        - Row 4+: Data

        Args:
            path: Path to CSV file

        Returns:
            RecordingSession with loaded signals
        """
        # Load CSV skipping first 3 rows (header metadata)
        df = pd.read_csv(path, skiprows=3, header=None)

        if len(df) < 2:
            raise ValueError("CSV must have at least 2 data rows")

        # Read header row separately to get column names
        header_df = pd.read_csv(path, nrows=1)
        column_names = header_df.columns.tolist()

        # Assign column names to data
        df.columns = column_names[:len(df.columns)]

        # First column is time (in milliseconds for Shimmer)
        time_col = df.columns[0]
        timestamps_ms = df[time_col].values.astype(np.float64)

        # Convert milliseconds to seconds
        timestamps = timestamps_ms / 1000.0

        # Align to start at time 0
        timestamps = timestamps - timestamps[0]

        # Validate timestamps
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

    # Try each loader
    loaders = [XdfLoader(), CsvLoader()]

    for loader in loaders:
        if loader.can_load(path):
            return loader

    raise ValueError(f"No loader available for file: {path}")
