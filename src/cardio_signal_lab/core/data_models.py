"""Data models for physiological signal storage and processing.

Uses attrs with validators for type-safe, validated data containers.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pandas as pd
from attrs import define, field


class SignalType(Enum):
    """Physiological signal types."""

    ECG = "ecg"
    PPG = "ppg"
    EDA = "eda"
    UNKNOWN = "unknown"


class ProcessingState(Enum):
    """Signal processing pipeline state."""

    RAW = "raw"
    FILTERED = "filtered"
    PEAKS_DETECTED = "peaks_detected"
    CORRECTED = "corrected"


class PeakClassification(Enum):
    """Peak classification types for interactive editing."""

    AUTO = 0  # Auto-detected (blue) - from NeuroKit2
    MANUAL = 1  # Manually added (green) - user double-click
    ECTOPIC = 2  # Ectopic beat (magenta) - user marked
    BAD = 3  # Bad/incorrect (red) - artifact


def _validate_ndarray_1d(instance, attribute, value):
    """Validator: ensure value is a 1D numpy array."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{attribute.name} must be ndarray, got {type(value).__name__}")
    if value.ndim != 1:
        raise ValueError(f"{attribute.name} must be 1D, got shape {value.shape}")


def _validate_ndarray_1d_min_2(instance, attribute, value):
    """Validator: ensure 1D array with at least 2 elements."""
    _validate_ndarray_1d(instance, attribute, value)
    if len(value) < 2:
        raise ValueError(f"{attribute.name} must have >= 2 elements, got {len(value)}")


def _validate_timestamps_monotonic(instance, attribute, value):
    """Validator: ensure timestamps are strictly increasing."""
    _validate_ndarray_1d_min_2(instance, attribute, value)
    diffs = np.diff(value)
    if not np.all(diffs > 0):
        non_monotonic_idx = np.where(diffs <= 0)[0]
        raise ValueError(
            f"{attribute.name} must be strictly increasing. "
            f"Non-monotonic at indices: {non_monotonic_idx[:5].tolist()}"
        )


def _validate_positive(instance, attribute, value):
    """Validator: ensure value is positive."""
    if value <= 0:
        raise ValueError(f"{attribute.name} must be positive, got {value}")


def _validate_positive_or_zero(instance, attribute, value):
    """Validator: ensure value is positive or zero."""
    if value < 0:
        raise ValueError(f"{attribute.name} must be >= 0, got {value}")


@define
class ChannelInfo:
    """Information about a single signal channel."""

    name: str = field(validator=attrs.validators.instance_of(str))
    unit: str = field(validator=attrs.validators.instance_of(str))
    channel_type: str = field(default="physiological", validator=attrs.validators.instance_of(str))
    location: str | None = field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(str)))
    impedance: float | None = field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(float)))


@define
class TimestampInfo:
    """Timestamp information for a signal.

    Maintains dual timestamp system:
    - timestamps: Authoritative device timestamps for signal timing
    - lsl_alignment_*: LSL reference times for multi-stream alignment (optional)
    """

    timestamps: np.ndarray = field(validator=_validate_timestamps_monotonic)
    lsl_alignment_start: float = field(default=0.0, validator=attrs.validators.instance_of(float))
    lsl_alignment_end: float = field(default=0.0, validator=attrs.validators.instance_of(float))
    zero_reference: float = field(default=0.0, validator=attrs.validators.instance_of(float))
    gaps: list[dict[str, Any]] | None = field(default=None)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def num_samples(self) -> int:
        """Number of timestamp samples."""
        return len(self.timestamps)

    def get_relative_timestamps(self) -> np.ndarray:
        """Get timestamps relative to zero_reference."""
        return self.timestamps - self.zero_reference


@define
class PeakData:
    """Peak annotation data with classification tracking.

    Stores peak indices and their classifications for interactive editing.
    Classifications:
    - AUTO (0): Auto-detected by NeuroKit2 (blue)
    - MANUAL (1): User-added via double-click (green)
    - ECTOPIC (2): User-marked ectopic beat (magenta)
    - BAD (3): User-marked as artifact (red)
    """

    indices: np.ndarray = field(validator=_validate_ndarray_1d)
    classifications: np.ndarray = field(validator=_validate_ndarray_1d)  # PeakClassification values

    def __attrs_post_init__(self):
        """Validate indices and classifications have same length."""
        if len(self.indices) != len(self.classifications):
            raise ValueError(
                f"indices ({len(self.indices)}) and classifications ({len(self.classifications)}) must have same length"
            )

    @property
    def num_peaks(self) -> int:
        """Total number of peaks."""
        return len(self.indices)

    @property
    def num_auto(self) -> int:
        """Number of auto-detected peaks."""
        return int(np.sum(self.classifications == PeakClassification.AUTO.value))

    @property
    def num_manual(self) -> int:
        """Number of manually-added peaks."""
        return int(np.sum(self.classifications == PeakClassification.MANUAL.value))

    @property
    def num_ectopic(self) -> int:
        """Number of ectopic peaks."""
        return int(np.sum(self.classifications == PeakClassification.ECTOPIC.value))

    @property
    def num_bad(self) -> int:
        """Number of bad/incorrect peaks."""
        return int(np.sum(self.classifications == PeakClassification.BAD.value))


@define
class EventData:
    """Event marker with timestamp and label.

    Represents experimental events (e.g., baseline_start, stimulus_onset)
    with timestamps aligned to signal data.  Timestamps are zero-referenced
    relative to the recording start (lsl_t0_reference) and may be negative
    if an event was sent before the physiological signal stream began.
    """

    timestamp: float = field(validator=attrs.validators.instance_of(float))
    label: str = field(validator=attrs.validators.instance_of(str))
    duration: float | None = field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(float)))
    metadata: dict[str, Any] = field(factory=dict, validator=attrs.validators.instance_of(dict))


@define
class ProcessingStep:
    """Record of a single processing operation."""

    operation: str = field(validator=attrs.validators.instance_of(str))
    parameters: dict[str, Any] = field(factory=dict, validator=attrs.validators.instance_of(dict))
    timestamp: float | None = field(default=None)


@define
class SignalData:
    """Single physiological signal with metadata.

    Represents one signal type (ECG, PPG, or EDA) with its samples, timestamps,
    and associated metadata.
    """

    samples: np.ndarray = field(validator=_validate_ndarray_1d)
    sampling_rate: float = field(validator=[attrs.validators.instance_of(float), _validate_positive])
    timestamps: np.ndarray = field(validator=_validate_timestamps_monotonic)
    channel_name: str = field(validator=attrs.validators.instance_of(str))
    signal_type: SignalType = field(validator=attrs.validators.instance_of(SignalType))

    # Optional metadata
    unit: str = field(default="", validator=attrs.validators.instance_of(str))
    quality: np.ndarray | None = field(default=None, validator=attrs.validators.optional(_validate_ndarray_1d))
    # Raw LSL timestamps (absolute clock values, before zero-referencing); None for CSV files
    lsl_timestamps: np.ndarray | None = field(default=None, validator=attrs.validators.optional(_validate_ndarray_1d))

    def __attrs_post_init__(self):
        """Validate samples and timestamps have same length."""
        if len(self.samples) != len(self.timestamps):
            raise ValueError(
                f"samples ({len(self.samples)}) and timestamps ({len(self.timestamps)}) must have same length"
            )

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.samples)

    @property
    def mean_sampling_rate(self) -> float:
        """Mean sampling rate computed from timestamps."""
        return self.num_samples / self.duration if self.duration > 0 else 0.0


@define
class DerivedSignalData:
    """Signal derived from multiple source channels (e.g., L2 Norm).

    Stores the computed signal along with metadata about its derivation.
    """

    samples: np.ndarray = field(validator=_validate_ndarray_1d)
    sampling_rate: float = field(validator=[attrs.validators.instance_of(float), _validate_positive])
    timestamps: np.ndarray = field(validator=_validate_timestamps_monotonic)
    channel_name: str = field(validator=attrs.validators.instance_of(str))
    signal_type: SignalType = field(validator=attrs.validators.instance_of(SignalType))
    source_channels: list[str] = field(factory=list, validator=attrs.validators.instance_of(list))
    derivation_method: str = field(default="", validator=attrs.validators.instance_of(str))

    # Optional metadata
    unit: str = field(default="", validator=attrs.validators.instance_of(str))

    def __attrs_post_init__(self):
        """Validate samples and timestamps have same length."""
        if len(self.samples) != len(self.timestamps):
            raise ValueError(
                f"samples ({len(self.samples)}) and timestamps ({len(self.timestamps)}) must have same length"
            )

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.samples)


def create_l2_norm(signals: list[SignalData]) -> DerivedSignalData:
    """Create L2 Norm derived channel from multiple signals.

    Computes sqrt(sum(x_i^2)) across channels at each time point.

    Args:
        signals: List of SignalData (must have same timestamps and sampling rate)

    Returns:
        DerivedSignalData with L2 norm values

    Raises:
        ValueError: If signals list is empty or signals have mismatched lengths
    """
    if not signals:
        raise ValueError("Cannot create L2 norm from empty signal list")

    if len(signals) == 1:
        raise ValueError("L2 norm requires at least 2 signals")

    # Validate all signals have same length
    n_samples = len(signals[0].samples)
    for sig in signals[1:]:
        if len(sig.samples) != n_samples:
            raise ValueError(
                f"All signals must have same length. "
                f"Got {n_samples} and {len(sig.samples)}"
            )

    # Compute L2 norm: sqrt(sum(x_i^2))
    squared_sum = np.zeros(n_samples, dtype=np.float64)
    for sig in signals:
        squared_sum += sig.samples ** 2
    l2_norm = np.sqrt(squared_sum)

    return DerivedSignalData(
        samples=l2_norm,
        sampling_rate=signals[0].sampling_rate,
        timestamps=signals[0].timestamps.copy(),
        channel_name="L2 Norm",
        signal_type=signals[0].signal_type,
        source_channels=[sig.channel_name for sig in signals],
        derivation_method="l2_norm",
    )


@define
class RecordingSession:
    """Complete multi-signal recording session.

    Contains multiple signals (ECG, PPG, EDA) loaded from a single file,
    along with processing history and metadata.
    """

    # Mandatory fields must come first
    source_path: Path = field(converter=Path, validator=attrs.validators.instance_of(Path))
    signals: list[SignalData] = field(factory=list, validator=attrs.validators.instance_of(list))

    # Optional fields with defaults
    processing_history: list[ProcessingStep] = field(factory=list, validator=attrs.validators.instance_of(list))
    events: list[EventData] = field(factory=list, validator=attrs.validators.instance_of(list))
    derived_signals: list[DerivedSignalData] = field(factory=list, validator=attrs.validators.instance_of(list))
    metadata: dict[str, Any] = field(factory=dict, validator=attrs.validators.instance_of(dict))
    # First LSL timestamp of the primary physiological signal stream.
    # Used as the common zero-reference for both signal and event timestamps in XDF files.
    lsl_t0_reference: float | None = field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(float)))

    @property
    def num_signals(self) -> int:
        """Number of signals in session."""
        return len(self.signals)

    @property
    def signal_types(self) -> list[SignalType]:
        """List of signal types present."""
        return [sig.signal_type for sig in self.signals]

    def get_signal(self, signal_type: SignalType) -> SignalData | None:
        """Get first signal of specified type.

        Args:
            signal_type: Signal type to retrieve

        Returns:
            SignalData if found, None otherwise
        """
        for signal in self.signals:
            if signal.signal_type == signal_type:
                return signal
        return None

    def get_all_signals(self, signal_type: SignalType) -> list[SignalData]:
        """Get all signals of specified type.

        Args:
            signal_type: Signal type to retrieve

        Returns:
            List of SignalData (may be empty)
        """
        return [sig for sig in self.signals if sig.signal_type == signal_type]

    def has_signal_type(self, signal_type: SignalType) -> bool:
        """Check if session contains signal type.

        Args:
            signal_type: Signal type to check

        Returns:
            True if at least one signal of this type exists
        """
        return any(sig.signal_type == signal_type for sig in self.signals)
