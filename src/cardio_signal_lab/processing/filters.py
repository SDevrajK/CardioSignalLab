"""Signal filtering and preprocessing operations.

Provides bandpass filtering, baseline correction, zero-referencing, and
signal segmentation. All functions follow the pipeline convention:
    func(samples, sampling_rate, **params) -> processed_samples

Ported from EKG_Peak_Corrector v2 signal_filtering.py with simplification.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

from cardio_signal_lab.processing.pipeline import register_operation


def detect_signal_dropouts(
    timestamps: np.ndarray, sampling_rate: float, threshold_multiplier: float = 1.5
) -> tuple[np.ndarray, float]:
    """Detect signal dropouts (gaps in timestamps).

    Args:
        timestamps: Timestamp array in seconds
        sampling_rate: Expected sampling rate in Hz
        threshold_multiplier: Gap threshold as multiple of expected interval (default: 1.5)

    Returns:
        Tuple of (gap_indices, dropout_percentage):
            - gap_indices: Indices where gaps occur (shape: (N,))
            - dropout_percentage: Percentage of total duration that is dropout
    """
    if len(timestamps) < 2:
        return np.array([], dtype=int), 0.0

    # Calculate time differences
    time_diffs = np.diff(timestamps)

    # Expected interval between samples
    expected_interval = 1.0 / sampling_rate

    # Detect gaps (intervals larger than threshold)
    gap_threshold = expected_interval * threshold_multiplier
    gap_indices = np.where(time_diffs > gap_threshold)[0]

    if len(gap_indices) == 0:
        return gap_indices, 0.0

    # Calculate dropout percentage
    gap_durations = time_diffs[gap_indices] - expected_interval
    total_gap_duration = np.sum(gap_durations)
    total_duration = timestamps[-1] - timestamps[0]
    dropout_percentage = 100.0 * total_gap_duration / total_duration if total_duration > 0 else 0.0

    logger.debug(
        f"Detected {len(gap_indices)} dropouts ({dropout_percentage:.2f}% of signal duration)"
    )

    return gap_indices, dropout_percentage


def bandpass_filter(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Retains frequency content between lowcut and highcut, removing baseline
    wander and high-frequency noise. Uses sosfiltfilt for zero-phase filtering
    to preserve peak timing.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        lowcut: Lower cutoff frequency in Hz (removes baseline wander)
        highcut: Upper cutoff frequency in Hz (removes high-freq noise)
        order: Filter order (higher = sharper rolloff, more phase distortion)

    Returns:
        Bandpass filtered signal
    """
    nyquist = sampling_rate / 2.0

    if lowcut <= 0:
        raise ValueError(f"lowcut must be positive, got {lowcut}")
    if highcut >= nyquist:
        logger.warning(
            f"highcut ({highcut} Hz) >= Nyquist ({nyquist} Hz), "
            f"clamping to {nyquist * 0.95:.1f} Hz"
        )
        highcut = nyquist * 0.95
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")

    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, samples)

    logger.debug(
        f"Bandpass filter applied: {lowcut}-{highcut} Hz, order {order}, "
        f"sampling rate {sampling_rate} Hz"
    )
    return filtered


def highpass_filter(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    cutoff: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth highpass filter.

    Removes baseline wander below cutoff frequency. 0.5 Hz cutoff preserves
    cardiac signals (HR >= 30 bpm = 0.5 Hz) while removing respiratory
    drift (~0.2-0.3 Hz) and electrode motion artifacts.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Highpass filtered signal
    """
    nyquist = sampling_rate / 2.0

    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if cutoff >= nyquist:
        raise ValueError(f"cutoff ({cutoff}) must be < Nyquist ({nyquist})")

    sos = butter(order, cutoff / nyquist, btype="high", output="sos")
    filtered = sosfiltfilt(sos, samples)

    logger.debug(f"Highpass filter applied: {cutoff} Hz, order {order}")
    return filtered


def lowpass_filter(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    cutoff: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth lowpass filter.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Lowpass filtered signal
    """
    nyquist = sampling_rate / 2.0

    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if cutoff >= nyquist:
        logger.warning(f"cutoff ({cutoff}) >= Nyquist ({nyquist}), clamping")
        cutoff = nyquist * 0.95

    sos = butter(order, cutoff / nyquist, btype="low", output="sos")
    filtered = sosfiltfilt(sos, samples)

    logger.debug(f"Lowpass filter applied: {cutoff} Hz, order {order}")
    return filtered


def notch_filter(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    freq: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove powerline interference.

    Quality factor of 30 gives a narrow notch (~2 Hz bandwidth) to minimize
    distortion of nearby frequencies. Use 60 Hz for North America, 50 Hz for
    Europe.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        freq: Notch frequency in Hz (50 or 60 for powerline)
        quality_factor: Q factor (higher = narrower notch)

    Returns:
        Notch filtered signal
    """
    nyquist = sampling_rate / 2.0

    if freq >= nyquist:
        logger.warning(
            f"Notch frequency ({freq} Hz) >= Nyquist ({nyquist} Hz), skipping"
        )
        return samples.copy()

    b, a = iirnotch(freq, quality_factor, sampling_rate)
    filtered = filtfilt(b, a, samples)

    logger.debug(f"Notch filter applied: {freq} Hz, Q={quality_factor}")
    return filtered


def baseline_correction(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    poly_order: int = 3,
) -> np.ndarray:
    """Remove baseline drift via polynomial detrending.

    Fits a polynomial to the signal and subtracts it, removing slow baseline
    drift from respiration and electrode motion.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz (unused, kept for pipeline compat)
        poly_order: Polynomial order for fitting (1=linear, 3=cubic)

    Returns:
        Baseline-corrected signal
    """
    x = np.arange(len(samples))
    coeffs = np.polyfit(x, samples, poly_order)
    baseline = np.polyval(coeffs, x)
    corrected = samples - baseline

    logger.debug(f"Baseline correction applied: polynomial order {poly_order}")
    return corrected


def zero_reference(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    method: str = "mean",
    n_samples: int = 100,
) -> np.ndarray:
    """Zero-reference signal by subtracting an offset.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz (unused, kept for pipeline compat)
        method: 'mean' to subtract signal mean, 'first_n' to subtract mean
                of first n_samples
        n_samples: Number of samples for 'first_n' method

    Returns:
        Zero-referenced signal
    """
    if method == "mean":
        offset = np.mean(samples)
    elif method == "first_n":
        n = min(n_samples, len(samples))
        offset = np.mean(samples[:n])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'first_n'")

    result = samples - offset

    logger.debug(f"Zero-reference applied: method={method}, offset={offset:.4f}")
    return result


def segment_signal(
    samples: np.ndarray,
    sampling_rate: float,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> np.ndarray:
    """Extract a time segment from the signal.

    Note: This changes the signal length, so timestamps must also be updated
    by the caller. In the pipeline, this should typically be the last step.

    Args:
        samples: Signal samples (1D array)
        sampling_rate: Sampling rate in Hz
        start_time: Start time in seconds
        end_time: End time in seconds (None = end of signal)

    Returns:
        Segmented signal
    """
    start_idx = int(start_time * sampling_rate)
    start_idx = max(0, min(start_idx, len(samples) - 1))

    if end_time is None:
        end_idx = len(samples)
    else:
        end_idx = int(end_time * sampling_rate)
        end_idx = max(start_idx + 1, min(end_idx, len(samples)))

    result = samples[start_idx:end_idx].copy()

    logger.debug(
        f"Segment extracted: {start_time:.2f}s-{end_time or 'end'}s "
        f"({len(result)} samples)"
    )
    return result


# --- Pipeline operation wrappers ---
# These adapt the keyword-argument functions to pipeline convention:
# func(samples, sampling_rate, **params) -> samples


def _pipeline_bandpass(samples, sampling_rate, **params):
    return bandpass_filter(samples, sampling_rate, **params)


def _pipeline_highpass(samples, sampling_rate, **params):
    return highpass_filter(samples, sampling_rate, **params)


def _pipeline_lowpass(samples, sampling_rate, **params):
    return lowpass_filter(samples, sampling_rate, **params)


def _pipeline_notch(samples, sampling_rate, **params):
    return notch_filter(samples, sampling_rate, **params)


def _pipeline_baseline(samples, sampling_rate, **params):
    return baseline_correction(samples, sampling_rate, **params)


def _pipeline_zero_reference(samples, sampling_rate, **params):
    return zero_reference(samples, sampling_rate, **params)


def _pipeline_segment(samples, sampling_rate, **params):
    return segment_signal(samples, sampling_rate, **params)


# Register all operations
register_operation("bandpass", _pipeline_bandpass)
register_operation("highpass", _pipeline_highpass)
register_operation("lowpass", _pipeline_lowpass)
register_operation("notch", _pipeline_notch)
register_operation("baseline_correction", _pipeline_baseline)
register_operation("zero_reference", _pipeline_zero_reference)
register_operation("segment", _pipeline_segment)
