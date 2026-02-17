"""Signal processing modules for CardioSignalLab.

Provides composable pipeline, filtering, EEMD artifact removal,
NeuroKit2 peak detection, peak correction, and background worker.
"""

from cardio_signal_lab.processing.pipeline import (
    ProcessingPipeline,
    get_operation,
    list_operations,
    register_operation,
)
from cardio_signal_lab.processing.filters import (
    bandpass_filter,
    baseline_correction,
    highpass_filter,
    lowpass_filter,
    notch_filter,
    segment_signal,
    zero_reference,
)
from cardio_signal_lab.processing.peak_detection import (
    detect_ecg_peaks,
    detect_eda_features,
    detect_ppg_peaks,
)
from cardio_signal_lab.processing.peak_correction import PeakEditor
from cardio_signal_lab.processing.worker import ProcessingWorker
from cardio_signal_lab.processing.bad_segment_detection import (
    detect_bad_segments,
    detect_amplitude_artifacts,
    detect_timestamp_gaps,
    interpolate_bad_segments,
)

# Note: EEMD is imported lazily to avoid slow PyEMD import at startup.
# Use: from cardio_signal_lab.processing.eemd import eemd_artifact_removal

__all__ = [
    # Pipeline
    "ProcessingPipeline",
    "register_operation",
    "get_operation",
    "list_operations",
    # Filters
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
    "notch_filter",
    "baseline_correction",
    "zero_reference",
    "segment_signal",
    # Peak detection
    "detect_ecg_peaks",
    "detect_ppg_peaks",
    "detect_eda_features",
    # Peak correction
    "PeakEditor",
    # Worker
    "ProcessingWorker",
    # Bad segment detection
    "detect_bad_segments",
    "detect_amplitude_artifacts",
    "detect_timestamp_gaps",
    "interpolate_bad_segments",
]
