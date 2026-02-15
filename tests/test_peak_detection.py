"""Tests for NeuroKit2 peak detection wrappers.

Uses synthetic signals with known peak locations to verify detection accuracy.
"""
import numpy as np
import pytest

from cardio_signal_lab.processing.peak_detection import (
    detect_ecg_peaks,
    detect_eda_features,
    detect_ppg_peaks,
)


@pytest.fixture
def synthetic_ecg():
    """Create synthetic ECG-like signal with known R-peaks.

    Generates a simple signal with periodic sharp peaks at 1 Hz (60 bpm)
    at 250 Hz sampling rate.
    """
    import neurokit2 as nk
    ecg = nk.ecg_simulate(duration=10, sampling_rate=250, heart_rate=72)
    return np.array(ecg), 250.0


@pytest.fixture
def synthetic_ppg():
    """Create synthetic PPG signal with known peaks."""
    import neurokit2 as nk
    ppg = nk.ppg_simulate(duration=10, sampling_rate=250, heart_rate=72)
    return np.array(ppg), 250.0


@pytest.fixture
def synthetic_eda():
    """Create synthetic EDA signal with SCR events."""
    import neurokit2 as nk
    eda = nk.eda_simulate(duration=30, sampling_rate=250, scr_number=5)
    return np.array(eda), 250.0


class TestDetectECGPeaks:
    def test_detects_peaks(self, synthetic_ecg):
        signal, sr = synthetic_ecg
        peaks = detect_ecg_peaks(signal, sr)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) > 0
        # At 72 bpm over 10 seconds, expect ~12 peaks
        assert 8 <= len(peaks) <= 16

    def test_peaks_within_bounds(self, synthetic_ecg):
        signal, sr = synthetic_ecg
        peaks = detect_ecg_peaks(signal, sr)
        assert all(0 <= p < len(signal) for p in peaks)

    def test_peaks_are_sorted(self, synthetic_ecg):
        signal, sr = synthetic_ecg
        peaks = detect_ecg_peaks(signal, sr)
        assert all(peaks[i] < peaks[i + 1] for i in range(len(peaks) - 1))

    def test_integer_indices(self, synthetic_ecg):
        signal, sr = synthetic_ecg
        peaks = detect_ecg_peaks(signal, sr)
        assert peaks.dtype in (np.int32, np.int64, int)


class TestDetectPPGPeaks:
    def test_detects_peaks(self, synthetic_ppg):
        signal, sr = synthetic_ppg
        peaks = detect_ppg_peaks(signal, sr)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) > 0
        # At 72 bpm over 10 seconds, expect ~12 peaks
        assert 8 <= len(peaks) <= 16

    def test_peaks_within_bounds(self, synthetic_ppg):
        signal, sr = synthetic_ppg
        peaks = detect_ppg_peaks(signal, sr)
        assert all(0 <= p < len(signal) for p in peaks)


class TestDetectEDAFeatures:
    def test_detects_scr_peaks(self, synthetic_eda):
        signal, sr = synthetic_eda
        peaks = detect_eda_features(signal, sr)
        assert isinstance(peaks, np.ndarray)
        # Should detect some SCR peaks (simulated 5)
        assert len(peaks) >= 1

    def test_peaks_within_bounds(self, synthetic_eda):
        signal, sr = synthetic_eda
        peaks = detect_eda_features(signal, sr)
        if len(peaks) > 0:
            assert all(0 <= p < len(signal) for p in peaks)
