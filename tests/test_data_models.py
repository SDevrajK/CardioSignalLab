"""Unit tests for core data models."""
from pathlib import Path

import numpy as np
import pytest

from cardio_signal_lab.core import (
    ChannelInfo,
    PeakData,
    ProcessingState,
    ProcessingStep,
    RecordingSession,
    SignalData,
    SignalType,
    TimestampInfo,
)


class TestSignalType:
    """Tests for SignalType enum."""

    def test_signal_types_exist(self):
        """Test that all expected signal types are defined."""
        assert SignalType.ECG.value == "ecg"
        assert SignalType.PPG.value == "ppg"
        assert SignalType.EDA.value == "eda"
        assert SignalType.UNKNOWN.value == "unknown"


class TestProcessingState:
    """Tests for ProcessingState enum."""

    def test_processing_states_exist(self):
        """Test that all expected processing states are defined."""
        assert ProcessingState.RAW.value == "raw"
        assert ProcessingState.FILTERED.value == "filtered"
        assert ProcessingState.PEAKS_DETECTED.value == "peaks_detected"
        assert ProcessingState.CORRECTED.value == "corrected"


class TestChannelInfo:
    """Tests for ChannelInfo data model."""

    def test_channel_info_creation(self):
        """Test valid ChannelInfo creation."""
        channel = ChannelInfo(
            name="ECG Lead I",
            unit="mV",
        )
        assert channel.name == "ECG Lead I"
        assert channel.unit == "mV"
        assert channel.channel_type == "physiological"

    def test_channel_info_with_location(self):
        """Test ChannelInfo with optional location."""
        channel = ChannelInfo(
            name="ECG Lead I",
            unit="mV",
            location="chest",
        )
        assert channel.location == "chest"


class TestTimestampInfo:
    """Tests for TimestampInfo data model."""

    def test_timestamp_info_creation(self):
        """Test valid TimestampInfo creation with monotonic timestamps."""
        timestamps = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        ts_info = TimestampInfo(timestamps=timestamps)
        assert len(ts_info.timestamps) == 5
        assert ts_info.lsl_alignment_start == 0.0
        assert ts_info.zero_reference == 0.0

    def test_timestamp_info_non_monotonic_fails(self):
        """Test that non-monotonic timestamps raise error."""
        timestamps = np.array([0.0, 0.01, 0.015, 0.012, 0.02])  # Not strictly increasing
        with pytest.raises(ValueError, match="must be strictly increasing"):
            TimestampInfo(timestamps=timestamps)

    def test_timestamp_info_with_gaps(self):
        """Test TimestampInfo with gap detection."""
        timestamps = np.array([0.0, 0.01, 0.02])
        gaps = [{"start_idx": 10, "end_idx": 15, "duration": 0.05}]
        ts_info = TimestampInfo(timestamps=timestamps, gaps=gaps)
        assert ts_info.gaps == gaps


class TestPeakData:
    """Tests for PeakData data model."""

    def test_peak_data_creation(self):
        """Test valid PeakData creation."""
        indices = np.array([10, 20, 30, 40])
        sources = np.array([0, 0, 1, 0])  # 0=auto, 1=manual
        peak_data = PeakData(indices=indices, sources=sources)
        assert len(peak_data.indices) == 4
        assert peak_data.num_peaks == 4
        assert peak_data.num_auto == 3
        assert peak_data.num_manual == 1

    def test_peak_data_length_mismatch_fails(self):
        """Test that mismatched indices and sources lengths raise error."""
        indices = np.array([10, 20, 30])
        sources = np.array([0, 0])  # Length mismatch
        with pytest.raises(ValueError, match="must have same length"):
            PeakData(indices=indices, sources=sources)

    def test_peak_data_empty(self):
        """Test empty PeakData creation."""
        peak_data = PeakData(indices=np.array([]), sources=np.array([]))
        assert peak_data.num_peaks == 0
        assert peak_data.num_auto == 0
        assert peak_data.num_manual == 0


class TestSignalData:
    """Tests for SignalData data model."""

    def test_signal_data_creation(self):
        """Test valid SignalData creation."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)
        signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )
        assert len(signal.samples) == 1000
        assert signal.sampling_rate == 100.0
        assert signal.signal_type == SignalType.ECG
        assert signal.duration == pytest.approx(10.0, rel=1e-2)
        assert signal.num_samples == 1000

    def test_signal_data_negative_sampling_rate_fails(self):
        """Test that negative sampling rate raises error."""
        samples = np.random.randn(100)
        timestamps = np.linspace(0, 1, 100)
        with pytest.raises(ValueError, match="must be positive"):
            SignalData(
                samples=samples,
                sampling_rate=-100.0,
                timestamps=timestamps,
                channel_name="ECG",
                signal_type=SignalType.ECG,
            )

    def test_signal_data_2d_array_fails(self):
        """Test that 2D samples array raises error."""
        samples = np.random.randn(100, 2)  # 2D array
        timestamps = np.linspace(0, 1, 100)
        with pytest.raises(ValueError, match="must be 1D"):
            SignalData(
                samples=samples,
                sampling_rate=100.0,
                timestamps=timestamps,
                channel_name="ECG",
                signal_type=SignalType.ECG,
            )

    def test_signal_data_non_monotonic_timestamps_fails(self):
        """Test that non-monotonic timestamps raise error."""
        samples = np.random.randn(100)
        timestamps = np.array([0.0, 0.01, 0.015, 0.012] + list(np.linspace(0.02, 1, 96)))
        with pytest.raises(ValueError, match="must be strictly increasing"):
            SignalData(
                samples=samples,
                sampling_rate=100.0,
                timestamps=timestamps,
                channel_name="ECG",
                signal_type=SignalType.ECG,
            )

    def test_signal_data_length_mismatch_fails(self):
        """Test that samples/timestamps length mismatch raises error."""
        samples = np.random.randn(100)
        timestamps = np.linspace(0, 1, 99)  # Length mismatch
        with pytest.raises(ValueError, match="must have same length"):
            SignalData(
                samples=samples,
                sampling_rate=100.0,
                timestamps=timestamps,
                channel_name="ECG",
                signal_type=SignalType.ECG,
            )

    def test_signal_data_mean_sampling_rate(self):
        """Test mean sampling rate calculation."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)
        signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG",
            signal_type=SignalType.ECG,
        )
        # Mean sampling rate should be approximately 100 Hz
        assert signal.mean_sampling_rate == pytest.approx(100.0, rel=0.01)


class TestProcessingStep:
    """Tests for ProcessingStep data model."""

    def test_processing_step_creation(self):
        """Test valid ProcessingStep creation."""
        step = ProcessingStep(
            operation="bandpass_filter",
            parameters={"lowcut": 0.5, "highcut": 40.0, "order": 4},
        )
        assert step.operation == "bandpass_filter"
        assert step.parameters["lowcut"] == 0.5
        assert step.parameters["order"] == 4


class TestRecordingSession:
    """Tests for RecordingSession data model."""

    def test_recording_session_creation(self):
        """Test valid RecordingSession creation."""
        samples1 = np.random.randn(1000)
        timestamps1 = np.linspace(0, 10, 1000)
        signal1 = SignalData(
            samples=samples1,
            sampling_rate=100.0,
            timestamps=timestamps1,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )

        samples2 = np.random.randn(500)
        timestamps2 = np.linspace(0, 10, 500)
        signal2 = SignalData(
            samples=samples2,
            sampling_rate=50.0,
            timestamps=timestamps2,
            channel_name="PPG",
            signal_type=SignalType.PPG,
        )

        session = RecordingSession(
            source_path=Path("test.xdf"),
            signals=[signal1, signal2],
            processing_history=[],
        )

        assert session.num_signals == 2
        assert session.source_path.name == "test.xdf"
        assert len(session.processing_history) == 0

    def test_recording_session_get_signal(self):
        """Test get_signal method retrieves by signal type."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)
        signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )

        session = RecordingSession(
            source_path=Path("test.xdf"),
            signals=[signal],
            processing_history=[],
        )

        # Get signal by type
        retrieved = session.get_signal(SignalType.ECG)
        assert retrieved is not None
        assert retrieved.signal_type == SignalType.ECG

        # Try to get non-existent type
        not_found = session.get_signal(SignalType.PPG)
        assert not_found is None

    def test_recording_session_has_signal_type(self):
        """Test has_signal_type method."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)
        signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )

        session = RecordingSession(
            source_path=Path("test.xdf"),
            signals=[signal],
            processing_history=[],
        )

        assert session.has_signal_type(SignalType.ECG) is True
        assert session.has_signal_type(SignalType.PPG) is False

    def test_recording_session_get_all_signals(self):
        """Test get_all_signals method with type filter."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)

        ecg_signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )

        ppg_signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="PPG",
            signal_type=SignalType.PPG,
        )

        session = RecordingSession(
            source_path=Path("test.xdf"),
            signals=[ecg_signal, ppg_signal],
            processing_history=[],
        )

        # Get only ECG signals
        ecg_signals = session.get_all_signals(SignalType.ECG)
        assert len(ecg_signals) == 1
        assert ecg_signals[0].signal_type == SignalType.ECG

        # Get only PPG signals
        ppg_signals = session.get_all_signals(SignalType.PPG)
        assert len(ppg_signals) == 1
        assert ppg_signals[0].signal_type == SignalType.PPG

    def test_recording_session_with_processing_history(self):
        """Test RecordingSession with processing history."""
        samples = np.random.randn(1000)
        timestamps = np.linspace(0, 10, 1000)
        signal = SignalData(
            samples=samples,
            sampling_rate=100.0,
            timestamps=timestamps,
            channel_name="ECG Lead I",
            signal_type=SignalType.ECG,
        )

        step1 = ProcessingStep(
            operation="bandpass_filter",
            parameters={"lowcut": 0.5, "highcut": 40.0},
        )
        step2 = ProcessingStep(
            operation="detect_peaks",
            parameters={"method": "neurokit"},
        )

        session = RecordingSession(
            source_path=Path("test.xdf"),
            signals=[signal],
            processing_history=[step1, step2],
        )

        assert len(session.processing_history) == 2
        assert session.processing_history[0].operation == "bandpass_filter"
        assert session.processing_history[1].operation == "detect_peaks"
