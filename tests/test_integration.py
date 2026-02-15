"""Integration tests for end-to-end workflows."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardio_signal_lab.core import (
    CsvLoader,
    SignalType,
    export_csv,
    export_npy,
    export_annotations,
    save_session,
    load_session,
    PeakData,
    PeakClassification,
)
from cardio_signal_lab.processing import ProcessingPipeline
from cardio_signal_lab.processing.peak_detection import detect_ecg_peaks


@pytest.fixture
def sample_ecg_csv(tmp_path):
    """Create a sample ECG CSV file for testing."""
    # Use NeuroKit2 to generate a realistic ECG signal
    import neurokit2 as nk

    fs = 250  # 250 Hz
    duration = 10  # 10 seconds

    # Generate realistic ECG using NeuroKit2
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=70)
    t = np.arange(len(ecg)) / fs

    df = pd.DataFrame({
        "time": t,
        "ECG": ecg,
    })

    csv_path = tmp_path / "sample_ecg.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_ppg_csv(tmp_path):
    """Create a sample PPG CSV file for testing."""
    fs = 100  # 100 Hz
    duration = 10
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples)

    # Simple PPG-like waveform
    heart_rate = 70 / 60  # 70 bpm
    ppg = 1 + 0.1 * np.sin(2 * np.pi * heart_rate * t)
    ppg += 0.02 * np.random.randn(n_samples)

    df = pd.DataFrame({
        "time": t,
        "PPG": ppg,
    })

    csv_path = tmp_path / "sample_ppg.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestFileLoadingWorkflow:
    """Test complete file loading workflows."""

    def test_load_csv_auto_detect(self, sample_ecg_csv):
        """Test loading CSV with automatic signal type detection."""
        loader = CsvLoader(signal_type=SignalType.UNKNOWN, auto_detect_type=True)
        session = loader.load(sample_ecg_csv)

        assert session is not None
        assert session.num_signals == 1
        assert session.signals[0].signal_type == SignalType.ECG
        assert session.signals[0].sampling_rate > 0
        assert session.signals[0].num_samples > 0

    def test_load_csv_manual_type(self, sample_ppg_csv):
        """Test loading CSV with manually specified signal type."""
        loader = CsvLoader(signal_type=SignalType.PPG, auto_detect_type=False)
        session = loader.load(sample_ppg_csv)

        assert session is not None
        assert session.num_signals == 1
        assert session.signals[0].signal_type == SignalType.PPG

    def test_load_validates_sampling_rate(self, tmp_path):
        """Test that file loading warns about unusual sampling rates."""
        # Create CSV with very high sampling rate (unusual but not invalid)
        t = np.linspace(0, 1, 10000)  # 10000 Hz
        df = pd.DataFrame({"time": t, "ECG": np.sin(t)})
        csv_path = tmp_path / "high_fs.csv"
        df.to_csv(csv_path, index=False)

        loader = CsvLoader(signal_type=SignalType.ECG)
        # Should load successfully but log a warning
        session = loader.load(csv_path)
        assert session is not None
        # Sampling rate should be detected as ~10000 Hz
        assert session.signals[0].sampling_rate > 5000


class TestProcessingWorkflow:
    """Test complete signal processing workflows."""

    def test_bandpass_filter_workflow(self, sample_ecg_csv):
        """Test loading signal, applying bandpass filter."""
        # Load signal
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        # Create processing pipeline
        pipeline = ProcessingPipeline()
        original_samples = signal.samples.copy()

        # Apply bandpass filter
        pipeline.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 4})
        filtered = pipeline.apply(original_samples, signal.sampling_rate)

        assert filtered.shape == original_samples.shape
        assert not np.array_equal(filtered, original_samples)  # Should be different
        assert pipeline.num_steps == 1

    def test_multi_step_processing_workflow(self, sample_ecg_csv):
        """Test applying multiple processing steps in sequence."""
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        pipeline = ProcessingPipeline()
        original = signal.samples.copy()

        # Multi-step processing
        pipeline.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 4})
        pipeline.add_step("zero_reference", {"method": "mean"})

        processed = pipeline.apply(original, signal.sampling_rate)

        assert processed.shape == original.shape
        assert pipeline.num_steps == 2
        # After zero-referencing, mean should be close to 0
        assert abs(np.mean(processed)) < 0.1

    def test_peak_detection_workflow(self, sample_ecg_csv):
        """Test loading signal, filtering, and detecting peaks."""
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        # Apply light filtering
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 2})
        filtered = pipeline.apply(signal.samples, signal.sampling_rate)

        # Detect peaks
        peak_indices = detect_ecg_peaks(filtered, signal.sampling_rate)

        assert len(peak_indices) > 0  # Should detect some peaks
        # For 10 seconds at ~60 bpm, expect ~10 peaks
        assert 5 <= len(peak_indices) <= 15


class TestExportWorkflow:
    """Test complete export workflows."""

    def test_csv_export_workflow(self, sample_ecg_csv, tmp_path):
        """Test loading, processing, and exporting to CSV."""
        # Load and process
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 4})
        processed = pipeline.apply(signal.samples, signal.sampling_rate)

        # Update signal with processed samples
        from cardio_signal_lab.core.data_models import SignalData
        processed_signal = SignalData(
            samples=processed,
            timestamps=signal.timestamps,
            sampling_rate=signal.sampling_rate,
            signal_type=signal.signal_type,
            channel_name=signal.channel_name,
        )

        # Detect peaks
        peak_indices = detect_ecg_peaks(processed, signal.sampling_rate)
        peaks = PeakData(
            indices=peak_indices.astype(int),
            classifications=np.full(len(peak_indices), PeakClassification.AUTO.value, dtype=int),
        )

        # Export to CSV
        output_path = tmp_path / "exported_ecg.csv"
        export_csv(processed_signal, peaks, output_path, include_peaks=True)

        assert output_path.exists()

        # Verify exported CSV
        df = pd.read_csv(output_path)
        assert "time_s" in df.columns
        assert "signal" in df.columns
        assert "peak" in df.columns
        assert "peak_classification" in df.columns
        assert len(df) == len(processed)
        assert df["peak"].sum() == len(peak_indices)  # Number of peaks marked as 1

    def test_npy_export_workflow(self, sample_ecg_csv, tmp_path):
        """Test exporting to NumPy arrays."""
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        peak_indices = detect_ecg_peaks(signal.samples, signal.sampling_rate)
        peaks = PeakData(
            indices=peak_indices.astype(int),
            classifications=np.full(len(peak_indices), PeakClassification.AUTO.value, dtype=int),
        )

        # Export
        base_path = tmp_path / "exported"
        export_npy(signal, peaks, base_path)

        # Check files exist
        assert (tmp_path / "exported_signal.npy").exists()
        assert (tmp_path / "exported_peaks.npy").exists()
        assert (tmp_path / "exported_classifications.npy").exists()

        # Verify contents
        signal_data = np.load(tmp_path / "exported_signal.npy")
        loaded_peaks = np.load(tmp_path / "exported_peaks.npy")
        loaded_classifications = np.load(tmp_path / "exported_classifications.npy")

        # Signal file contains [timestamps, samples] in column stack
        assert signal_data.shape[1] == 2
        assert np.array_equal(signal_data[:, 0], signal.timestamps)
        assert np.array_equal(signal_data[:, 1], signal.samples)
        assert np.array_equal(loaded_peaks, peak_indices)
        assert np.array_equal(loaded_classifications, peaks.classifications)

    def test_annotations_export_workflow(self, sample_ecg_csv, tmp_path):
        """Test exporting annotations only."""
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        peak_indices = detect_ecg_peaks(signal.samples, signal.sampling_rate)
        peaks = PeakData(
            indices=peak_indices.astype(int),
            classifications=np.full(len(peak_indices), PeakClassification.AUTO.value, dtype=int),
        )

        # Export annotations
        output_path = tmp_path / "annotations.csv"
        export_annotations(signal, peaks, output_path)

        assert output_path.exists()

        # Verify annotations CSV
        df = pd.read_csv(output_path)
        assert "peak_index" in df.columns
        assert "time_s" in df.columns
        assert "amplitude" in df.columns
        assert "classification" in df.columns
        assert len(df) == len(peak_indices)
        assert list(df["peak_index"]) == list(peak_indices)


class TestSessionWorkflow:
    """Test complete session save/load workflows."""

    def test_session_save_load_workflow(self, sample_ecg_csv, tmp_path):
        """Test saving and loading session with processing pipeline."""
        # Load and process
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        # Create pipeline
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 4})
        pipeline.add_step("zero_reference", {"method": "mean"})

        # Detect peaks
        processed = pipeline.apply(signal.samples, signal.sampling_rate)
        peak_indices = detect_ecg_peaks(processed, signal.sampling_rate)
        peaks = PeakData(
            indices=peak_indices.astype(int),
            classifications=np.full(len(peak_indices), PeakClassification.AUTO.value, dtype=int),
        )

        # Save session
        session_path = tmp_path / "test_session.csl.json"
        pipeline_data = pipeline.serialize()
        save_session(
            source_file=sample_ecg_csv,
            pipeline_data=pipeline_data,
            peaks=peaks,
            output_path=session_path,
        )

        assert session_path.exists()

        # Load session
        loaded_session = load_session(session_path)

        assert loaded_session["source_file"] == str(sample_ecg_csv)
        assert "processing_pipeline" in loaded_session
        assert len(loaded_session["processing_pipeline"]["steps"]) == 2
        assert loaded_session["peaks"]["indices"] == peaks.indices.tolist()
        assert loaded_session["peaks"]["classifications"] == peaks.classifications.tolist()

    def test_session_pipeline_reproducibility(self, sample_ecg_csv, tmp_path):
        """Test that loaded pipeline produces same results."""
        # Original processing
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(sample_ecg_csv)
        signal = session.signals[0]

        pipeline1 = ProcessingPipeline()
        pipeline1.add_step("bandpass", {"lowcut": 0.5, "highcut": 40.0, "order": 4})
        result1 = pipeline1.apply(signal.samples, signal.sampling_rate)

        # Save and reload pipeline
        session_path = tmp_path / "pipeline_test.csl.json"
        save_session(
            source_file=sample_ecg_csv,
            pipeline_data=pipeline1.serialize(),
            peaks=None,
            output_path=session_path,
        )

        loaded_data = load_session(session_path)

        # Reconstruct pipeline from saved data
        pipeline2 = ProcessingPipeline()
        for step in loaded_data["processing_pipeline"]["steps"]:
            pipeline2.add_step(step["operation"], step["parameters"])

        result2 = pipeline2.apply(signal.samples, signal.sampling_rate)

        # Results should be identical
        assert np.allclose(result1, result2, rtol=1e-10)
