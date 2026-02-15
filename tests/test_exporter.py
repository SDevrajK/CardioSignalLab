"""Tests for export functionality (CSV, NPY, annotations, processing parameters)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardio_signal_lab.core.data_models import (
    PeakClassification,
    PeakData,
    ProcessingStep,
    SignalData,
    SignalType,
)
from cardio_signal_lab.core.exporter import (
    export_annotations,
    export_csv,
    export_npy,
    save_processing_parameters,
)


@pytest.fixture
def simple_signal():
    """Create a simple test signal."""
    timestamps = np.linspace(0, 10, 1000)
    samples = np.sin(2 * np.pi * timestamps)
    return SignalData(
        channel_name="Test Channel",
        signal_type=SignalType.ECG,
        timestamps=timestamps,
        samples=samples,
        sampling_rate=100.0,
    )


@pytest.fixture
def simple_peaks():
    """Create simple peak data with mixed classifications."""
    return PeakData(
        indices=np.array([100, 200, 300, 400], dtype=int),
        classifications=np.array([
            PeakClassification.AUTO.value,
            PeakClassification.MANUAL.value,
            PeakClassification.ECTOPIC.value,
            PeakClassification.BAD.value,
        ], dtype=int),
    )


class TestExportCSV:
    def test_export_csv_with_peaks(self, simple_signal, simple_peaks, tmp_path):
        """Test CSV export with peaks included."""
        output = tmp_path / "test.csv"
        result = export_csv(simple_signal, simple_peaks, output, include_peaks=True)

        assert result == output
        assert output.exists()

        # Read and verify
        df = pd.read_csv(output)
        assert len(df) == len(simple_signal.samples)
        assert list(df.columns) == ["time_s", "signal", "peak", "peak_classification"]

        # Check peak markers
        assert df["peak"].sum() == simple_peaks.num_peaks
        assert df.loc[100, "peak"] == 1
        assert df.loc[200, "peak"] == 1

        # Check classifications
        assert df.loc[100, "peak_classification"] == PeakClassification.AUTO.value
        assert df.loc[200, "peak_classification"] == PeakClassification.MANUAL.value

    def test_export_csv_without_peaks(self, simple_signal, tmp_path):
        """Test CSV export without peaks."""
        output = tmp_path / "test.csv"
        result = export_csv(simple_signal, None, output, include_peaks=False)

        assert result == output
        df = pd.read_csv(output)
        assert list(df.columns) == ["time_s", "signal"]
        assert len(df) == len(simple_signal.samples)

    def test_export_csv_no_peaks_but_requested(self, simple_signal, tmp_path):
        """Test CSV export with include_peaks=True but no peaks provided."""
        output = tmp_path / "test.csv"
        result = export_csv(simple_signal, None, output, include_peaks=True)

        # Should still work, just no peak columns
        assert result == output
        df = pd.read_csv(output)
        assert "peak" not in df.columns

    def test_export_csv_empty_peaks(self, simple_signal, tmp_path):
        """Test CSV export with empty PeakData."""
        empty_peaks = PeakData(
            indices=np.array([], dtype=int),
            classifications=np.array([], dtype=int),
        )
        output = tmp_path / "test.csv"
        result = export_csv(simple_signal, empty_peaks, output, include_peaks=True)

        assert result == output
        df = pd.read_csv(output)
        assert "peak" not in df.columns or df["peak"].sum() == 0


class TestExportNPY:
    def test_export_npy_with_peaks(self, simple_signal, simple_peaks, tmp_path):
        """Test NPY export with peaks."""
        output = tmp_path / "test.npy"
        result = export_npy(simple_signal, simple_peaks, output)

        # Check signal file
        signal_file = tmp_path / "test_signal.npy"
        assert signal_file.exists()
        assert result == signal_file

        signal_data = np.load(signal_file)
        assert signal_data.shape == (len(simple_signal.samples), 2)
        np.testing.assert_array_equal(signal_data[:, 0], simple_signal.timestamps)
        np.testing.assert_array_equal(signal_data[:, 1], simple_signal.samples)

        # Check peaks file
        peaks_file = tmp_path / "test_peaks.npy"
        assert peaks_file.exists()
        peaks_data = np.load(peaks_file)
        np.testing.assert_array_equal(peaks_data, simple_peaks.indices)

        # Check classifications file
        class_file = tmp_path / "test_classifications.npy"
        assert class_file.exists()
        class_data = np.load(class_file)
        np.testing.assert_array_equal(class_data, simple_peaks.classifications)

    def test_export_npy_without_peaks(self, simple_signal, tmp_path):
        """Test NPY export without peaks."""
        output = tmp_path / "test.npy"
        result = export_npy(simple_signal, None, output)

        # Only signal file should exist
        signal_file = tmp_path / "test_signal.npy"
        assert signal_file.exists()
        assert result == signal_file

        peaks_file = tmp_path / "test_peaks.npy"
        class_file = tmp_path / "test_classifications.npy"
        assert not peaks_file.exists()
        assert not class_file.exists()

    def test_export_npy_empty_peaks(self, simple_signal, tmp_path):
        """Test NPY export with empty PeakData."""
        empty_peaks = PeakData(
            indices=np.array([], dtype=int),
            classifications=np.array([], dtype=int),
        )
        output = tmp_path / "test.npy"
        result = export_npy(simple_signal, empty_peaks, output)

        # Only signal file should exist (no peaks files for empty data)
        signal_file = tmp_path / "test_signal.npy"
        assert signal_file.exists()
        peaks_file = tmp_path / "test_peaks.npy"
        assert not peaks_file.exists()


class TestExportAnnotations:
    def test_export_annotations(self, simple_signal, simple_peaks, tmp_path):
        """Test annotation export."""
        output = tmp_path / "annotations.csv"
        result = export_annotations(simple_signal, simple_peaks, output)

        assert result == output
        assert output.exists()

        df = pd.read_csv(output)
        assert len(df) == simple_peaks.num_peaks
        assert list(df.columns) == ["peak_index", "time_s", "amplitude", "classification"]

        # Check values
        assert df["peak_index"].tolist() == [100, 200, 300, 400]
        np.testing.assert_array_almost_equal(
            df["time_s"].values,
            simple_signal.timestamps[simple_peaks.indices]
        )
        np.testing.assert_array_almost_equal(
            df["amplitude"].values,
            simple_signal.samples[simple_peaks.indices]
        )
        assert df["classification"].tolist() == [
            PeakClassification.AUTO.value,
            PeakClassification.MANUAL.value,
            PeakClassification.ECTOPIC.value,
            PeakClassification.BAD.value,
        ]

    def test_export_annotations_empty(self, simple_signal, tmp_path):
        """Test annotation export with no peaks."""
        empty_peaks = PeakData(
            indices=np.array([], dtype=int),
            classifications=np.array([], dtype=int),
        )
        output = tmp_path / "annotations.csv"
        result = export_annotations(simple_signal, empty_peaks, output)

        assert result == output
        assert output.exists()

        df = pd.read_csv(output)
        assert len(df) == 0
        assert list(df.columns) == ["peak_index", "time_s", "amplitude", "classification"]


class TestSaveProcessingParameters:
    def test_save_processing_parameters(self, tmp_path):
        """Test saving processing parameters to JSON."""
        steps = [
            ProcessingStep(
                operation="bandpass",
                parameters={"lowcut": 0.5, "highcut": 40.0, "order": 4},
                timestamp="2024-01-01T00:00:00",
            ),
            ProcessingStep(
                operation="baseline_correction",
                parameters={"poly_order": 3},
                timestamp="2024-01-01T00:01:00",
            ),
        ]

        output = tmp_path / "parameters.json"
        result = save_processing_parameters(
            pipeline_steps=steps,
            signal_type="ecg",
            sampling_rate=100.0,
            output_path=output,
        )

        assert result == output
        assert output.exists()

        # Read and verify
        with open(output) as f:
            data = json.load(f)

        assert data["signal_type"] == "ecg"
        assert data["sampling_rate"] == 100.0
        assert data["software"] == "CardioSignalLab MVP"
        assert len(data["processing_pipeline"]) == 2

        # Check first step
        step1 = data["processing_pipeline"][0]
        assert step1["operation"] == "bandpass"
        assert step1["parameters"] == {"lowcut": 0.5, "highcut": 40.0, "order": 4}
        assert step1["timestamp"] == "2024-01-01T00:00:00"

    def test_save_processing_parameters_empty(self, tmp_path):
        """Test saving with no processing steps."""
        output = tmp_path / "parameters.json"
        result = save_processing_parameters(
            pipeline_steps=[],
            signal_type="ppg",
            sampling_rate=64.0,
            output_path=output,
        )

        assert result == output
        with open(output) as f:
            data = json.load(f)

        assert data["processing_pipeline"] == []
        assert data["signal_type"] == "ppg"
