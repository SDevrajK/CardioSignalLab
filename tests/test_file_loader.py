"""Unit tests for file loaders."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardio_signal_lab.core import (
    CsvLoader,
    SignalType,
    XdfLoader,
    get_loader,
)


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file with signal data.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary CSV file
    """
    # Create sample ECG-like data
    timestamps = np.linspace(0, 10, 1000)
    ecg_samples = np.sin(2 * np.pi * 1.2 * timestamps) + 0.1 * np.random.randn(1000)
    ppg_samples = np.sin(2 * np.pi * 1.0 * timestamps) + 0.1 * np.random.randn(1000)

    df = pd.DataFrame({
        "time": timestamps,
        "ECG": ecg_samples,
        "PPG": ppg_samples,
    })

    csv_path = tmp_path / "test_signals.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def temp_xdf_file(tmp_path):
    """Create a temporary XDF file with signal data.

    Note: Creating a full XDF file is complex. This fixture returns a path
    that would be used for XDF testing. In production tests, use real XDF
    sample files from the test_data directory.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary XDF file (note: file may not actually exist for this fixture)
    """
    xdf_path = tmp_path / "test_signals.xdf"
    # In real tests, copy a sample XDF file here or use a fixture from test_data/
    return xdf_path


class TestGetLoader:
    """Tests for get_loader dispatcher function."""

    def test_get_loader_xdf(self):
        """Test that get_loader returns XdfLoader for .xdf files."""
        loader = get_loader(Path("test.xdf"))
        assert isinstance(loader, XdfLoader)

    def test_get_loader_csv(self):
        """Test that get_loader returns CsvLoader for .csv files."""
        loader = get_loader(Path("test.csv"))
        assert isinstance(loader, CsvLoader)

    def test_get_loader_unsupported_extension(self):
        """Test that get_loader raises error for unsupported file types."""
        with pytest.raises(ValueError, match="No loader available"):
            get_loader(Path("test.txt"))


class TestCsvLoader:
    """Tests for CsvLoader."""

    def test_csv_loader_can_load(self):
        """Test that CsvLoader can identify CSV files."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        assert loader.can_load(Path("test.csv")) is True
        assert loader.can_load(Path("test.CSV")) is True
        assert loader.can_load(Path("test.xdf")) is False

    def test_csv_loader_load_with_signal_type(self, temp_csv_file):
        """Test loading CSV file with forced signal type (auto-detect disabled)."""
        loader = CsvLoader(signal_type=SignalType.ECG, auto_detect_type=False)
        session = loader.load(temp_csv_file)

        assert session is not None
        assert session.num_signals == 2  # ECG and PPG columns
        assert session.source_path == temp_csv_file

        # All signals should have the specified type (ECG) since auto-detect is off
        for signal in session.signals:
            assert signal.signal_type == SignalType.ECG

    def test_csv_loader_default_unknown_type(self, temp_csv_file):
        """Test that CsvLoader defaults to UNKNOWN when auto-detect is disabled."""
        loader = CsvLoader(auto_detect_type=False)  # Disable auto-detection
        session = loader.load(temp_csv_file)

        # All signals should be UNKNOWN type since auto-detect is off
        for signal in session.signals:
            assert signal.signal_type == SignalType.UNKNOWN

    def test_csv_loader_auto_detects_signal_types(self, temp_csv_file):
        """Test that CsvLoader auto-detects signal types from column names."""
        loader = CsvLoader()  # auto_detect_type=True by default
        session = loader.load(temp_csv_file)

        # ECG column should be detected as ECG, PPG as PPG
        ecg_signals = [s for s in session.signals if s.signal_type == SignalType.ECG]
        ppg_signals = [s for s in session.signals if s.signal_type == SignalType.PPG]
        assert len(ecg_signals) == 1
        assert len(ppg_signals) == 1

    def test_csv_loader_calculates_sampling_rate(self, temp_csv_file):
        """Test that CsvLoader auto-calculates sampling rate from timestamps."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        session = loader.load(temp_csv_file)

        # Expected: 1000 samples over 10 seconds = ~100 Hz
        for signal in session.signals:
            assert signal.sampling_rate == pytest.approx(99.9, rel=0.01)

    def test_csv_loader_preserves_channel_names(self, temp_csv_file):
        """Test that CsvLoader preserves column names as channel names."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        session = loader.load(temp_csv_file)

        channel_names = [sig.channel_name for sig in session.signals]
        assert "ECG" in channel_names
        assert "PPG" in channel_names

    def test_csv_loader_timestamps_monotonic(self, temp_csv_file):
        """Test that loaded timestamps are strictly increasing."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        session = loader.load(temp_csv_file)

        for signal in session.signals:
            diffs = np.diff(signal.timestamps)
            assert np.all(diffs > 0), "Timestamps must be strictly increasing"

    def test_csv_loader_file_not_found(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent_file.csv"))

    def test_csv_loader_empty_file(self, tmp_path):
        """Test that loading empty CSV raises ValueError."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        loader = CsvLoader(signal_type=SignalType.ECG)
        with pytest.raises(ValueError):
            loader.load(empty_csv)

    def test_csv_loader_skips_empty_columns(self, tmp_path):
        """Test that CsvLoader skips empty (all-zero or all-NaN) columns."""
        timestamps = np.linspace(0, 10, 100)
        valid_signal = np.sin(2 * np.pi * timestamps)
        empty_zeros = np.zeros(100)
        empty_nans = np.full(100, np.nan)

        df = pd.DataFrame({
            "time": timestamps,
            "valid": valid_signal,
            "empty_zeros": empty_zeros,
            "empty_nans": empty_nans,
        })

        csv_path = tmp_path / "test_empty_cols.csv"
        df.to_csv(csv_path, index=False)

        loader = CsvLoader(signal_type=SignalType.ECG)
        session = loader.load(csv_path)

        # Should only load the "valid" column
        assert session.num_signals == 1
        assert session.signals[0].channel_name == "valid"


class TestXdfLoader:
    """Tests for XdfLoader.

    Note: These tests require actual XDF files or mocking pyxdf.
    For comprehensive XDF testing, use real sample files.
    """

    def test_xdf_loader_can_load(self):
        """Test that XdfLoader can identify XDF files."""
        loader = XdfLoader()
        assert loader.can_load(Path("test.xdf")) is True
        assert loader.can_load(Path("test.XDF")) is True
        assert loader.can_load(Path("test.csv")) is False

    def test_xdf_loader_file_not_found(self):
        """Test that loading non-existent XDF file raises FileNotFoundError."""
        loader = XdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent_file.xdf"))

    # Additional XDF tests would require:
    # 1. Sample XDF files in tests/test_data/
    # 2. Mocking pyxdf.load_xdf() for controlled testing
    # 3. Testing signal type detection from stream metadata
    # 4. Testing device vs LSL timestamp extraction
    # 5. Testing selective stream loading

    @pytest.mark.skip(reason="Requires sample XDF file")
    def test_xdf_loader_load_real_file(self):
        """Test loading actual XDF file.

        This test is skipped by default. To run it:
        1. Place a sample XDF file in tests/test_data/sample.xdf
        2. Remove the @pytest.mark.skip decorator
        """
        loader = XdfLoader()
        session = loader.load(Path("tests/test_data/sample.xdf"))

        assert session is not None
        assert session.num_signals > 0
        assert session.source_path.name == "sample.xdf"

        # Verify signal type detection worked
        for signal in session.signals:
            assert signal.signal_type in [SignalType.ECG, SignalType.PPG, SignalType.EDA, SignalType.UNKNOWN]


class TestFileLoaderProtocol:
    """Tests for FileLoader Protocol compliance."""

    def test_csv_loader_implements_protocol(self):
        """Test that CsvLoader implements can_load and load methods."""
        loader = CsvLoader(signal_type=SignalType.ECG)
        assert hasattr(loader, "can_load")
        assert hasattr(loader, "load")
        assert callable(loader.can_load)
        assert callable(loader.load)

    def test_xdf_loader_implements_protocol(self):
        """Test that XdfLoader implements can_load and load methods."""
        loader = XdfLoader()
        assert hasattr(loader, "can_load")
        assert hasattr(loader, "load")
        assert callable(loader.can_load)
        assert callable(loader.load)
