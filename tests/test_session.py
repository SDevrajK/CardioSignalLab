"""Tests for session save/load functionality."""
import json
from pathlib import Path

import numpy as np
import pytest

from cardio_signal_lab.core.data_models import PeakClassification, PeakData
from cardio_signal_lab.core.session import load_session, save_session


@pytest.fixture
def simple_peaks():
    """Create simple peak data."""
    return PeakData(
        indices=np.array([100, 200, 300], dtype=int),
        classifications=np.array([
            PeakClassification.AUTO.value,
            PeakClassification.MANUAL.value,
            PeakClassification.ECTOPIC.value,
        ], dtype=int),
    )


@pytest.fixture
def pipeline_data():
    """Create sample pipeline data."""
    return {
        "steps": [
            {
                "operation": "bandpass",
                "parameters": {"lowcut": 0.5, "highcut": 40.0},
                "timestamp": "2024-01-01T00:00:00",
            }
        ]
    }


class TestSaveSession:
    def test_save_session_basic(self, tmp_path):
        """Test basic session save."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()  # Create dummy file

        output = tmp_path / "session.csl.json"
        pipeline = {"steps": []}

        result = save_session(
            source_file=source_file,
            pipeline_data=pipeline,
            peaks=None,
            output_path=output,
        )

        assert result == output
        assert output.exists()

        # Verify contents
        with open(output) as f:
            data = json.load(f)

        assert "source_file" in data
        assert "processing_pipeline" in data
        assert "peaks" in data
        assert "view_state" in data
        assert data["peaks"] is None
        assert data["view_state"] == {}

    def test_save_session_with_peaks(self, simple_peaks, tmp_path):
        """Test session save with peaks."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        output = tmp_path / "session.csl.json"
        pipeline = {"steps": []}

        result = save_session(
            source_file=source_file,
            pipeline_data=pipeline,
            peaks=simple_peaks,
            output_path=output,
        )

        assert result == output

        with open(output) as f:
            data = json.load(f)

        assert data["peaks"] is not None
        assert data["peaks"]["indices"] == [100, 200, 300]
        assert data["peaks"]["classifications"] == [
            PeakClassification.AUTO.value,
            PeakClassification.MANUAL.value,
            PeakClassification.ECTOPIC.value,
        ]

    def test_save_session_with_pipeline(self, pipeline_data, tmp_path):
        """Test session save with processing pipeline."""
        source_file = tmp_path / "test.csv"
        source_file.touch()

        output = tmp_path / "session.csl.json"

        result = save_session(
            source_file=source_file,
            pipeline_data=pipeline_data,
            peaks=None,
            output_path=output,
        )

        assert result == output

        with open(output) as f:
            data = json.load(f)

        assert data["processing_pipeline"] == pipeline_data
        assert len(data["processing_pipeline"]["steps"]) == 1

    def test_save_session_with_view_state(self, tmp_path):
        """Test session save with view state."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        output = tmp_path / "session.csl.json"
        view_state = {
            "x_range": [0.0, 10.0],
            "y_range": [-1.0, 1.0],
            "signal_type": "ecg",
            "channel_name": "ECG-Ch1",
        }

        result = save_session(
            source_file=source_file,
            pipeline_data={"steps": []},
            peaks=None,
            output_path=output,
            view_state=view_state,
        )

        assert result == output

        with open(output) as f:
            data = json.load(f)

        assert data["view_state"] == view_state

    def test_save_session_empty_peaks(self, tmp_path):
        """Test session save with empty PeakData."""
        empty_peaks = PeakData(
            indices=np.array([], dtype=int),
            classifications=np.array([], dtype=int),
        )

        source_file = tmp_path / "test.xdf"
        source_file.touch()
        output = tmp_path / "session.csl.json"

        result = save_session(
            source_file=source_file,
            pipeline_data={"steps": []},
            peaks=empty_peaks,
            output_path=output,
        )

        assert result == output

        with open(output) as f:
            data = json.load(f)

        # Empty peaks should result in None (num_peaks == 0 check)
        assert data["peaks"] is None

    def test_save_session_absolute_path(self, tmp_path):
        """Test that source_file is saved as absolute path."""
        source_file = tmp_path / "subdir" / "test.xdf"
        source_file.parent.mkdir(parents=True)
        source_file.touch()

        output = tmp_path / "session.csl.json"

        save_session(
            source_file=source_file,
            pipeline_data={"steps": []},
            peaks=None,
            output_path=output,
        )

        with open(output) as f:
            data = json.load(f)

        # Should be absolute path
        saved_path = Path(data["source_file"])
        assert saved_path.is_absolute()
        assert saved_path == source_file.absolute()


class TestLoadSession:
    def test_load_session_basic(self, tmp_path):
        """Test basic session load."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        session_file = tmp_path / "session.csl.json"
        session_data = {
            "source_file": str(source_file.absolute()),
            "processing_pipeline": {"steps": []},
            "peaks": None,
            "view_state": {},
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f)

        result = load_session(session_file)

        assert result["source_file"] == str(source_file.absolute())
        assert result["processing_pipeline"] == {"steps": []}
        assert result["peaks"] is None
        assert result["view_state"] == {}

    def test_load_session_with_peaks(self, tmp_path):
        """Test loading session with peaks."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        session_file = tmp_path / "session.csl.json"
        session_data = {
            "source_file": str(source_file),
            "processing_pipeline": {"steps": []},
            "peaks": {
                "indices": [50, 100, 150],
                "classifications": [0, 1, 2],
            },
            "view_state": {},
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f)

        result = load_session(session_file)

        assert result["peaks"]["indices"] == [50, 100, 150]
        assert result["peaks"]["classifications"] == [0, 1, 2]

    def test_load_session_with_pipeline(self, tmp_path):
        """Test loading session with processing pipeline."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        session_file = tmp_path / "session.csl.json"
        pipeline = {
            "steps": [
                {
                    "operation": "bandpass",
                    "parameters": {"lowcut": 0.5, "highcut": 40.0},
                }
            ]
        }
        session_data = {
            "source_file": str(source_file),
            "processing_pipeline": pipeline,
            "peaks": None,
            "view_state": {},
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f)

        result = load_session(session_file)

        assert result["processing_pipeline"] == pipeline
        assert len(result["processing_pipeline"]["steps"]) == 1


class TestSaveLoadRoundTrip:
    def test_round_trip_full(self, simple_peaks, pipeline_data, tmp_path):
        """Test save/load round-trip with all data."""
        source_file = tmp_path / "test.xdf"
        source_file.touch()

        session_file = tmp_path / "session.csl.json"
        view_state = {"x_range": [0, 10], "y_range": [-1, 1]}

        # Save
        save_session(
            source_file=source_file,
            pipeline_data=pipeline_data,
            peaks=simple_peaks,
            output_path=session_file,
            view_state=view_state,
        )

        # Load
        loaded = load_session(session_file)

        # Verify all fields
        assert Path(loaded["source_file"]) == source_file.absolute()
        assert loaded["processing_pipeline"] == pipeline_data
        assert loaded["peaks"]["indices"] == [100, 200, 300]
        assert loaded["peaks"]["classifications"] == [
            PeakClassification.AUTO.value,
            PeakClassification.MANUAL.value,
            PeakClassification.ECTOPIC.value,
        ]
        assert loaded["view_state"] == view_state

    def test_round_trip_minimal(self, tmp_path):
        """Test save/load round-trip with minimal data."""
        source_file = tmp_path / "test.csv"
        source_file.touch()

        session_file = tmp_path / "session.csl.json"

        # Save
        save_session(
            source_file=source_file,
            pipeline_data={"steps": []},
            peaks=None,
            output_path=session_file,
        )

        # Load
        loaded = load_session(session_file)

        assert Path(loaded["source_file"]) == source_file.absolute()
        assert loaded["processing_pipeline"] == {"steps": []}
        assert loaded["peaks"] is None
        assert loaded["view_state"] == {}
