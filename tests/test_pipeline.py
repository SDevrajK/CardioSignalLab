"""Tests for the composable processing pipeline."""
import numpy as np
import pytest

from cardio_signal_lab.processing.pipeline import (
    ProcessingPipeline,
    get_operation,
    list_operations,
    register_operation,
)


@pytest.fixture
def sine_signal():
    """Create a 5 Hz sine wave at 1000 Hz sampling rate."""
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 5 * t), 1000.0


class TestProcessingPipeline:
    def test_empty_pipeline(self, sine_signal):
        signal, sr = sine_signal
        pipeline = ProcessingPipeline()
        result = pipeline.apply(signal, sr)
        np.testing.assert_array_equal(result, signal)
        assert pipeline.is_empty
        assert pipeline.num_steps == 0

    def test_add_step(self):
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        assert pipeline.num_steps == 1
        assert not pipeline.is_empty

    def test_add_step_unknown_operation(self):
        pipeline = ProcessingPipeline()
        with pytest.raises(KeyError, match="Unknown operation"):
            pipeline.add_step("nonexistent_op")

    def test_apply_bandpass(self, sine_signal):
        signal, sr = sine_signal
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0, "order": 4})
        result = pipeline.apply(signal, sr)
        assert result.shape == signal.shape
        # 5 Hz signal should pass through 1-20 Hz bandpass
        assert np.corrcoef(signal, result)[0, 1] > 0.9

    def test_apply_multiple_steps(self, sine_signal):
        signal, sr = sine_signal
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        pipeline.add_step("zero_reference", {"method": "mean"})
        result = pipeline.apply(signal, sr)
        assert result.shape == signal.shape
        assert pipeline.num_steps == 2

    def test_reset(self, sine_signal):
        signal, sr = sine_signal
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        pipeline.add_step("zero_reference", {"method": "mean"})
        pipeline.reset()
        assert pipeline.is_empty
        assert pipeline.num_steps == 0
        # After reset, apply should return copy of input
        result = pipeline.apply(signal, sr)
        np.testing.assert_array_equal(result, signal)

    def test_remove_last(self):
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        pipeline.add_step("zero_reference", {"method": "mean"})
        step = pipeline.remove_last()
        assert step.operation == "zero_reference"
        assert pipeline.num_steps == 1

    def test_remove_last_empty(self):
        pipeline = ProcessingPipeline()
        assert pipeline.remove_last() is None

    def test_serialize_deserialize(self, sine_signal):
        signal, sr = sine_signal
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        pipeline.add_step("zero_reference", {"method": "mean"})

        data = pipeline.serialize()
        assert "steps" in data
        assert len(data["steps"]) == 2

        pipeline2 = ProcessingPipeline.deserialize(data)
        assert pipeline2.num_steps == 2

        result1 = pipeline.apply(signal, sr)
        result2 = pipeline2.apply(signal, sr)
        np.testing.assert_allclose(result1, result2)

    def test_serialize_empty(self):
        pipeline = ProcessingPipeline()
        data = pipeline.serialize()
        assert data == {"steps": []}
        pipeline2 = ProcessingPipeline.deserialize(data)
        assert pipeline2.is_empty

    def test_deserialize_unknown_operation(self):
        data = {"steps": [{"operation": "nonexistent", "parameters": {}}]}
        with pytest.raises(KeyError):
            ProcessingPipeline.deserialize(data)

    def test_repr(self):
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        assert "bandpass" in repr(pipeline)

    def test_apply_does_not_modify_input(self, sine_signal):
        signal, sr = sine_signal
        original = signal.copy()
        pipeline = ProcessingPipeline()
        pipeline.add_step("bandpass", {"lowcut": 1.0, "highcut": 20.0})
        pipeline.apply(signal, sr)
        np.testing.assert_array_equal(signal, original)


class TestOperationRegistry:
    def test_list_operations(self):
        ops = list_operations()
        assert "bandpass" in ops
        assert "highpass" in ops
        assert "notch" in ops
        assert "zero_reference" in ops

    def test_get_operation(self):
        func = get_operation("bandpass")
        assert callable(func)

    def test_get_unknown_operation(self):
        with pytest.raises(KeyError):
            get_operation("nonexistent_op")

    def test_register_custom_operation(self, sine_signal):
        signal, sr = sine_signal

        def my_op(samples, sampling_rate, **params):
            return samples * params.get("scale", 2.0)

        register_operation("test_scale", my_op)
        assert "test_scale" in list_operations()

        pipeline = ProcessingPipeline()
        pipeline.add_step("test_scale", {"scale": 3.0})
        result = pipeline.apply(signal, sr)
        np.testing.assert_allclose(result, signal * 3.0)
