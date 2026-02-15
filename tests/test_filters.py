"""Tests for signal filtering and preprocessing operations."""
import numpy as np
import pytest

from cardio_signal_lab.processing.filters import (
    bandpass_filter,
    baseline_correction,
    highpass_filter,
    lowpass_filter,
    notch_filter,
    segment_signal,
    zero_reference,
)


@pytest.fixture
def sampling_rate():
    return 1000.0


@pytest.fixture
def composite_signal(sampling_rate):
    """Signal with 5 Hz cardiac + 0.1 Hz baseline + 60 Hz noise."""
    t = np.linspace(0, 2, int(2 * sampling_rate))
    cardiac = np.sin(2 * np.pi * 5 * t)
    baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    noise = 0.3 * np.sin(2 * np.pi * 60 * t)
    return cardiac + baseline + noise, t


class TestBandpassFilter:
    def test_preserves_cardiac(self, composite_signal, sampling_rate):
        signal, t = composite_signal
        cardiac = np.sin(2 * np.pi * 5 * t)
        filtered = bandpass_filter(signal, sampling_rate, lowcut=1.0, highcut=20.0)
        # Cardiac component should be preserved (correlation > 0.9)
        corr = np.corrcoef(cardiac, filtered)[0, 1]
        assert corr > 0.9

    def test_removes_baseline(self, composite_signal, sampling_rate):
        signal, t = composite_signal
        filtered = bandpass_filter(signal, sampling_rate, lowcut=1.0, highcut=20.0)
        # Baseline (0.1 Hz) should be attenuated
        baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
        corr_baseline = np.corrcoef(baseline, filtered)[0, 1]
        assert abs(corr_baseline) < 0.3

    def test_removes_noise(self, composite_signal, sampling_rate):
        signal, t = composite_signal
        filtered = bandpass_filter(signal, sampling_rate, lowcut=1.0, highcut=20.0)
        # 60 Hz noise should be attenuated
        noise = 0.3 * np.sin(2 * np.pi * 60 * t)
        corr_noise = np.corrcoef(noise, filtered)[0, 1]
        assert abs(corr_noise) < 0.3

    def test_invalid_lowcut(self, sampling_rate):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError, match="lowcut must be positive"):
            bandpass_filter(signal, sampling_rate, lowcut=-1.0)

    def test_lowcut_exceeds_highcut(self, sampling_rate):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError, match="lowcut.*must be < highcut"):
            bandpass_filter(signal, sampling_rate, lowcut=50.0, highcut=10.0)

    def test_highcut_clamps_to_nyquist(self, sampling_rate):
        signal = np.random.randn(1000)
        # Should not raise, just clamp
        filtered = bandpass_filter(signal, sampling_rate, lowcut=1.0, highcut=600.0)
        assert len(filtered) == len(signal)

    def test_output_shape(self, sampling_rate):
        signal = np.random.randn(2000)
        filtered = bandpass_filter(signal, sampling_rate, lowcut=1.0, highcut=40.0)
        assert filtered.shape == signal.shape


class TestHighpassFilter:
    def test_removes_dc(self, sampling_rate):
        signal = np.ones(1000) * 5.0 + np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        filtered = highpass_filter(signal, sampling_rate, cutoff=1.0)
        # DC offset should be removed
        assert abs(np.mean(filtered)) < 0.5

    def test_invalid_cutoff(self, sampling_rate):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError):
            highpass_filter(signal, sampling_rate, cutoff=-1.0)


class TestLowpassFilter:
    def test_removes_high_freq(self, sampling_rate):
        t = np.linspace(0, 1, 1000)
        low = np.sin(2 * np.pi * 5 * t)
        high = np.sin(2 * np.pi * 100 * t)
        signal = low + high
        filtered = lowpass_filter(signal, sampling_rate, cutoff=20.0)
        # Low freq should be preserved
        corr = np.corrcoef(low, filtered)[0, 1]
        assert corr > 0.9

    def test_clamps_to_nyquist(self, sampling_rate):
        signal = np.random.randn(1000)
        filtered = lowpass_filter(signal, sampling_rate, cutoff=600.0)
        assert len(filtered) == len(signal)


class TestNotchFilter:
    def test_removes_powerline(self, sampling_rate):
        t = np.linspace(0, 1, 1000)
        cardiac = np.sin(2 * np.pi * 5 * t)
        powerline = 0.5 * np.sin(2 * np.pi * 60 * t)
        signal = cardiac + powerline
        filtered = notch_filter(signal, sampling_rate, freq=60.0)
        # Cardiac should be preserved
        corr_cardiac = np.corrcoef(cardiac, filtered)[0, 1]
        assert corr_cardiac > 0.9
        # 60 Hz should be attenuated - check energy in 60 Hz band
        from scipy.signal import welch
        f, psd_orig = welch(signal, fs=sampling_rate)
        f, psd_filt = welch(filtered, fs=sampling_rate)
        idx_60 = np.argmin(np.abs(f - 60))
        assert psd_filt[idx_60] < psd_orig[idx_60] * 0.1

    def test_skips_above_nyquist(self, sampling_rate):
        signal = np.random.randn(1000)
        filtered = notch_filter(signal, sampling_rate, freq=600.0)
        np.testing.assert_array_equal(filtered, signal)


class TestBaselineCorrection:
    def test_removes_linear_drift(self, sampling_rate):
        n = 1000
        t = np.linspace(0, 1, n)
        drift = 5.0 * t  # Linear upward drift
        cardiac = np.sin(2 * np.pi * 10 * t)
        signal = cardiac + drift
        corrected = baseline_correction(signal, sampling_rate, poly_order=1)
        # Drift should be removed, signal mean should be near 0
        assert abs(np.mean(corrected)) < 0.5

    def test_preserves_signal_shape(self, sampling_rate):
        n = 1000
        t = np.linspace(0, 1, n)
        signal = np.sin(2 * np.pi * 10 * t)
        corrected = baseline_correction(signal, sampling_rate, poly_order=3)
        assert corrected.shape == signal.shape


class TestZeroReference:
    def test_mean_method(self, sampling_rate):
        signal = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        result = zero_reference(signal, sampling_rate, method="mean")
        assert abs(np.mean(result)) < 1e-10

    def test_first_n_method(self, sampling_rate):
        signal = np.array([10.0, 12.0, 8.0, 100.0, 200.0])
        result = zero_reference(signal, sampling_rate, method="first_n", n_samples=3)
        # Mean of first 3 is 10.0
        expected_offset = 10.0
        np.testing.assert_allclose(result, signal - expected_offset)

    def test_invalid_method(self, sampling_rate):
        signal = np.random.randn(10)
        with pytest.raises(ValueError, match="Unknown method"):
            zero_reference(signal, sampling_rate, method="invalid")


class TestSegmentSignal:
    def test_basic_segment(self, sampling_rate):
        signal = np.arange(2000, dtype=float)
        result = segment_signal(signal, sampling_rate, start_time=0.5, end_time=1.0)
        expected_len = int(0.5 * sampling_rate)
        assert len(result) == expected_len
        assert result[0] == 500.0

    def test_segment_to_end(self, sampling_rate):
        signal = np.arange(2000, dtype=float)
        result = segment_signal(signal, sampling_rate, start_time=1.0)
        assert len(result) == 1000
        assert result[0] == 1000.0

    def test_segment_clamps_bounds(self, sampling_rate):
        signal = np.arange(1000, dtype=float)
        result = segment_signal(signal, sampling_rate, start_time=-1.0, end_time=100.0)
        assert len(result) == 1000
