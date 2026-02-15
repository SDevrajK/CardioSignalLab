"""Tests for EEMD artifact removal.

Note: EEMD is computationally expensive. These tests use short signals
and small ensemble sizes for speed.
"""
import numpy as np
import pytest

from cardio_signal_lab.processing.eemd import (
    analyze_imf_characteristics,
    auto_select_artifact_imfs,
    eemd_decompose,
    reconstruct_from_imfs,
)


@pytest.fixture
def simple_signal():
    """Short signal with low + high frequency components."""
    np.random.seed(42)
    t = np.linspace(0, 2, 256)  # 128 Hz for 2 seconds
    low_freq = np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz cardiac-like
    high_freq = 0.3 * np.sin(2 * np.pi * 30 * t)  # 30 Hz noise
    return low_freq + high_freq, 128.0


class TestEEMDDecompose:
    def test_basic_decomposition(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(
            signal, ensemble_size=50, noise_width=0.2, random_seed=42
        )
        assert imfs.ndim == 2
        assert imfs.shape[1] == len(signal)
        assert residue.shape == (len(signal),)
        assert imfs.shape[0] >= 2  # Should have at least 2 IMFs

    def test_perfect_reconstruction(self, simple_signal):
        """Sum of all IMFs + residue should approximately equal original."""
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(
            signal, ensemble_size=50, noise_width=0.2, random_seed=42
        )
        reconstructed = np.sum(imfs, axis=0) + residue
        # EEMD is not perfectly additive due to noise, but should be close
        corr = np.corrcoef(signal, reconstructed)[0, 1]
        assert corr > 0.95

    def test_signal_too_short(self):
        signal = np.random.randn(50)
        with pytest.raises(ValueError, match="too short"):
            eemd_decompose(signal, ensemble_size=10)

    def test_reproducibility(self, simple_signal):
        signal, sr = simple_signal
        imfs1, res1 = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        imfs2, res2 = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        np.testing.assert_allclose(imfs1, imfs2, atol=1e-10)
        np.testing.assert_allclose(res1, res2, atol=1e-10)


class TestAnalyzeIMFCharacteristics:
    def test_basic_analysis(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        chars = analyze_imf_characteristics(imfs, sr)
        assert len(chars) == imfs.shape[0]
        for c in chars:
            assert "imf_index" in c
            assert "peak_freq" in c
            assert "energy" in c
            assert "energy_pct" in c
            assert "mean_period" in c
            assert c["energy"] >= 0
            assert c["peak_freq"] >= 0

    def test_energy_sums_to_100(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        chars = analyze_imf_characteristics(imfs, sr)
        total_pct = sum(c["energy_pct"] for c in chars)
        assert abs(total_pct - 100.0) < 1.0  # Allow small floating point error

    def test_frequency_ordering(self, simple_signal):
        """Earlier IMFs should generally have higher frequencies."""
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        chars = analyze_imf_characteristics(imfs, sr)
        freqs = [c["peak_freq"] for c in chars]
        # First IMF should have higher freq than last
        assert freqs[0] >= freqs[-1]


class TestAutoSelectArtifactIMFs:
    def test_excludes_high_frequency(self):
        chars = [
            {"imf_index": 0, "peak_freq": 30.0, "energy_pct": 5.0},
            {"imf_index": 1, "peak_freq": 1.5, "energy_pct": 80.0},
            {"imf_index": 2, "peak_freq": 0.3, "energy_pct": 15.0},
        ]
        excluded = auto_select_artifact_imfs(chars)
        assert 0 in excluded  # 30 Hz > 15 Hz threshold
        assert 1 not in excluded  # 1.5 Hz cardiac
        assert 2 not in excluded  # 0.3 Hz drift

    def test_excludes_low_energy_spikes(self):
        chars = [
            {"imf_index": 0, "peak_freq": 12.0, "energy_pct": 0.3},  # Spike
            {"imf_index": 1, "peak_freq": 1.5, "energy_pct": 95.0},
        ]
        excluded = auto_select_artifact_imfs(chars)
        assert 0 in excluded

    def test_excludes_artifact_range(self):
        chars = [
            {"imf_index": 0, "peak_freq": 6.0, "energy_pct": 5.0},  # Artifact range
            {"imf_index": 1, "peak_freq": 1.5, "energy_pct": 80.0},
        ]
        excluded = auto_select_artifact_imfs(chars)
        assert 0 in excluded

    def test_keeps_high_energy_artifact_range(self):
        chars = [
            {"imf_index": 0, "peak_freq": 6.0, "energy_pct": 20.0},  # High energy
            {"imf_index": 1, "peak_freq": 1.5, "energy_pct": 80.0},
        ]
        excluded = auto_select_artifact_imfs(chars)
        assert 0 not in excluded  # Energy > 15% threshold


class TestReconstructFromIMFs:
    def test_exclude_none(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        reconstructed = reconstruct_from_imfs(imfs, residue, exclude_imfs=None)
        full_sum = np.sum(imfs, axis=0) + residue
        np.testing.assert_allclose(reconstructed, full_sum)

    def test_exclude_some(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        reconstructed = reconstruct_from_imfs(imfs, residue, exclude_imfs=[0])
        expected = np.sum(imfs[1:], axis=0) + residue
        np.testing.assert_allclose(reconstructed, expected)

    def test_exclude_all(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        all_indices = list(range(imfs.shape[0]))
        reconstructed = reconstruct_from_imfs(imfs, residue, exclude_imfs=all_indices)
        # Only residue should remain
        np.testing.assert_allclose(reconstructed, residue)

    def test_without_residue(self, simple_signal):
        signal, sr = simple_signal
        imfs, residue = eemd_decompose(signal, ensemble_size=50, random_seed=42)
        reconstructed = reconstruct_from_imfs(
            imfs, residue, exclude_imfs=None, include_residue=False
        )
        expected = np.sum(imfs, axis=0)
        np.testing.assert_allclose(reconstructed, expected)
