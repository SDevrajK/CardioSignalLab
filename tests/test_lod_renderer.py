"""Unit tests for LOD renderer."""
import numpy as np
import pytest

from cardio_signal_lab.gui.lod_renderer import LODRenderer


class TestLODRenderer:
    """Tests for LODRenderer initialization and pyramid building."""

    def test_lod_renderer_creation(self):
        """Test valid LODRenderer creation."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples, num_levels=5)

        assert renderer.num_samples == 1000
        assert renderer.duration == pytest.approx(10.0, rel=1e-6)
        assert renderer.num_levels == 5
        assert len(renderer.pyramids) > 0

    def test_lod_renderer_length_mismatch_fails(self):
        """Test that mismatched timestamps and samples raise error."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * np.linspace(0, 10, 999))  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            LODRenderer(timestamps, samples)

    def test_lod_renderer_non_monotonic_fails(self):
        """Test that non-monotonic timestamps raise error."""
        timestamps = np.array([0.0, 0.1, 0.15, 0.12, 0.2])  # Not strictly increasing
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError, match="must be strictly increasing"):
            LODRenderer(timestamps, samples)

    def test_lod_renderer_too_few_samples_fails(self):
        """Test that too few samples raise error."""
        timestamps = np.array([0.0])
        samples = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2 samples"):
            LODRenderer(timestamps, samples)

    def test_pyramid_structure(self):
        """Test that pyramid has correct structure."""
        timestamps = np.linspace(0, 10, 1024)  # Power of 2 for clean downsampling
        samples = np.random.randn(1024)

        renderer = LODRenderer(timestamps, samples, num_levels=5)

        # Check pyramid levels exist
        assert len(renderer.pyramids) >= 1

        # Level 0 should be full resolution
        t0, s_min0, s_max0 = renderer.pyramids[0]
        assert len(t0) == 1024
        np.testing.assert_array_equal(s_min0, samples)  # Full res: min==max==samples
        np.testing.assert_array_equal(s_max0, samples)

        # Each level should be roughly half the size of previous
        for i in range(1, len(renderer.pyramids)):
            t_curr, s_min_curr, s_max_curr = renderer.pyramids[i]
            t_prev, s_min_prev, s_max_prev = renderer.pyramids[i - 1]

            # Downsampled by approximately factor of 2
            assert len(t_curr) <= len(t_prev) // 2 + 1
            assert len(t_curr) >= len(t_prev) // 2 - 1

            # Min/max arrays same length as timestamps
            assert len(s_min_curr) == len(t_curr)
            assert len(s_max_curr) == len(t_curr)

    def test_envelope_preserves_peaks(self):
        """Test that min/max envelopes correctly preserve peak features."""
        # Create signal with clear peaks
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)  # Peaks at ±1
        samples[500] = 2.0  # Add an outlier peak

        renderer = LODRenderer(timestamps, samples, num_levels=5)

        # Check that level 1 envelope contains the outlier
        t1, s_min1, s_max1 = renderer.pyramids[1]

        assert np.max(s_max1) >= 1.9, "Max envelope should preserve outlier peak"
        assert np.min(s_min1) <= -0.9, "Min envelope should preserve negative peaks"

    def test_get_full_range(self):
        """Test get_full_range returns correct bounds."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps) * 5  # Amplitude ±5

        renderer = LODRenderer(timestamps, samples)

        x_min, x_max, y_min, y_max = renderer.get_full_range()

        assert x_min == pytest.approx(0.0, rel=1e-6)
        assert x_max == pytest.approx(10.0, rel=1e-6)
        assert y_min <= -4.9  # Close to -5
        assert y_max >= 4.9  # Close to +5


class TestLODRendererGetData:
    """Tests for get_render_data method."""

    def test_get_render_data_full_resolution(self):
        """Test that zoomed-in view returns full resolution data."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples, num_levels=5)

        # View small range with high pixel density (zoomed in)
        t_render, s_render = renderer.get_render_data(x_min=2.0, x_max=3.0, pixel_width=800)

        # Should return full resolution data in range
        expected_mask = (timestamps >= 2.0) & (timestamps <= 3.0)
        expected_points = np.sum(expected_mask)

        assert len(t_render) == expected_points
        np.testing.assert_array_almost_equal(t_render, timestamps[expected_mask])
        np.testing.assert_array_almost_equal(s_render, samples[expected_mask])

    def test_get_render_data_downsampled(self):
        """Test that zoomed-out view returns downsampled data."""
        timestamps = np.linspace(0, 10, 10000)  # Large dataset
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples, num_levels=8)

        # View entire range with low pixel density (zoomed out)
        t_render, s_render = renderer.get_render_data(x_min=0.0, x_max=10.0, pixel_width=800)

        # Should return significantly fewer points than original
        assert len(t_render) < len(timestamps)
        assert len(t_render) > 0

        # For downsampled data, we interleave min/max, so even number of points
        if len(t_render) > len(timestamps) // 10:  # If downsampled
            assert len(t_render) % 2 == 0, "Downsampled data should have even length (min/max pairs)"

    def test_get_render_data_empty_range(self):
        """Test that range with no data returns empty arrays."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples)

        # Request range outside data
        t_render, s_render = renderer.get_render_data(x_min=20.0, x_max=30.0, pixel_width=800)

        assert len(t_render) == 0
        assert len(s_render) == 0

    def test_get_render_data_invalid_range_fails(self):
        """Test that invalid range raises error."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples)

        # x_min >= x_max
        with pytest.raises(ValueError, match="x_min.*must be < x_max"):
            renderer.get_render_data(x_min=5.0, x_max=5.0, pixel_width=800)

        with pytest.raises(ValueError, match="x_min.*must be < x_max"):
            renderer.get_render_data(x_min=6.0, x_max=5.0, pixel_width=800)

    def test_get_render_data_invalid_pixel_width_fails(self):
        """Test that invalid pixel width raises error."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples)

        with pytest.raises(ValueError, match="pixel_width must be > 0"):
            renderer.get_render_data(x_min=0.0, x_max=10.0, pixel_width=0)

        with pytest.raises(ValueError, match="pixel_width must be > 0"):
            renderer.get_render_data(x_min=0.0, x_max=10.0, pixel_width=-100)

    def test_lod_level_selection(self):
        """Test that appropriate LOD level is selected based on zoom."""
        timestamps = np.linspace(0, 100, 100000)  # Large dataset
        samples = np.sin(2 * np.pi * timestamps / 10)

        renderer = LODRenderer(timestamps, samples, num_levels=8)

        # Zoomed out: entire range, few pixels -> high downsampling
        t_zoom_out, s_zoom_out = renderer.get_render_data(
            x_min=0.0, x_max=100.0, pixel_width=800
        )

        # Zoomed in: small range, many pixels -> low/no downsampling
        t_zoom_in, s_zoom_in = renderer.get_render_data(x_min=10.0, x_max=11.0, pixel_width=800)

        # Zoomed-in should have more points per unit time
        zoom_out_density = len(t_zoom_out) / 100.0
        zoom_in_density = len(t_zoom_in) / 1.0

        assert zoom_in_density > zoom_out_density, "Zoomed in should have higher point density"

    def test_get_render_data_partial_overlap(self):
        """Test rendering with partial data overlap."""
        timestamps = np.linspace(5, 15, 1000)
        samples = np.sin(2 * np.pi * timestamps)

        renderer = LODRenderer(timestamps, samples)

        # Request range that partially overlaps data
        t_render, s_render = renderer.get_render_data(x_min=0.0, x_max=10.0, pixel_width=800)

        # Should only return data in overlapping region [5, 10]
        assert len(t_render) > 0
        assert np.min(t_render) >= 5.0
        assert np.max(t_render) <= 10.0


class TestLODRendererEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_dataset(self):
        """Test LODRenderer with minimal valid dataset."""
        timestamps = np.array([0.0, 1.0])
        samples = np.array([1.0, 2.0])

        renderer = LODRenderer(timestamps, samples, num_levels=5)

        # Should have at least level 0
        assert len(renderer.pyramids) >= 1

        # Can still render
        t_render, s_render = renderer.get_render_data(x_min=-1.0, x_max=2.0, pixel_width=100)
        assert len(t_render) == 2

    def test_uniform_signal(self):
        """Test renderer with constant signal."""
        timestamps = np.linspace(0, 10, 1000)
        samples = np.ones(1000) * 5.0  # Constant value

        renderer = LODRenderer(timestamps, samples)

        x_min, x_max, y_min, y_max = renderer.get_full_range()

        assert y_min == pytest.approx(5.0)
        assert y_max == pytest.approx(5.0)

    def test_large_dataset_performance(self):
        """Test that large dataset builds pyramid quickly."""
        import time

        timestamps = np.linspace(0, 100, 1_000_000)  # 1M points
        samples = np.random.randn(1_000_000)

        start = time.time()
        renderer = LODRenderer(timestamps, samples, num_levels=10)
        elapsed = time.time() - start

        # Should build pyramid in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Pyramid build took {elapsed:.2f}s, expected < 1s"

        # Can query render data quickly
        start = time.time()
        t_render, s_render = renderer.get_render_data(x_min=0.0, x_max=100.0, pixel_width=800)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"get_render_data took {elapsed:.3f}s, expected < 0.1s"
        assert len(t_render) > 0
