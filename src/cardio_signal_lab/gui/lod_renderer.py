"""Level-of-detail renderer for efficient large signal visualization.

Precomputes min/max envelope pyramids at multiple resolutions for smooth
rendering of 10M+ point signals. Automatically selects appropriate LOD level
based on visible range and pixel density.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


class LODRenderer:
    """Precomputed level-of-detail renderer for signal data.

    Builds a pyramid of progressively downsampled min/max envelopes to enable
    smooth rendering of very large datasets (10M+ points). Automatically selects
    the appropriate resolution level based on visible range vs pixel width.

    Attributes:
        timestamps: Original full-resolution timestamps
        samples: Original full-resolution samples
        pyramids: List of (timestamps, min_envelope, max_envelope) tuples at each LOD level
        num_levels: Number of LOD levels in the pyramid
    """

    def __init__(self, timestamps: np.ndarray, samples: np.ndarray, num_levels: int = 8):
        """Initialize LOD renderer with signal data.

        Args:
            timestamps: 1D array of timestamps (must be monotonic)
            samples: 1D array of signal values
            num_levels: Number of LOD levels to precompute (default: 8, covers 2^8=256x downsampling)

        Raises:
            ValueError: If timestamps and samples have different lengths
            ValueError: If timestamps are not monotonic
        """
        if len(timestamps) != len(samples):
            raise ValueError(
                f"timestamps ({len(timestamps)}) and samples ({len(samples)}) must have same length"
            )

        if len(timestamps) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(timestamps)}")

        # Verify monotonic timestamps
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")

        self.timestamps = timestamps
        self.samples = samples
        self.num_levels = num_levels

        # Build LOD pyramid
        self.pyramids = self._build_pyramids()

        logger.debug(
            f"LODRenderer initialized: {len(samples)} samples, {num_levels} levels, "
            f"pyramid sizes: {[len(p[0]) for p in self.pyramids]}"
        )

    def _build_pyramids(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Build multi-resolution min/max envelope pyramid.

        Each level is computed directly from the original full-resolution signal
        so that true peak amplitudes are always contained within the envelope.
        Computing from averaged intermediate levels causes peak clipping at high
        zoom-out, making detected peak markers appear above the rendered signal.

        Returns:
            List of (timestamps, min_values, max_values) tuples, one per LOD level.
            Level 0 is full resolution, level N downsamples by factor 2^N.
        """
        pyramids = []

        # Level 0: Full resolution (min == max == samples)
        pyramids.append((self.timestamps, self.samples, self.samples))

        for level in range(1, self.num_levels):
            block_size = 2 ** level
            n_blocks = len(self.samples) // block_size

            if n_blocks < 2:
                logger.debug(f"Stopping pyramid at level {level}: insufficient points")
                break

            # Reshape original signal into (n_blocks, block_size) for vectorised min/max.
            # Truncate to a multiple of block_size to allow clean reshape.
            n_samples = n_blocks * block_size
            s_blocks = self.samples[:n_samples].reshape(n_blocks, block_size)
            t_blocks = self.timestamps[:n_samples].reshape(n_blocks, block_size)

            s_min = s_blocks.min(axis=1)
            s_max = s_blocks.max(axis=1)
            t_downsampled = t_blocks[:, 0]  # First timestamp of each block

            pyramids.append((t_downsampled, s_min, s_max))

        logger.debug(f"Built {len(pyramids)} LOD levels")
        return pyramids

    def get_render_data(
        self, x_min: float, x_max: float, pixel_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get appropriately downsampled data for rendering.

        Selects the optimal LOD level based on visible range and pixel density.
        Returns full-resolution data when sufficiently zoomed in.

        Args:
            x_min: Left edge of visible range (timestamp)
            x_max: Right edge of visible range (timestamp)
            pixel_width: Width of plot in pixels

        Returns:
            Tuple of (timestamps, samples) to render, cropped to visible range.
            When using downsampled data, returns interleaved min/max envelopes.

        Raises:
            ValueError: If x_min >= x_max or pixel_width <= 0
        """
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be < x_max ({x_max})")

        if pixel_width <= 0:
            raise ValueError(f"pixel_width must be > 0, got {pixel_width}")

        # Find how many original samples are in visible range
        full_t, full_s, _ = self.pyramids[0]
        mask = (full_t >= x_min) & (full_t <= x_max)
        visible_points = np.sum(mask)

        if visible_points == 0:
            # No data in range, return empty
            return np.array([]), np.array([])

        # Calculate points per pixel
        points_per_pixel = visible_points / pixel_width

        # Select LOD level: target 1-2 points per pixel for optimal rendering
        # Use log2 to find downsampling level needed
        if points_per_pixel <= 2:
            # Zoomed in enough, use full resolution
            level = 0
        else:
            # Need downsampling
            level = int(np.log2(points_per_pixel / 2))
            level = min(level, len(self.pyramids) - 1)  # Cap at max level

        # Get data from selected level
        t_data, s_min, s_max = self.pyramids[level]

        # Crop to visible range
        mask = (t_data >= x_min) & (t_data <= x_max)
        t_visible = t_data[mask]
        s_min_visible = s_min[mask]
        s_max_visible = s_max[mask]

        # For level 0 (full res), min==max, just return samples
        if level == 0:
            result_t = t_visible
            result_s = s_min_visible
        else:
            # For downsampled levels, interleave min and max to draw envelope
            # This preserves peak visibility even when zoomed out
            result_t = np.repeat(t_visible, 2)
            result_s = np.empty(len(result_t))
            result_s[0::2] = s_min_visible
            result_s[1::2] = s_max_visible

        logger.debug(
            f"get_render_data: visible_range=[{x_min:.2f}, {x_max:.2f}], "
            f"pixel_width={pixel_width}, points_per_pixel={points_per_pixel:.1f}, "
            f"selected_level={level}/{len(self.pyramids)-1}, "
            f"returned {len(result_t)} points"
        )

        return result_t, result_s

    def get_full_range(self) -> tuple[float, float, float, float]:
        """Get full data range for initial plot setup.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max) covering all data
        """
        return (
            float(self.timestamps[0]),
            float(self.timestamps[-1]),
            float(np.min(self.samples)),
            float(np.max(self.samples)),
        )

    @property
    def num_samples(self) -> int:
        """Total number of samples at full resolution."""
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])
