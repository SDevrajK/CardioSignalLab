"""Interactive peak correction with undo/redo support.

Manages peak addition, deletion, classification, and navigation with a 20-level
undo/redo stack for interactive correction workflows.
"""
from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
from loguru import logger

from cardio_signal_lab.core.data_models import PeakClassification, PeakData


class PeakEditor:
    """Interactive peak editor with undo/redo support.

    Manages peak corrections with a 20-level undo stack. Supports:
    - Add peak (double-click)
    - Delete peak (Delete/Backspace key)
    - Classify peak (D/M/E/B hotkeys)
    - Navigate peaks (arrow keys)
    - Undo/redo (Ctrl+Z/Ctrl+Y)
    """

    def __init__(self, peaks: PeakData | None = None, max_undo_levels: int = 20):
        """Initialize peak editor.

        Args:
            peaks: Initial PeakData (optional)
            max_undo_levels: Maximum undo history depth
        """
        if peaks is None:
            self.indices = np.array([], dtype=int)
            self.classifications = np.array([], dtype=int)
        else:
            self.indices = peaks.indices.copy()
            self.classifications = peaks.classifications.copy()

        self.max_undo_levels = max_undo_levels
        self.undo_stack: deque = deque(maxlen=max_undo_levels)
        self.redo_stack: deque = deque(maxlen=max_undo_levels)

        self.selected_index: int | None = None

    def get_peak_data(self) -> PeakData:
        """Get current peak data as PeakData object.

        Returns:
            PeakData with current indices and classifications
        """
        return PeakData(
            indices=self.indices.copy(), classifications=self.classifications.copy()
        )

    def add_peak(
        self, index: int, classification: PeakClassification = PeakClassification.MANUAL
    ) -> bool:
        """Add a peak at the specified sample index.

        Inserts peak in sorted position by index. Prevents duplicate peaks at
        same index.

        Args:
            index: Sample index for new peak
            classification: Peak classification (default: MANUAL)

        Returns:
            True if peak added, False if peak already exists at this index
        """
        # Check for duplicate
        if len(self.indices) > 0 and index in self.indices:
            logger.warning(f"Peak already exists at index {index}")
            return False

        # Save state for undo
        self._save_state()

        # Find insertion position (maintain sorted order)
        insert_pos = np.searchsorted(self.indices, index)

        # Insert peak
        self.indices = np.insert(self.indices, insert_pos, index)
        self.classifications = np.insert(
            self.classifications, insert_pos, classification.value
        )

        # Update selection to new peak
        self.selected_index = insert_pos

        logger.info(
            f"Added peak at index {index} with classification {classification.name}"
        )
        return True

    def delete_peak(self, peak_index: int) -> bool:
        """Delete peak by its position in the peak array.

        Args:
            peak_index: Index in the peaks array (0-based)

        Returns:
            True if deleted, False if index invalid
        """
        if not 0 <= peak_index < len(self.indices):
            logger.warning(f"Invalid peak index: {peak_index}")
            return False

        # Save state for undo
        self._save_state()

        # Remove peak
        sample_index = self.indices[peak_index]
        self.indices = np.delete(self.indices, peak_index)
        self.classifications = np.delete(self.classifications, peak_index)

        # Update selection
        if len(self.indices) == 0:
            self.selected_index = None
        elif peak_index >= len(self.indices):
            # Deleted last peak, select new last
            self.selected_index = len(self.indices) - 1
        else:
            # Select peak that's now at the deleted position
            self.selected_index = peak_index

        logger.info(f"Deleted peak at sample index {sample_index}")
        return True

    def classify_peak(
        self, peak_index: int, classification: PeakClassification
    ) -> bool:
        """Change classification of a peak.

        Args:
            peak_index: Index in the peaks array
            classification: New classification

        Returns:
            True if updated, False if index invalid
        """
        if not 0 <= peak_index < len(self.indices):
            logger.warning(f"Invalid peak index: {peak_index}")
            return False

        # Save state for undo
        self._save_state()

        # Update classification
        self.classifications[peak_index] = classification.value

        logger.info(
            f"Changed peak {peak_index} classification to {classification.name}"
        )
        return True

    def cycle_classification(self, peak_index: int) -> bool:
        """Cycle peak classification through AUTO → MANUAL → ECTOPIC → BAD → AUTO.

        Args:
            peak_index: Index in the peaks array

        Returns:
            True if cycled, False if index invalid
        """
        if not 0 <= peak_index < len(self.indices):
            return False

        current = PeakClassification(self.classifications[peak_index])
        cycle_order = [
            PeakClassification.AUTO,
            PeakClassification.MANUAL,
            PeakClassification.ECTOPIC,
            PeakClassification.BAD,
        ]

        # Find next classification
        try:
            current_idx = cycle_order.index(current)
            next_idx = (current_idx + 1) % len(cycle_order)
            next_classification = cycle_order[next_idx]
        except ValueError:
            # Current classification not in cycle, default to AUTO
            next_classification = PeakClassification.AUTO

        return self.classify_peak(peak_index, next_classification)

    def select_peak(self, peak_index: int | None) -> bool:
        """Select a peak by its position in the array.

        Args:
            peak_index: Index to select (None to deselect)

        Returns:
            True if selection changed
        """
        if peak_index is not None and not 0 <= peak_index < len(self.indices):
            logger.warning(f"Invalid peak index: {peak_index}")
            return False

        if self.selected_index != peak_index:
            self.selected_index = peak_index
            return True
        return False

    def navigate_peaks(
        self, direction: Literal["next", "prev", "first", "last"]
    ) -> int | None:
        """Navigate between peaks.

        Args:
            direction: Navigation direction

        Returns:
            New selected peak index, or None if no peaks
        """
        if len(self.indices) == 0:
            return None

        if direction == "next":
            if self.selected_index is None:
                new_index = 0
            else:
                new_index = min(self.selected_index + 1, len(self.indices) - 1)
        elif direction == "prev":
            if self.selected_index is None:
                new_index = len(self.indices) - 1
            else:
                new_index = max(self.selected_index - 1, 0)
        elif direction == "first":
            new_index = 0
        elif direction == "last":
            new_index = len(self.indices) - 1
        else:
            logger.warning(f"Invalid direction: {direction}")
            return None

        self.selected_index = new_index
        return new_index

    def find_nearest_peak(self, sample_index: int, max_distance: int = 50) -> int | None:
        """Find peak nearest to a sample index.

        Args:
            sample_index: Sample index to search near
            max_distance: Maximum distance in samples to consider

        Returns:
            Peak array index if found within max_distance, else None
        """
        if len(self.indices) == 0:
            return None

        # Find distances to all peaks
        distances = np.abs(self.indices - sample_index)
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]

        if nearest_distance <= max_distance:
            return int(nearest_idx)
        return None

    def undo(self) -> bool:
        """Undo last operation.

        Returns:
            True if undone, False if nothing to undo
        """
        if len(self.undo_stack) == 0:
            logger.info("Nothing to undo")
            return False

        # Save current state to redo stack
        self._save_to_redo()

        # Restore previous state
        state = self.undo_stack.pop()
        self.indices = state["indices"]
        self.classifications = state["classifications"]
        self.selected_index = state["selected_index"]

        logger.info("Undid last operation")
        return True

    def redo(self) -> bool:
        """Redo last undone operation.

        Returns:
            True if redone, False if nothing to redo
        """
        if len(self.redo_stack) == 0:
            logger.info("Nothing to redo")
            return False

        # Save current state to undo stack (without clearing redo)
        state = {
            "indices": self.indices.copy(),
            "classifications": self.classifications.copy(),
            "selected_index": self.selected_index,
        }
        self.undo_stack.append(state)

        # Restore redo state
        state = self.redo_stack.pop()
        self.indices = state["indices"]
        self.classifications = state["classifications"]
        self.selected_index = state["selected_index"]

        logger.info("Redid operation")
        return True

    def reset(self):
        """Reset to empty state and clear undo/redo history.

        Called when loading a new file.
        """
        self.indices = np.array([], dtype=int)
        self.classifications = np.array([], dtype=int)
        self.selected_index = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        logger.info("Peak editor reset")

    def _save_state(self):
        """Save current state to undo stack and clear redo stack."""
        state = {
            "indices": self.indices.copy(),
            "classifications": self.classifications.copy(),
            "selected_index": self.selected_index,
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo when new action taken

    def _save_to_redo(self):
        """Save current state to redo stack."""
        state = {
            "indices": self.indices.copy(),
            "classifications": self.classifications.copy(),
            "selected_index": self.selected_index,
        }
        self.redo_stack.append(state)

    @property
    def num_peaks(self) -> int:
        """Total number of peaks."""
        return len(self.indices)

    @property
    def can_undo(self) -> bool:
        """Whether undo is available."""
        return len(self.undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Whether redo is available."""
        return len(self.redo_stack) > 0
