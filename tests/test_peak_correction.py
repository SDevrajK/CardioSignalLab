"""Tests for interactive peak correction with undo/redo.

Tests PeakEditor class for peak addition, deletion, classification,
navigation, and undo/redo stack management.
"""
import numpy as np
import pytest

from cardio_signal_lab.core.data_models import PeakClassification, PeakData
from cardio_signal_lab.processing.peak_correction import PeakEditor


@pytest.fixture
def empty_editor():
    """Create empty PeakEditor."""
    return PeakEditor()


@pytest.fixture
def editor_with_peaks():
    """Create PeakEditor with some initial peaks."""
    peaks = PeakData(
        indices=np.array([100, 200, 300, 400], dtype=int),
        classifications=np.array(
            [
                PeakClassification.AUTO.value,
                PeakClassification.AUTO.value,
                PeakClassification.MANUAL.value,
                PeakClassification.AUTO.value,
            ],
            dtype=int,
        ),
    )
    return PeakEditor(peaks=peaks)


class TestPeakEditorBasics:
    def test_initialize_empty(self, empty_editor):
        """Test initialization with no peaks."""
        assert empty_editor.num_peaks == 0
        assert len(empty_editor.indices) == 0
        assert len(empty_editor.classifications) == 0
        assert empty_editor.selected_index is None

    def test_initialize_with_peaks(self, editor_with_peaks):
        """Test initialization with existing peaks."""
        assert editor_with_peaks.num_peaks == 4
        assert len(editor_with_peaks.indices) == 4
        assert editor_with_peaks.indices[0] == 100

    def test_get_peak_data(self, editor_with_peaks):
        """Test getting PeakData object."""
        peak_data = editor_with_peaks.get_peak_data()
        assert isinstance(peak_data, PeakData)
        assert len(peak_data.indices) == 4
        assert peak_data.num_auto == 3
        assert peak_data.num_manual == 1


class TestAddPeak:
    def test_add_peak_to_empty(self, empty_editor):
        """Test adding peak to empty editor."""
        success = empty_editor.add_peak(150, PeakClassification.MANUAL)
        assert success
        assert empty_editor.num_peaks == 1
        assert empty_editor.indices[0] == 150
        assert empty_editor.classifications[0] == PeakClassification.MANUAL.value
        assert empty_editor.selected_index == 0

    def test_add_peak_maintains_sort(self, editor_with_peaks):
        """Test that adding peak maintains sorted order."""
        editor_with_peaks.add_peak(250, PeakClassification.MANUAL)
        assert editor_with_peaks.num_peaks == 5
        # Should be inserted between 200 and 300
        assert np.all(np.diff(editor_with_peaks.indices) > 0)  # Check sorted
        assert 250 in editor_with_peaks.indices

    def test_add_peak_at_start(self, editor_with_peaks):
        """Test adding peak before all existing peaks."""
        editor_with_peaks.add_peak(50, PeakClassification.MANUAL)
        assert editor_with_peaks.indices[0] == 50
        assert editor_with_peaks.num_peaks == 5

    def test_add_peak_at_end(self, editor_with_peaks):
        """Test adding peak after all existing peaks."""
        editor_with_peaks.add_peak(500, PeakClassification.MANUAL)
        assert editor_with_peaks.indices[-1] == 500
        assert editor_with_peaks.num_peaks == 5

    def test_add_duplicate_peak_fails(self, editor_with_peaks):
        """Test that adding duplicate peak fails."""
        success = editor_with_peaks.add_peak(200, PeakClassification.MANUAL)
        assert not success
        assert editor_with_peaks.num_peaks == 4  # No change


class TestDeletePeak:
    def test_delete_peak(self, editor_with_peaks):
        """Test deleting a peak."""
        success = editor_with_peaks.delete_peak(1)  # Delete peak at index 200
        assert success
        assert editor_with_peaks.num_peaks == 3
        assert 200 not in editor_with_peaks.indices

    def test_delete_first_peak(self, editor_with_peaks):
        """Test deleting first peak."""
        editor_with_peaks.delete_peak(0)
        assert editor_with_peaks.num_peaks == 3
        assert editor_with_peaks.indices[0] == 200

    def test_delete_last_peak(self, editor_with_peaks):
        """Test deleting last peak."""
        editor_with_peaks.delete_peak(3)
        assert editor_with_peaks.num_peaks == 3
        assert editor_with_peaks.indices[-1] == 300

    def test_delete_invalid_index(self, editor_with_peaks):
        """Test deleting with invalid index."""
        success = editor_with_peaks.delete_peak(10)
        assert not success
        assert editor_with_peaks.num_peaks == 4

    def test_delete_updates_selection(self, editor_with_peaks):
        """Test that deletion updates selection appropriately."""
        editor_with_peaks.selected_index = 2
        editor_with_peaks.delete_peak(2)
        # Selection should move to same position (now a different peak)
        assert editor_with_peaks.selected_index == 2

    def test_delete_last_peak_clears_selection(self):
        """Test deleting the only peak clears selection."""
        editor = PeakEditor(
            PeakData(
                indices=np.array([100]), classifications=np.array([PeakClassification.AUTO.value])
            )
        )
        editor.selected_index = 0
        editor.delete_peak(0)
        assert editor.selected_index is None


class TestClassifyPeak:
    def test_classify_peak(self, editor_with_peaks):
        """Test changing peak classification."""
        success = editor_with_peaks.classify_peak(0, PeakClassification.ECTOPIC)
        assert success
        assert editor_with_peaks.classifications[0] == PeakClassification.ECTOPIC.value

    def test_classify_invalid_index(self, editor_with_peaks):
        """Test classifying with invalid index."""
        success = editor_with_peaks.classify_peak(10, PeakClassification.BAD)
        assert not success

    def test_cycle_classification(self, editor_with_peaks):
        """Test cycling through classifications."""
        # Start with AUTO
        assert editor_with_peaks.classifications[0] == PeakClassification.AUTO.value
        # Cycle: AUTO -> MANUAL
        editor_with_peaks.cycle_classification(0)
        assert editor_with_peaks.classifications[0] == PeakClassification.MANUAL.value
        # Cycle: MANUAL -> ECTOPIC
        editor_with_peaks.cycle_classification(0)
        assert editor_with_peaks.classifications[0] == PeakClassification.ECTOPIC.value
        # Cycle: ECTOPIC -> BAD
        editor_with_peaks.cycle_classification(0)
        assert editor_with_peaks.classifications[0] == PeakClassification.BAD.value
        # Cycle: BAD -> AUTO
        editor_with_peaks.cycle_classification(0)
        assert editor_with_peaks.classifications[0] == PeakClassification.AUTO.value


class TestSelection:
    def test_select_peak(self, editor_with_peaks):
        """Test selecting a peak."""
        success = editor_with_peaks.select_peak(2)
        assert success
        assert editor_with_peaks.selected_index == 2

    def test_deselect_peak(self, editor_with_peaks):
        """Test deselecting."""
        editor_with_peaks.selected_index = 1
        editor_with_peaks.select_peak(None)
        assert editor_with_peaks.selected_index is None

    def test_select_invalid_index(self, editor_with_peaks):
        """Test selecting invalid index."""
        success = editor_with_peaks.select_peak(10)
        assert not success


class TestNavigation:
    def test_navigate_next(self, editor_with_peaks):
        """Test navigating to next peak."""
        editor_with_peaks.selected_index = 1
        new_idx = editor_with_peaks.navigate_peaks("next")
        assert new_idx == 2
        assert editor_with_peaks.selected_index == 2

    def test_navigate_prev(self, editor_with_peaks):
        """Test navigating to previous peak."""
        editor_with_peaks.selected_index = 2
        new_idx = editor_with_peaks.navigate_peaks("prev")
        assert new_idx == 1

    def test_navigate_first(self, editor_with_peaks):
        """Test navigating to first peak."""
        new_idx = editor_with_peaks.navigate_peaks("first")
        assert new_idx == 0

    def test_navigate_last(self, editor_with_peaks):
        """Test navigating to last peak."""
        new_idx = editor_with_peaks.navigate_peaks("last")
        assert new_idx == 3

    def test_navigate_next_at_end(self, editor_with_peaks):
        """Test navigating next when at last peak."""
        editor_with_peaks.selected_index = 3
        new_idx = editor_with_peaks.navigate_peaks("next")
        assert new_idx == 3  # Stays at last

    def test_navigate_prev_at_start(self, editor_with_peaks):
        """Test navigating previous when at first peak."""
        editor_with_peaks.selected_index = 0
        new_idx = editor_with_peaks.navigate_peaks("prev")
        assert new_idx == 0  # Stays at first

    def test_navigate_empty(self, empty_editor):
        """Test navigation with no peaks."""
        new_idx = empty_editor.navigate_peaks("next")
        assert new_idx is None


class TestFindNearestPeak:
    def test_find_nearest_peak(self, editor_with_peaks):
        """Test finding nearest peak."""
        # Nearest to 195 should be peak at index 200
        nearest = editor_with_peaks.find_nearest_peak(195, max_distance=50)
        assert nearest == 1  # Peak array index

    def test_find_nearest_beyond_threshold(self, editor_with_peaks):
        """Test finding with distance beyond threshold."""
        nearest = editor_with_peaks.find_nearest_peak(195, max_distance=4)
        assert nearest is None

    def test_find_nearest_empty(self, empty_editor):
        """Test finding in empty editor."""
        nearest = empty_editor.find_nearest_peak(100)
        assert nearest is None


class TestUndo:
    def test_undo_add(self, empty_editor):
        """Test undoing peak addition."""
        empty_editor.add_peak(100, PeakClassification.MANUAL)
        assert empty_editor.num_peaks == 1
        success = empty_editor.undo()
        assert success
        assert empty_editor.num_peaks == 0

    def test_undo_delete(self, editor_with_peaks):
        """Test undoing peak deletion."""
        original_count = editor_with_peaks.num_peaks
        editor_with_peaks.delete_peak(1)
        assert editor_with_peaks.num_peaks == original_count - 1
        editor_with_peaks.undo()
        assert editor_with_peaks.num_peaks == original_count
        assert 200 in editor_with_peaks.indices

    def test_undo_classification(self, editor_with_peaks):
        """Test undoing classification change."""
        original = editor_with_peaks.classifications[0]
        editor_with_peaks.classify_peak(0, PeakClassification.ECTOPIC)
        editor_with_peaks.undo()
        assert editor_with_peaks.classifications[0] == original

    def test_undo_empty_stack(self, empty_editor):
        """Test undo with nothing to undo."""
        success = empty_editor.undo()
        assert not success

    def test_undo_restores_selection(self, editor_with_peaks):
        """Test that undo restores selection state."""
        editor_with_peaks.selected_index = 2
        editor_with_peaks.add_peak(250, PeakClassification.MANUAL)
        # Selection was updated to insertion position (between indices 200 and 300)
        # Peaks: [100, 200, 250, 300, 400], so insertion at index 2
        assert editor_with_peaks.selected_index == 2
        # Undo should restore previous selection (was also 2 before)
        editor_with_peaks.undo()
        assert editor_with_peaks.selected_index == 2


class TestRedo:
    def test_redo_after_undo(self, editor_with_peaks):
        """Test redoing after undo."""
        editor_with_peaks.add_peak(250, PeakClassification.MANUAL)
        assert editor_with_peaks.num_peaks == 5
        editor_with_peaks.undo()
        assert editor_with_peaks.num_peaks == 4
        success = editor_with_peaks.redo()
        assert success
        assert editor_with_peaks.num_peaks == 5

    def test_redo_empty_stack(self, empty_editor):
        """Test redo with nothing to redo."""
        success = empty_editor.redo()
        assert not success

    def test_new_action_clears_redo_stack(self, editor_with_peaks):
        """Test that new action clears redo stack."""
        editor_with_peaks.add_peak(250, PeakClassification.MANUAL)
        editor_with_peaks.undo()
        # Redo is available
        assert editor_with_peaks.can_redo
        # New action clears redo stack
        editor_with_peaks.add_peak(260, PeakClassification.MANUAL)
        assert not editor_with_peaks.can_redo


class TestUndoStackLimit:
    def test_undo_stack_limit(self):
        """Test that undo stack is limited to max_undo_levels."""
        editor = PeakEditor(max_undo_levels=5)
        # Add 10 peaks
        for i in range(10):
            editor.add_peak(i * 100, PeakClassification.MANUAL)
        # Should only be able to undo 5 times
        undo_count = 0
        while editor.undo():
            undo_count += 1
        assert undo_count == 5
        # Should have 5 peaks left (10 - 5 undos)
        assert editor.num_peaks == 5

    def test_redo_stack_limit(self):
        """Test that redo stack is also limited."""
        editor = PeakEditor(max_undo_levels=3)
        for i in range(5):
            editor.add_peak(i * 100, PeakClassification.MANUAL)
        # Undo all
        for _ in range(5):
            editor.undo()
        # Should only be able to redo 3 times (stack limit)
        redo_count = 0
        while editor.redo():
            redo_count += 1
        assert redo_count == 3


class TestReset:
    def test_reset(self, editor_with_peaks):
        """Test resetting editor."""
        editor_with_peaks.reset()
        assert editor_with_peaks.num_peaks == 0
        assert len(editor_with_peaks.undo_stack) == 0
        assert len(editor_with_peaks.redo_stack) == 0
        assert editor_with_peaks.selected_index is None


class TestCanUndoRedo:
    def test_can_undo_redo_properties(self, empty_editor):
        """Test can_undo and can_redo properties."""
        assert not empty_editor.can_undo
        assert not empty_editor.can_redo

        empty_editor.add_peak(100, PeakClassification.MANUAL)
        assert empty_editor.can_undo
        assert not empty_editor.can_redo

        empty_editor.undo()
        assert not empty_editor.can_undo
        assert empty_editor.can_redo
