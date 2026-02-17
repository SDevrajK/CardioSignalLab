# Manual Testing Checklist: CardioSignalLab MVP

**Tester**: Sayeed A. D. Kizuk
**Date**: ___________
**Build**: `95ca6a0` (MVP)
**Environment**: Miniconda `ekgpeakcorrector`, Python 3.12, WSL2

Mark each item: `[P]` Pass | `[F]` Fail | `[S]` Skip (with reason)
Failures: note what happened in the space below the item.

---

## 1. App Launch

- [x] 1.1 App launches without errors: `python main.py`
- [x] 1.2 Window title is correct
- [x] 1.3 Menu bar shows **File, Edit, Select, View, Help** (multi-signal mode defaults)
- [x] 1.4 Edit menu items (Add Peak, Delete Peak, Undo, Redo) are disabled
  * note: they are disabled, but it makes no sense for add peak and delete peak to be present here.
- [x] 1.5 Status bar shows an appropriate idle/empty message
- [] 1.6 No errors or warnings in the terminal on launch
  * note: there is a warning: 2026-02-17 07:13:43 | INFO     | cardio_signal_lab.gui.event_overlay:set_events:56 - EventOverlay.set_events() called with 16 events
C:\Users\sayee\Documents\Research\HébertLab\CardioSignalLab\src\cardio_signal_lab\gui\event_overlay.py:111: RuntimeWarning: Failed to disconnect (<bound method EventOverlay._on_range_changed of <cardio_signal_lab.gui.event_overlay.EventOverlay object at 0x00000189B4C6BD70>>) from signal "sigRangeChanged(PyObject,PyObject,PyObject)".
  view_box.sigRangeChanged.disconnect(self._on_range_changed)
---

## 2. File Loading -- XDF

**Test file**: use a real XDF file with at least ECG and one other signal type.

- [x] 2.1 File > Open (Ctrl+O) opens a file dialog filtered to XDF/CSV
- [x] 2.2 Loading an XDF file succeeds without error dialog
- [x] 2.3 Metadata dialog appears showing: signal count, signal types, sampling rates, duration
- [x] 2.4 After dismissing the dialog, multi-signal view is shown with all signals plotted
- [x] 2.5 Status bar updates to "Multi-Signal Mode (N signals loaded)"
- [x] 2.6 Signal types are correctly auto-detected (ECG, PPG, EDA) -- check against known file contents
- [x] 2.7 Each signal uses the correct sampling rate (inspect axis scale vs expected duration)
- [x] 2.8 Event markers appear on the plots (if the XDF file contains a marker stream)

**Notes**:
everything looks correct. 
---

## 3. File Loading -- CSV

**Test file**: use a Shimmer CSV file (3-row header format) or a plain CSV.

- [x] 3.1 File > Open opens a file dialog; selecting a CSV prompts for signal type
  * note: it didn't prompt for signal type, it detected it
- [x] 3.2 Choosing signal type (e.g., PPG) loads the file successfully
  * note: it detected signal type automatically
- [x] 3.3 Signal appears in multi-signal view with correct label
- [x] 3.4 Sampling rate is auto-calculated from timestamps (check axis scale)
- [x] 3.5 **Shimmer 3-row header**: if using a Shimmer CSV, column names are parsed correctly
- [x] 3.6 **Companion events file**: if `events_*.json` or `events_*.csv` exists alongside the CSV, events load automatically
  * note : in multi-signal view, the event labels are cut off slightly at the top. We only see the bottom half of the label. It's fine in single signal view.

**Notes**:

---

## 4. File Loading -- Error Handling

- [x] 4.1 Attempting to open a non-existent file shows a user-friendly error dialog (not a crash)
- [x] 4.2 Opening an unrecognized file format shows a clear error message
- [x] 4.3 Log panel (View > Log Panel or `L`) shows the validation/error messages

**Notes**:

---

## 5. Multi-Signal View

- [x] 5.1 All loaded signals are displayed in stacked subplots
- [x] 5.2 Zoom (scroll wheel) is synchronized across all subplots (all zoom together)
- [x] 5.3 Pan (drag) is synchronized across all subplots
- [ ] 5.4 Zoom In/Out via Ctrl+Plus / Ctrl+Minus works
  * note: Ctrl+Minus works, but Ctrl+Plus does not
- [x] 5.5 Home key jumps to start of recording; End key jumps to end
- [x] 5.6 View > Jump to Time (Ctrl+T) scrolls to the entered timestamp
- [x] 5.7 Zoom Mode (Z key): dragging a rectangle zooms to that region
- [x] 5.8 Pan Mode (P key): dragging pans without zooming
- [ ] 5.9 Right-click context menu on a plot shows "View All", "Pan Mode", "Zoom Mode"
  * bug: right clicking goes into single channel view incorrectly
- [x] 5.10 Event markers (vertical lines with labels) are visible across all subplots
- [x] 5.11 View > Toggle Event Markers (E key) hides/shows event markers

**Notes**:

---

## 6. View Hierarchy Navigation (Multi -> Type -> Channel)

**Requires a multi-channel signal type (e.g., ECG with 4 channels).**

- [x] 6.1 Select menu lists available signal types (e.g., ECG, PPG)
- [x] 6.2 Clicking a signal type in multi-signal view enters the **signal-type view**
  - Stacked plots showing each channel of that type
  - Menu shows **File, Edit, Process, View, Help**
- [x] 6.3 L2 Norm button is visible in signal-type view
  - Clicking it creates and displays the derived L2 Norm channel
    * note: this procedure should force the user to select which channels get used in the norm
    * note: there's no reason for this to be a seperate button, it should be inside the menus
    * note: also, the L2 Norm channel disappears when you go back to the multi-signal view.
- [x] 6.4 Clicking a channel in the signal-type view enters **single-channel view**
  - Only that channel is shown full-size
- [x] 6.5 ESC from single-channel view returns to signal-type view
- [x] 6.6 ESC from signal-type view returns to multi-signal view
- [x] 6.7 View > Return to Multi-Signal View also returns correctly from any depth
- [x] 6.8 Status bar updates correctly at each level
- [x] 6.9 Process menu is only enabled in single-channel view (grayed out at other levels)

**Notes**:

---

## 7. Signal Processing -- Filters

**Enter single-channel view (ECG) before testing.**

- [x] 7.1 Process > Filter > Bandpass opens a parameter dialog
  - Adjust low/high cutoff and order; confirm applies without error
- [x] 7.2 After bandpass filter, the signal visually changes (smoother, baseline reduced)
- [x] 7.3 Process > Filter > Notch filter applies without error
- [x] 7.4 Process > Baseline Correction applies without error
  * note: I don't understand why baseline correction has a polynomial parameter 
- [x] 7.5 Process > Zero Reference applies without error
  * note: this is baseline correction, I don't know what the other option is
- [x] 7.6 Processing history is tracked (check log panel for step messages)
  * note: the log panel does show the messages, but it's not very clear. A better solution would be a seperate processing panel that only shows the sequence of processing steps that have been applied
- [x] 7.7 Process > Reset Processing restores the original (unfiltered) signal

**Notes**:

---

## 8. Signal Processing -- EEMD Artifact Removal

- [x] 8.1 Process > Artifact Removal > EEMD opens a parameter dialog (or confirmation)
  * note: EEMD is currently only tested as a PPG artifact cleaning technique. For now, it should only be available for PPG signals. ECG signals should have access to neurokit ecg_clean.
- [x] 8.2 EEMD runs in a background thread -- a progress dialog appears and the GUI stays responsive
- [x] 8.3 Cancel button in the progress dialog cancels the operation without crashing
- [ ] 8.4 (Re-run without canceling) EEMD completes and the signal updates
    * note: it doesn't seem like the EEMD ever completes. Perhaps it is running into a silent error
- [ ] 8.5 Log panel shows EEMD progress or completion message
    * note: no progress, just these lines
    [INFO    ] Processing worker started: run_eemd
    [INFO    ] Starting EEMD decomposition: 114824 samples, ensemble_size=100, noise_width=0.2  
    [INFO    ] Processing cancellation requested
    
**Notes**:

---

## 9. Peak Detection

**Apply at least one filter before detecting peaks for realistic results.**

- [ ] 9.1 **ECG**: Process > Detect R-Peaks runs and overlays peak markers on the signal
- [x] 9.2 **PPG**: In a PPG single-channel view, Process > Detect Pulse Peaks runs correctly
  * note: the peaks don't show in the right y-location when the signal is at full view. When zoomed, they show at the right vertical position, but when unzoomed, they are higher up than they should be
  * note: if possible, it would be better if the marker size was smaller when zoomed out, and at the current size when zoomed in
- [ ] 9.3 **EDA**: In an EDA single-channel view, Process > Detect SCR runs correctly
- [x] 9.4 Auto-detected peaks are shown in blue
- [ ] 9.5 Detected peaks visually align with signal morphology (spot-check)
    * note: only when zoomed in
- [ ] 9.6 Peak detection runs in background with progress indicator (visible for >2s tasks)
- [ ] 9.7 Log panel shows peak count or detection summary

**Notes**:

---

## 10. Interactive Peak Correction

**Requires peaks to be detected first (Section 9).**

- [ ] 10.1 **Add peak**: Double-click on the signal at a location without a peak -- a green (MANUAL) marker appears
- [ ] 10.2 **Select peak**: Single-click on an existing peak marker -- it highlights (yellow/orange)
- [ ] 10.3 **Delete selected peak**: Press Delete or Backspace -- the selected peak is removed
- [ ] 10.4 **Undo**: Ctrl+Z restores the deleted peak
- [ ] 10.5 **Redo**: Ctrl+Y re-deletes the peak
- [ ] 10.6 Undo/redo stack works up to 20 operations without issue
- [ ] 10.7 **Classify as Ectopic**: Select a peak and press `E` -- marker turns orange
- [ ] 10.8 **Classify as Bad**: Select a peak and press `B` -- marker turns red
- [ ] 10.9 **Classify as Manual**: Press `M` -- marker turns green
- [ ] 10.10 Peak add/delete feels responsive (no visible lag, <100ms)
- [ ] 10.11 Zoom/pan continues to work normally during peak correction (no mode conflicts)
- [ ] 10.12 **Arrow key navigation**: Left/Right arrow keys cycle through peaks (check if implemented)

**Notes**:

---

## 11. Export

**After processing + peak correction (Sections 7-10).**

- [ ] 11.1 File > Export opens a format selection dialog or file dialog
- [ ] 11.2 **CSV export**: Exports a `.csv` file with columns: `time_s`, `signal`, `peak`, `peak_classification`
  - Open in a text editor and verify column headers and data look correct
- [ ] 11.3 **NPY export**: Exports `.npy` files for signal, peaks, classifications
  - Verify files exist on disk
- [ ] 11.4 **Annotations export**: Exports an annotations CSV with `peak_index`, `time`, `amplitude`, `classification`
- [ ] 11.5 A `_parameters.json` sidecar file is saved alongside the export (processing pipeline)
  - Open and verify it lists the processing steps you applied
- [ ] 11.6 Exported signal values match what is plotted (spot-check a few timestamps)

**Notes**:

---

## 12. Session Save and Load

- [ ] 12.1 File > Save (Ctrl+S) opens a save dialog and saves a `.csl.json` session file
- [ ] 12.2 Open the `.csl.json` in a text editor -- verify it contains: source path, pipeline steps, peak data
- [ ] 12.3 File > Open, select the `.csl.json` -- the session loads with source file, processing state, and peaks restored
- [ ] 12.4 After loading a session, peaks are visible on the signal and match what was saved
- [ ] 12.5 After loading a session, undo/redo history is empty (does not persist across sessions)

**Notes**:

---

## 13. Log Panel

- [ ] 13.1 View > Log Panel (or `L` key) toggles the log dock widget visible/hidden
- [ ] 13.2 Log messages appear in real-time during file loading and processing
- [ ] 13.3 WARNING messages appear in a distinct color (yellow/amber)
- [ ] 13.4 ERROR messages appear in a distinct color (red)
- [ ] 13.5 Log panel can be resized and repositioned (it's a dock widget)
- [ ] 13.6 Log messages also appear in the terminal (dual output)

**Notes**:

---

## 14. Keyboard Shortcuts

| Shortcut | Expected Action | Pass/Fail |
|----------|----------------|-----------|
| Ctrl+O | Open file dialog | |
| Ctrl+S | Save session | |
| Ctrl+E | Export dialog | |
| Ctrl+Z | Undo | |
| Ctrl+Y | Redo | |
| Delete | Delete selected peak | |
| ESC | Return to previous view level | |
| P | Toggle Pan Mode | |
| Z | Toggle Zoom Mode | |
| E (View) | Toggle event markers | |
| L | Toggle log panel | |
| Ctrl+Plus | Zoom in | |
| Ctrl+Minus | Zoom out | |
| Home | Jump to start | |
| End | Jump to end | |
| Ctrl+T | Jump to Time dialog | |

**Notes**:

---

## 15. Performance

- [ ] 15.1 **Large file load**: Load an XDF file > 1 hour @ 1000 Hz -- completes in < 10 seconds
  - Actual load time: __________ s
- [ ] 15.2 **Multi-signal scroll**: Scrolling/panning through a large file remains smooth (>30 FPS, no freezing)
- [ ] 15.3 **Zoomed out**: At maximum zoom-out (entire recording visible), rendering is fast -- no visible stutter
- [ ] 15.4 **Peak add/delete latency**: Peak markers appear immediately on double-click (no perceptible delay)

**Notes**:

---

## 16. Reproducibility

- [ ] 16.1 Load file -> apply bandpass filter (note parameters) -> detect peaks -> export CSV
- [ ] 16.2 Close the app and reopen it
- [ ] 16.3 Repeat the same file load + same filter parameters + same peak detection
- [ ] 16.4 Exported CSV values are identical to the first run (check a few rows)

**Notes**:

---

## Summary

| Section | Total Items | Passed | Failed | Skipped |
|---------|-------------|--------|--------|---------|
| 1. App Launch | 6 | | | |
| 2. XDF Loading | 8 | | | |
| 3. CSV Loading | 6 | | | |
| 4. Load Error Handling | 3 | | | |
| 5. Multi-Signal View | 11 | | | |
| 6. View Hierarchy | 9 | | | |
| 7. Filters | 7 | | | |
| 8. EEMD | 5 | | | |
| 9. Peak Detection | 7 | | | |
| 10. Peak Correction | 12 | | | |
| 11. Export | 6 | | | |
| 12. Session Save/Load | 5 | | | |
| 13. Log Panel | 6 | | | |
| 14. Keyboard Shortcuts | 16 | | | |
| 15. Performance | 4 | | | |
| 16. Reproducibility | 4 | | | |
| **Total** | **115** | | | |

### Bugs Found

| # | Section | Description | Severity (High/Med/Low) |
|---|---------|-------------|------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### Overall Result

- [ ] **PASS** -- no high-severity failures, app is ready for use
- [ ] **PASS WITH KNOWN ISSUES** -- minor failures documented above
- [ ] **FAIL** -- high-severity failures block use; fixes required before use
