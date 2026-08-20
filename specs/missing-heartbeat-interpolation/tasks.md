# Tasks: Missing Heartbeat Interpolation

**PRD:** specs/missing-heartbeat-interpolation/prd.md
**Generated:** 2026-06-12
**Status:** Not started

---

## Task 1: Integrate Missing Beat Detection and Fill Menu Items
**Covers PRD:** §5.1, AC 47, 49

- [ ] **1.1 [implementation]** Add "Detect Missing Beats" and "Fill Missing Beats" menu items to the `Analysis > Peaks` menu
- [ ] **1.2 [review]** Verify that clicking "Detect Missing Beats" and "Fill Missing Beats" triggers appropriate actions and that menu items are correctly placed

## Task 2: Implement Gap Detection Logic
**Covers PRD:** §5.2, §5.6, AC 47, 48, 51

- [ ] **2.1 [research]** Investigate how RR intervals are stored and accessed within `SignalContainer`
- [ ] **2.2 [implementation]** Implement gap detection based on RR intervals > 2.0 seconds OR > 1.5x median of surrounding 10 beats
- [ ] **2.3 [implementation]** Store detected gaps in a way that supports quick access and visualization
- [ ] **2.4 [review]** Verify that gaps are correctly detected and stored based on PRD requirements

## Task 3: Visualize Detected Gaps on Signal Plot
**Covers PRD:** §5.3, AC 47, 48

- [ ] **3.1 [implementation]** Implement visualization of detected gaps on signal plot using PyQtGraph
- [ ] **3.2 [implementation]** Ensure gaps remain visible when panning and zooming the plot
- [ ] **3.3 [review]** Verify that gaps are visually highlighted on the plot and remain visible during interaction

## Task 4: Implement Fill Missing Beats Dialog
**Covers PRD:** §5.4, §5.5, AC 49, 50, 52

- [ ] **4.1 [implementation]** Create a dialog for configuring interpolation method with options: "Linear", "Non-linear (Pchip)", and "Nearest-neighbor"
- [ ] **4.2 [implementation]** Validate interpolation methods with SciPy functions where required (e.g., PchipInterpolator)
- [ ] **4.3 [implementation]** Implement interpolation logic that operates on timestamps rather than RR durations
- [ ] **4.4 [review]** Confirm that dialog opens correctly and interpolation methods work as expected

## Task 5: Implement Gap Filling with Visual Distinction
**Covers PRD:** §5.7, §5.8, AC 52, 53

- [ ] **5.1 [implementation]** Add interpolated peaks to signal plot with distinct marker shape from normal peaks
- [ ] **5.2 [implementation]** Ensure interpolated peaks use the same color as normal peaks but with different marker style (e.g., square marker)
- [ ] **5.3 [implementation]** Integrate gap filling operation into undo/redo stack
- [ ] **5.4 [review]** Verify that interpolated peaks are visually distinct and operation is undoable

## Task 6: Add Gap Duration Limit
**Covers PRD:** §5.6, AC 51

- [ ] **6.1 [implementation]** Implement threshold check for gap duration (10 seconds default)
- [ ] **6.2 [implementation]** Ensure gaps longer than threshold are not filled and remain empty
- [ ] **6.3 [review]** Verify that operation correctly skips filling gaps exceeding the 10-second limit
