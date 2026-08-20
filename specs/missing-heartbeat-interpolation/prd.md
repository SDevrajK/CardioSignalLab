# PRD: Missing Heartbeat Interpolation

**Status:** Draft
**Created:** 2026-06-12
**Slug:** missing-heartbeat-interpolation

## 1. Problem & Motivation
In physiological signal analysis, noisy data sections often result in missing heartbeat detections. While simply dropping these long intervals works adequately for calculating mean heart rate (meanNN), it severely degrades variability metrics like RMSSD, which rely on successive beat-to-beat differences. To maintain the integrity of HRV metrics, the application needs a way to detect these gaps and intelligently interpolate the missing heartbeats based on established literature methods.

## 2. Goals
- Accurately detect gaps in the heartbeat series using a combination of absolute and dynamic thresholds.
- Visually highlight detected gaps on the signal plot so the user can review them before modification.
- Provide multiple literature-backed interpolation methods to fill missing beats.
- Protect data integrity by refusing to interpolate excessively large gaps where interpolation becomes unreliable.
- Visually distinguish interpolated peaks from normally detected peaks.

## 3. Non-Goals
- Real-time or automatic gap filling during the initial file load or peak detection phase (it must be a deliberate, user-initiated two-step process).
- Selective filling of individual gaps (for this initial implementation, all valid detected gaps are filled at once).
- Implementation of complex model-based (e.g., IPFM) or machine-learning-based interpolation methods.

## 4. User Stories / Use Cases
- As a researcher, I want to detect missing heartbeats so that I can see where the signal quality degraded.
- As a researcher, I want to review the detected gaps by panning through the signal before committing to filling them, so that I can ensure the detection logic didn't misidentify normal physiological variations.
- As a researcher, I want to choose between different interpolation methods (Linear, Pchip, Nearest-neighbor) so that I can apply the most appropriate correction for my specific HRV analysis goals.
- As a researcher, I want interpolated peaks to look slightly different from normal peaks so that I can visually verify where synthetic data was introduced.

## 5. Functional Requirements
1. **Menu Integration:** Add two new items under `Analysis > Peaks`: "Detect Missing Beats" and "Fill Missing Beats".
2. **Gap Detection Logic:** When "Detect Missing Beats" is clicked, the system must identify gaps. A gap is defined as any RR interval that is either longer than 2.0 seconds OR greater than 1.5x the median of the surrounding 10 beats (5 before, 5 after).
3. **Gap Visualization:** Detected gaps must be visually highlighted on the signal plot (e.g., with a background shading or horizontal span marker) so the user can pan and review them.
4. **Fill Configuration:** When "Fill Missing Beats" is clicked, a dialog must appear allowing the user to select the interpolation method from a dropdown: "Linear", "Non-linear (Pchip)", and "Nearest-neighbor".
5. **Interpolation Execution:** Upon confirming the dialog, the system must interpolate missing beats for all detected gaps using the selected method. Interpolation must be performed on the timestamps (time domain), not on the RR interval durations.
6. **Gap Duration Limit:** The system must automatically refuse to interpolate any gap that is longer than a configurable threshold (default 10 seconds). These gaps must remain empty.
7. **Peak Visualization:** Interpolated peaks must be added to the plot using the same color as normal peaks but with a different marker shape (e.g., square instead of circle).
8. **Undo/Redo:** The gap filling operation must be added to the operation stack so it can be undone.

## 6. Non-Functional Requirements
- **Performance:** Gap detection and interpolation should compute in under 1 second for typical file lengths to maintain UI responsiveness.

## 7. Constraints & Dependencies
- Must integrate with the existing `SignalContainer` and undo/redo architecture.
- Must use existing plotting libraries (PyQtGraph/matplotlib as currently implemented in the views).
- Interpolation algorithms (like Pchip) may require specific SciPy functions (`scipy.interpolate.PchipInterpolator`).

## 8. Acceptance Criteria
- [ ] Clicking `Analysis > Peaks > Detect Missing Beats` highlights gaps on the plot based on the `> 2s OR > 1.5x median of 10` rule.
- [ ] The user can pan and zoom the plot to review the highlighted gaps without the gaps disappearing.
- [ ] Clicking `Analysis > Peaks > Fill Missing Beats` opens a dialog with a dropdown for "Linear", "Non-linear (Pchip)", and "Nearest-neighbor".
- [ ] Executing the fill operation populates the gaps with new peaks, calculated by interpolating timestamps.
- [ ] Gaps longer than 10 seconds are ignored and left empty during the fill operation.
- [ ] Interpolated peaks are visually distinct (e.g., square markers) from normal peaks (e.g., circle markers) on the plot.
- [ ] The fill operation can be successfully undone via the application's undo mechanism.

## 9. Assumptions
- The user has already run the standard peak detection before attempting to detect or fill missing beats.
- The surrounding 10 beats used for the median calculation are the 5 valid beats immediately preceding and the 5 valid beats immediately following the interval in question. If near the start/end of the signal, it uses whatever beats are available up to 10.

## 10. Out-of-Scope Follow-ups
- Selective filling of individual gaps (e.g., clicking a specific gap to fill only that one).
- Interactive adjustment of the gap detection thresholds (2s / 1.5x) via the UI.