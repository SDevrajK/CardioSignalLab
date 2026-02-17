# Project Status - CardioSignalLab

## Current Phase: Planning

**Last Updated**: 2026-02-12

## Background

This project was born from a brainstorm session about evolving EKG Peak Corrector v2 into a more general-purpose physiological signal application. Key motivations:

- v2 was ECG-reconstruction focused; the new app needs PPG-standalone support
- v2 used a 3-tab wizard-style GUI; the new app uses a single-window menu-driven design
- Artifact detection/removal is the primary workflow, not reconstruction
- ~90% of v2's core processing code is reusable

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| New project vs update v2 | New project | v2 is complete, GUI is near-total rewrite, name doesn't fit |
| GUI library | CustomTkinter + tk.Menu | No reason to switch; menu bars work natively |
| Plotting | Matplotlib (with downsampling) | Reuses v2 code; consider pyqtgraph only if perf insufficient |
| Project name | CardioSignalLab | General enough for ECG + PPG |

## Planned Features

### Phase 1: Core Viewer
- [ ] Menu-bar-driven single window
- [ ] XDF file loading
- [ ] Signal display with zoom/pan
- [ ] Channel selection

### Phase 2: Processing
- [ ] Basic filters (high-pass, low-pass, bandpass, notch)
- [ ] NeuroKit wrappers (ppg_clean, ecg_clean)
- [ ] EEMD denoising (from Shimmer_Testing)
- [ ] Spike removal (median filter, gradient-based)
- [ ] Undo/redo for all operations

### Phase 3: Peak Correction
- [ ] Peak detection (ECG and PPG)
- [ ] Interactive peak correction mode
- [ ] NN/RR interval export

### Phase 4: Polish
- [ ] Smart downsampling for large signals
- [ ] Keyboard shortcuts
- [ ] CSV file loading
- [ ] Batch processing (optional)

### Future: Reconstruction Module
- [ ] Port v2 reconstruction as optional plugin

### Future: Processing Panel Improvements
- [ ] Per-step removal from processing pipeline: right-click context menu on the
      Processing Panel to remove any individual step and re-apply remaining steps
      from `_raw_samples`. Requires adding `remove_step(idx)` to `ProcessingPipeline`
      and making the panel list interactive. Structural ops (crop/resample) remain
      non-removable since the pre-operation signal is not held in memory.

## Source Code Locations

| Component | Source |
|-----------|--------|
| Core processing | `../EKG_Peak_Corrector/v2/src/ekg_corrector/core/` |
| EEMD denoiser | `../Shimmer_Testing/lib/preprocessing/emd_denoising.py` |
| NeuroKit patterns | `../Acute_Tinnitus_PPG/`, `../Hyperacousie_TCC/` |
