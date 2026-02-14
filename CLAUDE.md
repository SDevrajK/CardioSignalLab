# CLAUDE.md

## Project Overview

CardioSignalLab is a desktop application for viewing, processing, and correcting physiological signals (ECG, PPG, EDA) from XDF and CSV files. It provides artifact detection/removal, signal filtering, and interactive peak correction in a menu-driven single-window Qt interface.

## Architecture

- **Python**: 3.12 (Miniconda environment)
- **GUI**: PySide6 (QMainWindow with QMenuBar, LGPL-licensed)
- **Plotting**: PyQtGraph with level-of-detail rendering (min/max envelopes)
- **Data Models**: attrs with validators for scientific data containers
- **Signal Processing**: NeuroKit2, SciPy, PyEMD (background-threaded via QThread)
- **Data Format**: XDF files via pyxdf, CSV; JSON session files for save/resume
- **Architecture**: Signal/slot event bus, FileLoader Protocol, composable processing pipeline
- **Tooling**: ruff (lint/format), pytest + pytest-qt (testing)

**CRITICAL**: Use `/mnt/c/Users/sayee/miniconda3/envs/ekgpeakcorrector/python.exe` for ALL Python commands until a dedicated environment is created.

## Project Structure

```
CardioSignalLab/
  src/cardio_signal_lab/
    core/          # Data models, file loading, signal containers
    gui/           # Menu-driven single-window GUI
    config/        # Settings, styles, keybindings
    processing/    # Filters, artifact detection, EEMD, NeuroKit wrappers
  tests/
  scripts/
  docs/
```

## Source Projects

Code is being selectively ported from:
- **EKG_Peak_Corrector v2** (`../EKG_Peak_Corrector/v2/`): Core processing, data models, peak correction
- **Shimmer_Testing** (`../Shimmer_Testing/lib/preprocessing/emd_denoising.py`): EEMD artifact removal
- **Acute_Tinnitus_PPG / Hyperacousie_TCC**: NeuroKit PPG processing patterns

## Development Guidelines

- Do not use Unicode characters, use ASCII instead
- No over-engineering; minimal code
- Use loguru for logging
- Hardcode paths for development; parameterize later

## Current Session Context
<!-- Auto-managed by /save-context -->

**Last Updated**: 2026-02-13
**Session Focus**: Visualization implementation and PyQtGraph rendering bug resolution
**Project Status**: Active development - visualization working, ready for signal processing tasks

### Critical Bug RESOLVED ✓
**Problem**: PlotDataItem and ScatterPlotItem did not render in PyQtGraph plots
- **Root Cause**: PySide6 6.9.1 introduced a rendering bug that breaks PyQtGraph
- **Solution**: Downgraded PySide6 from 6.9.1 → 6.9.0
- **Source**: Found forum post describing identical issue with same versions
- **Result**: All plots now render correctly with full functionality

**Command to maintain fix**: `pip install pyside6==6.9.0` (do not upgrade to 6.9.1)

### UI Improvement Identified
**Issue**: With multiple stacked signals (6+ channels), plot titles above each plot consume too much vertical space
**Suggested fix**: Move titles to y-axis labels instead of above plots
**Status**: Documented for future UI polish phase

### Completed This Session
1. ✅ Implemented Parent Task 3.0: Visualization system
   - MultiSignalView with synchronized stacked plots
   - SingleSignalView for focused signal viewing
   - PeakOverlay for interactive peak markers
   - LOD rendering integration
2. ✅ Fixed infinite feedback loop in ViewBox range updates
3. ✅ Debugged and resolved PyQtGraph rendering issue
4. ✅ Cleaned up debug code and restored production settings
5. ✅ Re-enabled x-axis linking for synchronized zoom/pan

### Next Priority
**Task 3.9**: Complete Parent Task 3.0 - Advanced Navigation Controls
- Wire Ctrl+Plus/Minus keyboard shortcuts for programmatic zoom in/out
- Add "Jump to Time" dialog in View menu
- Implement next/previous peak navigation (arrow keys when peak selected)
- Add zoom to selection (drag region, right-click → Zoom to Selection)
- Add Home/End keys to jump to start/end of signal
- Ensure all features work in both multi-signal and single-signal modes

**After 3.9 Complete**:
**Parent Task 4.0**: Signal Processing Pipeline
- Filtering (bandpass, notch)
- Peak detection (ECG, PPG, EDA)
- Artifact removal (EEMD)
- NeuroKit2 integration

### Files Modified
- `src/cardio_signal_lab/gui/plot_widget.py` - Rendering fixes, debug cleanup
- `src/cardio_signal_lab/gui/multi_signal_view.py` - X-axis linking restored
- `test_pyqtgraph_minimal.py`, `test_simple.py` - Diagnostic tests (can be moved to tests/)
