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

**Last Updated**: 2026-02-15
**Session Focus**: Export/session persistence and comprehensive error handling
**Project Status**: Active development - Tasks 6.0 and 7.0 (partial) complete

### Completed This Session
1. ✅ **Task 6.0**: Export and Session Persistence (ALL SUBTASKS COMPLETE)
   - Task 6.4: Wired export/session to GUI (File→Open .csl.json, File→Save, File→Export)
   - Task 6.5: 22 unit tests (test_exporter.py, test_session.py) - 100% coverage
   - Task 6.6: Verified 5/6 PRD requirements met (XDF re-export marked as future enhancement)
2. ✅ **Task 7.1**: Researched error handling patterns from EKG_Peak_Corrector
   - Documented 7 key patterns: layered validation, signal checks, gap detection, graceful fallback
3. ✅ **Task 7.2**: Added input validation to file loaders
   - Layered validation (type→exists→format), sampling rate validation (16-2000 Hz)
   - Timestamp monotonicity checks, signal quality validation (flat signals, NaN, outliers)
4. ✅ **Task 7.3**: Added error handling to processing pipeline
   - Pipeline skip_on_error parameter for partial results
   - Signal dropout detection utility (detect_signal_dropouts)
   - EEMD/peak detection error wrapping with graceful failures

### Next Priority
**Task 7.4**: Add GUI log panel
- QTextEdit dock widget with loguru sink
- Real-time display of warnings/errors
- Toggle visibility via View menu

**After 7.4**:
- Task 7.5: Integration tests (pytest-qt workflow tests)
- Task 7.6: Final review against all PRD requirements

### Key Session Notes
- 22 export/session tests passing with 100% coverage
- 5/6 PRD export requirements fully implemented (CSV, NPY, Annotations, Parameters, Session)
- XDF re-export (6.4) documented as optional future enhancement
- Layered validation pattern adopted from EKG_Peak_Corrector research
- Pipeline now supports graceful error handling with skip_on_error flag
- Signal dropout detection utility added for quality monitoring

### Files Modified This Session
- `src/cardio_signal_lab/gui/main_window.py` - Export/session GUI integration
- `src/cardio_signal_lab/core/exporter.py` - CSV/NPY/Annotations export (CREATED)
- `src/cardio_signal_lab/core/session.py` - Session save/load (CREATED)
- `src/cardio_signal_lab/core/file_loader.py` - Input validation
- `src/cardio_signal_lab/processing/pipeline.py` - Error handling
- `src/cardio_signal_lab/processing/filters.py` - Dropout detection utility
- `src/cardio_signal_lab/processing/eemd.py` - Error wrapping
- `src/cardio_signal_lab/processing/peak_detection.py` - Error wrapping
- `tests/test_exporter.py` - 11 tests (CREATED)
- `tests/test_session.py` - 11 tests (CREATED)
- `docs/tasks/review-task-6-export.md` - Review document (CREATED)
- `docs/research/error-handling-patterns.md` - Research notes (CREATED)
