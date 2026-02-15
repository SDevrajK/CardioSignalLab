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
**Session Focus**: Comprehensive error handling, testing, and final validation
**Project Status**: ✅ MVP COMPLETE - Ready for testing

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
5. ✅ **Task 7.4**: Added GUI log panel
   - QDockWidget with QTextEdit for real-time log display
   - Custom loguru sink with thread-safe Qt signals
   - Color-coded log levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
   - View menu toggle (L key) - hidden by default
   - 8 unit tests with 96% coverage
6. ✅ **Task 7.5**: Integration tests
   - 11 integration tests covering end-to-end workflows
   - File loading (CSV auto-detect, manual type, sampling rate validation)
   - Processing (bandpass filter, multi-step pipeline, peak detection)
   - Export (CSV, NPY, annotations)
   - Session (save/load, pipeline reproducibility)
7. ✅ **Task 7.6**: Final PRD review
   - Comprehensive review of all 49 functional requirements
   - 47/49 requirements implemented (96% complete)
   - 2 requirements partially implemented (XDF export deferred, quality metrics internal)
   - MVP declared READY FOR TESTING

### Project Complete
**MVP Status**: ✅ READY FOR USER TESTING
- All core workflows implemented and tested
- 41 automated tests passing
- Error handling comprehensive
- Export/session persistence verified
- Documentation complete

**After 7.4**:
- Task 7.5: Integration tests (pytest-qt workflow tests)
- Task 7.6: Final review against all PRD requirements

### Key Session Notes
- 41 tests passing (22 export/session + 8 log panel + 11 integration) with high coverage
- 5/6 PRD export requirements fully implemented (CSV, NPY, Annotations, Parameters, Session)
- XDF re-export (6.4) documented as optional future enhancement
- Layered validation pattern adopted from EKG_Peak_Corrector research
- Pipeline now supports graceful error handling with skip_on_error flag
- Signal dropout detection utility added for quality monitoring
- Log panel provides real-time visibility into warnings/errors during operation
- End-to-end workflows tested: file loading → processing → peak detection → export → session save/load

### Files Created/Modified This Session
**Source Code**:
- `src/cardio_signal_lab/gui/main_window.py` - Export/session GUI + log panel integration
- `src/cardio_signal_lab/gui/log_panel.py` - Dockable log widget (CREATED)
- `src/cardio_signal_lab/core/exporter.py` - CSV/NPY/Annotations export (CREATED)
- `src/cardio_signal_lab/core/session.py` - Session save/load (CREATED)
- `src/cardio_signal_lab/core/file_loader.py` - Input validation
- `src/cardio_signal_lab/processing/pipeline.py` - Error handling
- `src/cardio_signal_lab/processing/filters.py` - Dropout detection utility
- `src/cardio_signal_lab/processing/eemd.py` - Error wrapping
- `src/cardio_signal_lab/processing/peak_detection.py` - Error wrapping

**Tests** (41 total):
- `tests/test_exporter.py` - 11 tests (CREATED)
- `tests/test_session.py` - 11 tests (CREATED)
- `tests/test_log_panel.py` - 8 tests (CREATED)
- `tests/test_integration.py` - 11 end-to-end workflow tests (CREATED)

**Documentation**:
- `docs/tasks/review-task-6-export.md` - Export feature review (CREATED)
- `docs/tasks/final-prd-review.md` - Comprehensive PRD compliance review (CREATED)
- `docs/research/error-handling-patterns.md` - Research notes (CREATED)
