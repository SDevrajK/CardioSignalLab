# PRD: CardioSignalLab MVP - Core Application

## Introduction/Overview

CardioSignalLab is a desktop application for viewing, processing, and correcting physiological signals (ECG, PPG, EDA) from multi-modal recordings. The application provides a complete signal processing pipeline with interactive peak correction in a menu-driven single-window interface. This PRD defines the Minimum Viable Product (MVP) covering the full workflow from file loading through artifact removal, filtering, peak detection, manual correction, and export.

**Problem it solves**: Researchers need a unified tool to process physiological signals with manual peak correction capabilities, combining automated processing with visual inspection and manual refinement for accurate analysis.

## Goals

- Provide a complete signal processing pipeline: Load → Filter → Detect peaks → Correct peaks → Export
- Support ECG, PPG, and EDA signals from multiple input formats with unified processing
- Enable fast, smooth interactive peak addition/deletion with visual feedback
- Implement EEMD artifact removal for signal cleaning
- Export processed signals and peak annotations in multiple formats
- Maintain processing reproducibility with saved parameters

## User Stories

- As a researcher, I want to load XDF files or CSV files containing multi-modal physiological recordings so that I can process ECG, PPG, and EDA signals in one application
- As a researcher, I want to apply EEMD artifact removal to clean noisy signals so that automated peak detection is more accurate
- As a researcher, I want to apply bandpass filters and preprocessing to raw signals so that I can prepare them for peak detection
- As a researcher, I want to automatically detect R-peaks (ECG), pulse peaks (PPG), and relevant features in EDA so that I can start with an automated baseline
- As a researcher, I want to visually inspect detected peaks on the signal plot so that I can identify detection errors
- As a researcher, I want to add missing peaks by clicking on the signal so that I can correct false negatives quickly
- As a researcher, I want to delete incorrect peaks by selecting them so that I can remove false positives efficiently
- As a researcher, I want peak addition/deletion to feel smooth and responsive so that manual correction doesn't become tedious
- As a researcher, I want to export processed signals with corrected peak annotations so that I can use the data in downstream analysis
- As a researcher, I want to export to multiple formats (CSV, NPY, XDF) so that I can integrate with different analysis pipelines
- As a researcher, I want processing to handle problematic data segments gracefully so that one bad segment doesn't stop my entire workflow

## Functional Requirements

### 1. File Loading and Signal Detection
1.1. The system must load XDF files and parse embedded physiological signals
1.2. The system must support loading other formats (CSV)
1.3. The system must automatically detect signal types (ECG, PPG, EDA) based on metadata or allow manual selection
1.4. The system must display signal metadata (sampling rate, duration, channel names) upon loading
1.5. The system must process signals with different sampling rates independently (no automatic resampling)

### 2. Signal Preprocessing and Filtering
2.1. The system must provide bandpass filtering with configurable cutoff frequencies
2.2. The system must support baseline correction and detrending
2.3. The system must implement EEMD (Ensemble Empirical Mode Decomposition) artifact removal
2.4. The system must allow users to preview filter effects before applying
2.5. The system must maintain processing parameters for reproducibility via a composable processing pipeline (ordered list of ProcessingStep records that can be serialized and replayed)
2.6. The system must run long-running processing operations (EEMD, peak detection) in a background thread with a progress indicator, keeping the GUI responsive and providing a cancel option

### 3. Peak Detection
3.1. The system must automatically detect R-peaks in ECG signals using NeuroKit2
3.2. The system must automatically detect pulse peaks in PPG signals using NeuroKit2
3.3. The system must detect relevant features in EDA signals (SCR onsets/peaks)
3.4. The system must display detected peaks as visual markers overlaid on signal plots
3.5. The system must provide peak detection quality metrics (if available from NeuroKit2)

### 4. Interactive Peak Correction
4.1. Users must be able to add peaks by clicking on the signal plot
4.2. Users must be able to delete peaks by selecting them (click or keyboard shortcut)
4.3. Peak addition/deletion must update the visualization in real-time (<100ms latency)
4.4. The system must support undo/redo for peak corrections with up to 20 levels of history
4.5. Undo/redo history must reset when a new file is loaded (does not persist across sessions)
4.6. The system must visually distinguish between auto-detected and manually-added peaks
4.7. The system must allow zooming and panning during peak correction

### 5. Visualization
5.1. The system must use PyQtGraph for fast, interactive signal plotting
5.2. In multi-signal mode, the system must display all signals in synchronized plot widgets
5.3. In single-signal mode, the system must display the selected signal prominently (full window or emphasized view)
5.4. The system must support zooming, panning, and time-range selection in both modes
5.5. In multi-signal mode, zoom/pan operations must be synchronized across all plot widgets
5.6. The system must provide a time axis synchronized across all signal views
5.7. The system must visually indicate the current mode (e.g., window title, border, or status bar)
5.8. The system must implement level-of-detail (LOD) rendering for large datasets: precompute min/max envelopes at multiple zoom levels so that zoomed-out views render fast summaries while zoomed-in views show full-resolution data; must handle 10M+ data points smoothly

### 6. Export and Session Persistence
6.1. The system must export processed signals in CSV format (time, signal_value, peaks)
6.2. The system must export processed signals in NumPy (.npy) format
6.3. The system must export corrected peak times as separate annotations files
6.4. The system must optionally export back to XDF format with corrected annotations
6.5. The system must save processing parameters alongside exported data for reproducibility
6.6. File -> Save must save the current session to a JSON session file containing: source file path, processing pipeline (ordered list of steps and parameters), corrected peak data, and current view state; File -> Open must support reopening session files to resume work

### 7. Error Handling
7.1. The system must warn users about problematic data segments (e.g., signal dropout, extreme values)
7.2. The system must allow processing to continue when skipping bad segments
7.3. The system must log warnings and errors to a visible console or log panel
7.4. The system must validate file formats before attempting to load

### 8. Menu-Driven Interface

The application operates in two modes with dynamic menus that adapt to the current context:

#### Multi-Signal Mode (Default after file load)
8.1. Upon loading a file with multiple signals, the application starts in multi-signal mode
8.2. Menu bar in multi-signal mode: **File, Edit, Select, View, Help**
8.3. All signals displayed simultaneously in synchronized subplots
8.4. Select menu allows user to choose which signal to work with (ECG, PPG, EDA, etc.)
8.5. Processing operations are disabled in multi-signal mode

#### Single-Signal Mode (After selecting a signal)
8.6. After selecting a signal via the Select menu, application enters single-signal mode
8.7. Menu bar in single-signal mode: **File, Edit, Process, View, Help**
8.8. Select menu is replaced by Process menu with signal-specific options
8.9. Process menu items dynamically adapt to the selected signal type:
   - For ECG: Filter, Artifact Removal, Detect R-Peaks, Reset Processing
   - For PPG: Filter, Artifact Removal, Detect Pulse Peaks, Reset Processing
   - For EDA: Filter, Detect SCR, Reset Processing
8.10. View menu includes option to "Return to Multi-Signal View" to switch back to multi-signal mode

#### Common Menu Items (Both Modes)
8.11. File menu must include: Open, Save, Export, Recent Files, Exit
8.12. Edit menu must include: Add Peak, Delete Peak, Undo, Redo (only enabled in single-signal mode)
8.13. View menu must include: Zoom In/Out, Pan, Reset View, Show/Hide Peaks
8.14. Help menu must include: About, Documentation, Keyboard Shortcuts

#### Status Bar Feedback
8.15. Status bar at bottom of window displays current mode and selected signal
8.16. Status bar shows context like "Multi-Signal Mode (3 signals loaded)" or "Single-Signal Mode: ECG (Channel 1)"
8.17. Status bar updates when mode changes or signal selection changes

## Non-Goals (Out of Scope)

- This feature will NOT support real-time streaming data processing (offline analysis only)
- This feature will NOT include real-time parameter adjustment during visualization (filter parameters set before applying)
- This feature will NOT include advanced HRV analysis or spectral analysis in MVP
- This feature will NOT support batch processing multiple files in MVP
- This feature will NOT include machine learning-based peak detection in MVP
- This feature will NOT implement auto-save for peak corrections (manual save only)
- This feature will NOT support collaborative editing or cloud storage
- This feature will NOT include a plugin system or extensibility API in MVP
- We are NOT implementing multi-user or database backend features
- We are NOT building mobile or web versions

## Design Considerations

### User Interface
- Single-window Qt application (QMainWindow) with native menu bar (QMenuBar)
- PySide6 for modern, cross-platform GUI widgets
- PyQtGraph for fast, interactive signal visualization
- QStatusBar at bottom showing current mode and selected signal
- Minimal clutter: no unnecessary toolbars, clean interface
- Keyboard shortcuts for common operations (use reasonable defaults, defined in config/keybindings):
  - Ctrl+Z: Undo
  - Ctrl+Y: Redo
  - Delete: Remove selected peak
  - Ctrl+O: Open file
  - Ctrl+S: Save
  - Ctrl+E: Export
  - ESC: Return to multi-signal view

### Two-Mode System
**Multi-Signal Mode:**
- All loaded signals displayed in synchronized subplots
- Limited menus: File, Edit, Select, View, Help
- User explores data and selects which signal to process
- No peak correction or processing operations available

**Single-Signal Mode:**
- Selected signal displayed prominently (full window or emphasized)
- Full menus: File, Edit, Process, View, Help
- Process menu dynamically populated with signal-specific operations
- Peak correction and all processing operations available
- Can return to multi-signal mode via View menu

### Interaction Model
**Multi-Signal Mode:**
- Zoom/pan synchronized across all signals
- Click on signal or use Select menu to enter single-signal mode
- View metadata and explore data

**Single-Signal Mode:**
- Click to add peak: single left-click on signal
- Click to delete peak: single left-click on existing peak marker
- Right-click for context menu (optional)
- Scroll wheel for zoom, drag for pan
- Keyboard: Arrow keys for navigation, Delete for peak removal, Ctrl+Z for undo
- Return to multi-signal view: View → "Return to Multi-Signal View" or keyboard shortcut

### Visual Design
- Auto-detected peaks: one color (e.g., blue)
- Manually-added peaks: different color (e.g., green)
- Deleted peaks: marked for undo buffer but removed from view
- Signal: high-contrast colors for accessibility
- Grid and axis labels: clear and legible

## Technical Considerations

### Dependencies
- Python 3.12 (Miniconda environment)
- PySide6 (GUI framework -- official Qt for Python, LGPL-licensed)
- PyQtGraph (fast interactive plotting for signal visualization)
- attrs (data model validation -- lightweight validators for scientific data containers)
- NeuroKit2 (signal processing and peak detection)
- PyEMD (EEMD artifact removal)
- pyxdf (XDF file loading)
- SciPy (filtering and signal processing utilities)
- NumPy (numerical operations)
- loguru (logging)
- ruff (linting and formatting)
- pytest / pytest-qt (testing framework with Qt GUI test support)

### Architecture Patterns
- **Signal/Slot Event Bus**: Central `AppSignals` QObject with custom signals (file_loaded, signal_selected, mode_changed, peaks_updated, processing_started, processing_finished) to decouple GUI components from processing logic
- **FileLoader Protocol**: Abstract loader interface (`can_load(path) -> bool`, `load(path) -> RecordingSession`) implemented by `XdfLoader` and `CsvLoader`; new formats added by implementing the protocol
- **Composable Processing Pipeline**: Processing state represented as ordered list of `ProcessingStep` records (operation name + parameters dict); "Reset Processing" clears the pipeline; serializing the pipeline provides automatic reproducibility
- **Background Processing**: QThread workers for long-running operations (EEMD, peak detection) with progress signals and cancel support; GUI remains responsive during processing
- **Level-of-Detail Rendering**: Precomputed min/max envelope pyramids at multiple zoom levels; zoomed-out views render fast summaries, zoomed-in views show full-resolution data
- **Data Model Validation**: Use `attrs` with validators for scientific data containers (enforce positive sampling rates, 1D signal arrays, monotonic timestamps)

### Performance Constraints
- Peak addition/deletion must be smooth (<50ms response time with PyQtGraph)
- Large files (>1 hour of data at 1000 Hz) should load within 10 seconds
- LOD rendering should handle multi-million data point datasets smoothly (10M+ points)
- Zooming and panning must remain responsive even with large datasets via LOD envelope switching
- Multi-signal view with 3+ synchronized plots must maintain >30 FPS during interaction

### Code Porting Strategy
- Port interactive peak correction logic from `EKG_Peak_Corrector v2`
- Port EEMD artifact removal from `Shimmer_Testing/lib/preprocessing/emd_denoising.py`
- Adapt NeuroKit2 processing patterns from `Acute_Tinnitus_PPG` and `Hyperacousie_TCC`
- Follow development guidelines: minimal code, no over-engineering, remove legacy code

### Environment
- Development path: `/mnt/c/Users/sayee/miniconda3/envs/ekgpeakcorrector/python.exe`
- Hardcode paths during development; parameterize before release
- Use ASCII characters only (no Unicode)
- Use loguru for all logging

## Success Metrics

- **Visual Accuracy**: Corrected peaks visually align with signal morphology upon manual inspection
- **Export Completeness**: All processed signals and peak times export correctly in expected formats (CSV, NPY, annotations)
- **Workflow Efficiency**: Typical file processing (load → filter → detect → correct → export) completes in under 5 minutes
- **Interaction Responsiveness**: Peak addition/deletion updates visualization in <100ms
- **Reproducibility**: Same input file + same processing parameters = identical output across runs

## Notes

- This PRD defines the MVP for CardioSignalLab. Additional features (HRV analysis, batch processing, ML-based detection) will be scoped in future PRDs.
- User is the primary user (research workflow), so assumptions about expertise and error handling can be tailored accordingly.
- Follow development guidelines from CLAUDE.md: minimal code, prioritize conciseness, use descriptive naming, document scientific rationale for signal processing choices.
- All initial design questions have been resolved and integrated into the functional requirements and design considerations above.
