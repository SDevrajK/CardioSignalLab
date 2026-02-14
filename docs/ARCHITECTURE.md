# Architecture - CardioSignalLab

## System Overview

Menu-driven desktop application for physiological signal viewing, artifact detection/removal, and peak correction. Supports ECG and PPG signals independently or together.

## Application Model

Single-window, document-centric design:
1. Launch: empty window with menu bar
2. File > Open: load XDF file, display signal
3. Signal menu: apply preprocessing/filtering operations
4. Analysis menu: peak detection, peak correction mode
5. Export menu: save results (NN intervals, cleaned signal)

## Menu Structure (Planned)

```
File
  Open XDF...
  Open CSV...
  Save Signal...
  Export NN Intervals...
  Exit

Signal
  Channel Select >
  Preprocessing
    NeuroKit
      ppg_clean
      ecg_clean
    Filters
      High-pass...
      Low-pass...
      Bandpass (Butterworth)...
      Notch (50/60 Hz)...
    Advanced
      EEMD Denoising
      Baseline Wander Removal
  Spike Removal
    Median Filter
    Gradient-based

Analysis
  Peaks
    Detect
    Correct (enters peak correction mode)
  Quality Assessment

View
  Zoom to Fit
  Show/Hide Peaks
  Show/Hide Artifacts
```

## Module Architecture

### core/ - Data layer
- `data_models.py` - Signal containers (from v2, adapted for PPG-standalone)
- `xdf_loader.py` - XDF file loading with timestamp repair (from v2)
- `signal_container.py` - Undo/redo operation history

### gui/ - Presentation layer
- `app.py` - Main window, menu bar, canvas management
- `signal_view.py` - Signal plotting with zoom/pan (matplotlib)
- `peak_correction_mode.py` - Interactive peak add/remove/move (from v2 PeakCorrectionHandler)

### processing/ - Operations layer
- `filters.py` - Basic filters: high-pass, low-pass, bandpass, notch (from v2 SignalFilter)
- `neurokit_wrappers.py` - ppg_clean, ecg_clean, ppg_process wrappers
- `eemd_denoiser.py` - EEMD decomposition and reconstruction (from Shimmer_Testing)
- `artifact_detector.py` - Spike detection, amplitude outliers (from v2 ArtifactReducer)
- `peak_detector.py` - Peak detection for ECG and PPG

### config/ - Configuration
- `settings.py` - App-wide settings
- `styles.py` - UI colors, plot styles

## Data Flow

```
XDF/CSV --> Loader --> SignalContainer --> [Processing Operations] --> Display
                            |                                           |
                       undo/redo stack                          peak correction
                                                                        |
                                                                    Export
```

## Key Design Decisions

1. **Single window, not tabs** - Menu-driven operations on one canvas
2. **Operation stack** - Every processing step is undoable
3. **PPG-first capable** - No assumption that ECG exists
4. **Matplotlib for plotting** - With downsampling for large signals if needed
5. **CustomTkinter + tk.Menu** - No library change needed from v2

## Reuse from EKG Peak Corrector v2

| v2 Module | New Location | Changes Needed |
|-----------|-------------|----------------|
| `core/data_models.py` | `core/data_models.py` | Remove ECG-required assumption |
| `core/xdf_loader.py` | `core/xdf_loader.py` | Minimal |
| `core/signal_filtering.py` | `processing/filters.py` | Extract, simplify |
| `core/signal_filtering.py` (ArtifactReducer) | `processing/artifact_detector.py` | Extract |
| `gui/peak_correction_handler.py` | `gui/peak_correction_mode.py` | Adapt to single-window |
| `core/peak_processor.py` | `processing/peak_detector.py` | Keep |
| `core/ecg_processor.py` | `processing/peak_detector.py` | Merge ECG detection |
| `core/ppg_processor.py` | `processing/peak_detector.py` | Merge PPG detection |
| `core/nn_interval_exporter.py` | `core/` or `processing/` | Keep as-is |

## New Code Needed

1. **`gui/app.py`** - Menu-bar-driven main window (replaces v2's tab-based main_window.py)
2. **`gui/signal_view.py`** - Single-canvas signal viewer with smart downsampling
3. **`core/signal_container.py`** - Operation history with undo/redo
4. **`processing/eemd_denoiser.py`** - Adapted from Shimmer_Testing
5. **`processing/neurokit_wrappers.py`** - Thin wrappers around nk functions
