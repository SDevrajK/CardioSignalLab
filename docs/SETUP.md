# Setup - CardioSignalLab

## Environment

- **Python**: 3.12 (python.org or Miniconda)
- **GUI**: PySide6 (Qt6, LGPL)
- **Platform**: Windows, macOS, Linux

## Quick Start (pip only)

```
pip install -e .
cardio-signal-lab
```

## Quick Start (conda)

```
conda env create -f environment.yml
conda activate cardio-signal-lab
pip install -e .
cardio-signal-lab
```

## Dependencies

### Core (installed via environment.yml + pip install)
| Package | Purpose |
|---------|---------|
| PySide6 | Qt6 GUI framework |
| pyqtgraph | Signal plotting with LOD rendering |
| numpy, pandas, scipy | Numerical computing and signal processing |
| attrs | Data model definitions |
| neurokit2 | ECG/PPG/EDA peak detection |
| PyEMD | EEMD artifact removal |
| pyxdf | XDF file loading (LSL format) |
| loguru | Logging |

### Dev (pip install -e ".[dev]")
| Package | Purpose |
|---------|---------|
| pytest | Test runner |
| pytest-qt | Qt widget testing |
| pytest-cov | Coverage reporting |
| ruff | Linting and formatting |

## Project Structure

```
CardioSignalLab/
  main.py                    # Dev entry point (python main.py)
  pyproject.toml             # Package config and entry point
  environment.yml            # Conda environment spec
  requirements.txt           # Pinned dependency versions
  src/cardio_signal_lab/
    app.py                   # Installed entry point (cardio-signal-lab)
    core/                    # Data models, file loading, export, session
    gui/                     # Main window, views, dialogs, overlays
    config/                  # Settings, keybindings
    processing/              # Filters, EEMD, peak detection, pipeline
  tests/                     # pytest unit and integration tests
  docs/                      # Architecture, status, PRD, task lists
```

## Running Tests

```
pytest                        # All tests with coverage
pytest -m "not slow"          # Skip slow tests
pytest tests/test_filters.py  # Single test file
```

## Linting

```
ruff check src/
ruff format src/
```

## Development Notes

- Entry point for installed package: `cardio_signal_lab.app:main`
- Dev entry point (no install needed): `python main.py`
- Logs written to `cardio_signal_lab.log` (rotating, 10 MB, 7 days)
- Log panel in app: View menu or press `L`
- XDF files require stream names containing "ECG", "PPG", or "EDA" for auto-detection
