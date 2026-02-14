# Setup - CardioSignalLab

## Environment

- **Python**: 3.12
- **Environment Manager**: Miniconda
- **Conda Environment**: TBD (currently using `ekgpeakcorrector`)
- **Platform**: Windows 11 with WSL2 for development tooling
- **Python Executable**: `/mnt/c/Users/sayee/miniconda3/envs/ekgpeakcorrector/python.exe` (temporary)

## Dependencies

### Core
- numpy, pandas, scipy
- matplotlib
- neurokit2
- pyxdf
- customtkinter
- loguru

### Processing
- PyEMD (for EEMD denoising)
- scikit-learn

### Dev
- pytest

## Project Structure

```
CardioSignalLab/
  CLAUDE.md
  src/cardio_signal_lab/
    core/          # Data models, file loading, signal containers
    gui/           # Menu-driven single-window GUI
    config/        # Settings, styles
    processing/    # Filters, artifact detection, EEMD, NeuroKit wrappers
  tests/
  scripts/
  docs/
    ARCHITECTURE.md
    PROJECT_STATUS.md
    SETUP.md
```
