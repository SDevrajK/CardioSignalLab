# CardioSignalLab

Desktop application for viewing, processing, and correcting physiological signals (ECG, PPG, EDA) from XDF and CSV files.

## Features

- Load XDF (multi-stream) and CSV signal files
- Interactive signal viewer with zoom/pan
- Bandpass filter, notch filter, EEMD artifact removal
- Automatic peak detection (NeuroKit2) for ECG, PPG, and EDA
- Manual peak correction (add, delete, reclassify)
- Export peaks, annotations, and processed signals to CSV/NPY
- Session save/resume (.csl.json)

## Requirements

- Windows, macOS, or Linux
- Python 3.12 ([python.org](https://www.python.org/downloads/))

## Installation

### For colleagues (no git required)

If you have Anaconda or Miniconda, run these in your terminal:

```
conda create -n cardio-signal-lab python=3.12
conda activate cardio-signal-lab
pip install https://github.com/SDevrajK/CardioSignalLab/archive/refs/heads/main.zip
```

Then double-click `CardioSignalLab.bat` (download it from the repo) to launch the app.

To update to the latest version:

```
conda activate cardio-signal-lab
pip install --upgrade https://github.com/SDevrajK/CardioSignalLab/archive/refs/heads/main.zip
```

> **Note:** Installing into a dedicated conda environment (not base) avoids DLL conflicts with Qt on Windows.

### For developers (with git)

```
git clone https://github.com/SDevrajK/CardioSignalLab.git
cd CardioSignalLab
pip install -e .
```

## Running the App

```
cardio-signal-lab
```

### Using a virtual environment (recommended)

If you prefer to keep dependencies isolated from your base Python:

```
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -e .
cardio-signal-lab
```

### Using conda

If you have Miniconda or Anaconda:

```
conda env create -f environment.yml
conda activate cardio-signal-lab
pip install -e .
cardio-signal-lab
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| XDF | `.xdf` | Multi-stream physiological recordings (LSL) |
| CSV | `.csv` | Single-channel, auto-detects ECG/PPG/EDA from column names |
| Session | `.csl.json` | Saved CardioSignalLab session (resume processing) |

## Workflow

1. **File > Open** - load an XDF or CSV file
2. Select a signal channel from the sidebar
3. **Processing** menu - apply filters or EEMD artifact removal
4. **Peaks** menu - run automatic peak detection
5. Click peaks to correct manually (add/delete/reclassify)
6. **File > Export** - save peaks and processed signal
7. **File > Save Session** - save progress to resume later

## Development Setup

Install dev dependencies and run tests:

```
pip install -e ".[dev]"
pytest
```

See `docs/SETUP.md` for detailed development guidance.
