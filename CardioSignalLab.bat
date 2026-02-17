@echo off
python -c "import cardio_signal_lab" >nul 2>&1
if errorlevel 1 (
    echo Error: CardioSignalLab is not installed.
    echo.
    echo To install, open your Python IDE terminal and run:
    echo pip install git+https://github.com/SDevrajK/CardioSignalLab.git
    echo.
    pause
    exit /b 1
)
start "" pythonw -c "from cardio_signal_lab.app import main; main()"
