@echo off
call conda activate cardio-signal-lab 2>nul
python -c "import cardio_signal_lab" >nul 2>&1
if errorlevel 1 (
    echo Error: CardioSignalLab is not installed.
    echo.
    echo To install, open Anaconda Prompt or your terminal and run:
    echo conda create -n cardio-signal-lab python=3.12
    echo conda activate cardio-signal-lab
    echo pip install https://github.com/SDevrajK/CardioSignalLab/archive/refs/heads/main.zip
    echo.
    pause
    exit /b 1
)
start "" pythonw -c "from cardio_signal_lab.app import main; main()"
