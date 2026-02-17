@echo off
set CONDA_ENV=cardio-signal-lab

for %%D in (
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\miniconda3"
    "C:\anaconda3"
) do (
    if exist "%%~D\envs\%CONDA_ENV%\pythonw.exe" (
        start "" "%%~D\envs\%CONDA_ENV%\pythonw.exe" -c "from cardio_signal_lab.app import main; main()"
        exit /b 0
    )
)

echo Error: Could not find the "%CONDA_ENV%" conda environment.
echo.
echo Please set it up first. Open Anaconda Prompt and run:
echo     conda create -n %CONDA_ENV% python=3.12
echo     conda activate %CONDA_ENV%
echo     pip install https://github.com/SDevrajK/CardioSignalLab/archive/refs/heads/main.zip
echo.
pause
