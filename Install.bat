@echo off

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo python was not found. Install Python 3.12, then run this script again.
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo Venv creation failed. Fix the error above and run install.bat again.
        exit /b 1
    )
)
echo checked the venv folder. now installing requirements..

call "venv\Scripts\activate.bat"

python -m pip install -U pip
pip install -U -r "requirements.txt"

if errorlevel 1 (
    echo.
    echo Requirements installation failed. please remove venv folder and run install.bat again.
    exit /b 1
) else (
    echo.
    echo Requirements installed successfully.
)
pause
