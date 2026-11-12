@echo off

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo No virtualenv found. Run Install.bat first.
    exit /b 1
)

call "venv\Scripts\activate.bat"

echo launching the app
python app.py %*
set APP_EXIT=%ERRORLEVEL%

pause
exit /b %APP_EXIT%
