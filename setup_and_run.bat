@echo off
set ENV_PATH=echo

REM Check if the virtual environment exists, create it if not
if not exist "%ENV_PATH%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %ENV_PATH%
    echo Virtual environment created.
)

REM Activate the environment
call %ENV_PATH%\Scripts\activate

REM Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

REM Load environment variables from .env file
@REM for /f "delims=" %%a in (".env") do set "%%a"

REM Run the Python scripts with loaded environment variables
call python auth\trainer.py --wallet_address 0xidjr389v4y75ny73458g3ynf7d
call python auth\auth.py login echo2 sH9v6x
@REM call python auth\trainer.py --wallet_address "%WALLET_ADDRESS%"
@REM call python auth\auth.py login "%USERNAME%" "%PASSWORD%"

REM Deactivate the environment
call %ENV_PATH%\Scripts\deactivate

pause
