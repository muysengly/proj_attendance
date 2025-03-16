@echo off

@REM create a virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
)

@REM call venv\Scripts\activate
call venv\Scripts\activate

@REM update pip optional
@REM python.exe -m pip install --upgrade pip


@REM install dependencies offline
pip install --no-index --find-links=__package__ -r requirements.txt

echo.
echo.
echo Dependencies installed!
echo.
echo.

pause