@echo off
chcp 65001 >nul 2>&1

echo ============================================
echo   Financial RAG - Financial Knowledge Q&A System
echo ============================================
echo.

:: Check virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found, creating...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [+] Installing dependencies...
    pip install -r requirements.txt
    echo.
) else (
    call venv\Scripts\activate.bat
)

:: Check .env file
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [!] Created .env from .env.example, please edit and add ZHIPU_API_KEY
    ) else (
        echo [!] Please create .env file and set ZHIPU_API_KEY
    )
    echo.
)

set ANONYMIZED_TELEMETRY=False

echo [*] Starting Web service...
echo [*] Browser: http://localhost:8501
echo.

streamlit run app.py

pause
