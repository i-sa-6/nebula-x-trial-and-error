@echo off
title Rail Condition Monitoring - Launch Full App
echo ========================================================
echo Starting Rail Condition Monitoring Web Application
echo ========================================================
cd /d "%~dp0"

echo [1/2] Starting Backend Server (Port 8000)...
start "Backend API (Port 8000)" cmd /k "backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting Frontend Vite Server (Port 5173)...
timeout /t 2 >nul
start "Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers have been launched in separate terminal windows.
echo Open http://localhost:5173 in your browser to use the app.
echo.
pause

