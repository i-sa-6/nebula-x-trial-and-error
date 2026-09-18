@echo off
title Rail Condition Monitoring - Backend API
echo ========================================================
echo Starting FastAPI Backend Server on http://127.0.0.1:8000
echo ========================================================
cd /d "%~dp0"
backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause

