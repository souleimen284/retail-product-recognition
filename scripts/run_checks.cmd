@echo off
REM Pipeline checks (no real training) and the read-only data-integrity audit.
REM Log: results\logs\run_checks_log.txt
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist results\logs mkdir results\logs
set "LOG=results\logs\run_checks_log.txt"
echo Pipeline checks - %DATE% %TIME% > "%LOG%"
echo [1/2] Pipeline checks (about 1 minute)...
"%PY%" -m src.smoke_test >> "%LOG%" 2>&1 || goto :fail
echo [2/2] Data-integrity audit...
"%PY%" -m src.audit_data >> "%LOG%" 2>&1 || goto :fail
echo All checks passed. See %LOG%
pause
exit /b 0

:fail
echo Some checks failed - see %LOG%
pause
exit /b 1
