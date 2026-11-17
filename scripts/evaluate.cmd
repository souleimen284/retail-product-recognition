@echo off
REM ONE-TIME test-set evaluation of checkpoints\best_model.pt, then error analysis.
REM The model is not modified. Test images are only read. Log: results\logs\evaluate_log.txt
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist results\logs mkdir results\logs
set "LOG=results\logs\evaluate_log.txt"

if exist results\metrics\test_predictions.csv (
  echo The test set has already been evaluated - stopping so it is not evaluated twice.
  pause
  exit /b 1
)

echo Evaluation - %DATE% %TIME% > "%LOG%"
echo [1/2] Evaluating the checkpoint on the 355 test images (once)...
"%PY%" -m src.evaluate >> "%LOG%" 2>&1 || goto :fail
echo [2/2] Error analysis (reads saved predictions, no new inference)...
"%PY%" -m src.error_analysis >> "%LOG%" 2>&1 || goto :fail
echo Done. See %LOG%
pause
exit /b 0

:fail
echo Stopped with an error - see %LOG%
pause
exit /b 1
