@echo off
REM Train MobileNetV3-Small on the train split, selecting the checkpoint on the val split.
REM The test split is never used. Output is shown live and saved to results\logs\train_log.txt
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist results\logs mkdir results\logs

REM Never overwrite an existing trained model by accident.
if exist checkpoints\best_model.pt (
  echo checkpoints\best_model.pt already exists - move or rename it before re-training.
  pause
  exit /b 1
)

echo [1/2] Pipeline checks (about 1 minute)...
"%PY%" -m src.smoke_test > results\logs\train_precheck_log.txt 2>&1
if errorlevel 1 (
  echo Pipeline checks FAILED - training not started. See results\logs\train_precheck_log.txt
  pause
  exit /b 1
)

echo [2/2] Training on CPU (about 2-3 minutes)...
"%PY%" -m src.train ^
  --epochs-head 3 --epochs-finetune 25 ^
  --lr-head 1e-3 --lr-finetune 1e-4 ^
  --weight-decay 0.01 --label-smoothing 0.1 --patience 6 ^
  --batch-size 16 --workers 2 --threads 6 --seed 42 --device cpu ^
  --log-file results\logs\train_log.txt
if errorlevel 1 (
  echo Training stopped with an error. See results\logs\train_log.txt
) else (
  echo Training finished. Checkpoint: checkpoints\best_model.pt
)
pause
