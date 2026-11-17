@echo off
REM Download only the 12 selected classes (sparse checkout), verify splits and images,
REM and write the report-only duplicate analysis. Nothing is deleted or moved.
REM Log: results\logs\prepare_data_log.txt
setlocal
cd /d "%~dp0.."
set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist results\logs mkdir results\logs
set "LOG=results\logs\prepare_data_log.txt"
echo Data preparation - %DATE% %TIME% > "%LOG%"

echo [1/5] Downloading split files only (no images)...
if not exist data\GroceryStoreDataset\.git (
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/marcusklasson/GroceryStoreDataset.git data\GroceryStoreDataset >> "%LOG%" 2>&1 || goto :fail
)
git -C data\GroceryStoreDataset sparse-checkout set --no-cone "/dataset/*.txt" "/dataset/classes.csv" "/LICENSE" "/README.md" >> "%LOG%" 2>&1 || goto :fail
git -C data\GroceryStoreDataset checkout >> "%LOG%" 2>&1 || goto :fail

echo [2/5] Exact counts and split checks (stops here if any check fails)...
"%PY%" -m src.prepare_data --counts-only >> "%LOG%" 2>&1 || goto :fail

echo [3/5] Downloading images of the 12 selected classes only...
git -C data\GroceryStoreDataset sparse-checkout set --no-cone --stdin < data\sparse_paths.txt >> "%LOG%" 2>&1 || goto :fail

echo [4/5] Verifying every image file...
"%PY%" -m src.prepare_data >> "%LOG%" 2>&1 || goto :fail

echo [5/5] Duplicate analysis (report only)...
"%PY%" -m src.check_duplicates >> "%LOG%" 2>&1 || goto :fail

echo Done. See %LOG%
pause
exit /b 0

:fail
echo Stopped - see %LOG%. Nothing was deleted.
pause
exit /b 1
