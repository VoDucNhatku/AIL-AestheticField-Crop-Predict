@echo off
REM ============================================================
REM  run_034_reproduce.bat
REM  Paper 034: UNIC — full training pipeline (50 epochs)
REM
REM  GPU:  T4 x2  (~$11.80 total, ~10h)
REM  Needs: GAICD dataset in Modal Volume
REM         (optional) DETR-R50 init checkpoint
REM ============================================================

echo.
echo ============================================================
echo  UNIC — reproduce mode (FULL TRAINING)
echo  GPU: T4 x2   Cost: ~$11.80   Time: ~10h
echo  batch_size=4/GPU (reduced from 8 to fit T4 16GB)
echo ============================================================
echo.
echo  COST WARNING: This run will cost approximately $11.80 USD.
echo  The job runs for ~10 hours on Modal serverless GPUs.
echo ============================================================

REM -- Check Modal is installed
modal --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Modal not found. Run setup_modal.bat first.
    pause
    exit /b 1
)

REM -- Confirm volumes exist (non-fatal check)
echo.
echo [Check] Listing Modal Volume contents...
echo   /data/GAIC:
modal volume ls 034-unic-data /data/GAIC/ 2>nul || echo   (volume empty or not created yet)
echo   /data/GAIC/annotations:
modal volume ls 034-unic-data /data/GAIC/annotations/ 2>nul || echo   (annotations not uploaded yet)

echo.
echo [!] Make sure you have uploaded BEFORE proceeding:
echo     - GAICD dataset       to /data/GAIC/images/
echo     - GAICD annotations   to /data/GAIC/annotations/instances_train.json
echo                              /data/GAIC/annotations/instances_test.json
echo.
echo     Upload command:
echo       modal volume put 034-unic-data  .\GAIC  /data/GAIC
echo.
echo [Optional] DETR-R50 init checkpoint (speeds up convergence):
echo     Download: https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
echo     Upload:   modal volume put 034-unic-weights  detr-r50-e632da11.pth  /weights/detr-r50-init.pth
echo.
set /p CONFIRM="Start full training (~$11.80, ~10h)? (y/n): "
if /i "%CONFIRM%" NEQ "y" (
    echo Cancelled.
    exit /b 0
)

echo.
echo [Running] Submitting training job to Modal...
cd /d "%~dp0"
modal run 034-reproduce\modal_app.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo  Training job submitted!
    echo.
    echo  Monitor live logs:
    echo    modal app logs 034-unic-reproduce
    echo.
    echo  After training completes, retrieve checkpoints:
    echo    modal volume ls 034-unic-output /output/train/
    echo    modal volume get 034-unic-output /output/train  .\local_output\
    echo.
    echo  Then run evaluation:
    echo    run_034_run_results.bat
    echo    (update checkpoint_path in modal_app.py to /output/train/checkpointXXXX.pth)
    echo ============================================================
) else (
    echo.
    echo ERROR: modal run failed. Check the error above.
    echo Common issues:
    echo   - Dataset not uploaded: modal volume ls 034-unic-data /data/
    echo   - Not authenticated: modal setup
    echo   - Timeout too short: check modal_apps\034-reproduce\modal_app.py
)
pause
