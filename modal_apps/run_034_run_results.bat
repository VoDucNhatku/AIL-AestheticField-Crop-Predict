@echo off
REM ============================================================
REM  run_034_run_results.bat
REM  Paper 034: UNIC — evaluation on GAICD test set
REM
REM  GPU:  T4 x1  (~$0.59 total, ~1h)
REM  Needs: pretrained checkpoint in Modal Volume
REM         GAICD dataset in Modal Volume
REM ============================================================

echo.
echo ============================================================
echo  UNIC — run-results mode (evaluation only)
echo  GPU: T4 x1   Cost: ~$0.59   Time: ~1h
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
echo   /weights:
modal volume ls 034-unic-weights /weights/ 2>nul || echo   (volume empty or not created yet)
echo   /data/GAIC:
modal volume ls 034-unic-data /data/GAIC/ 2>nul || echo   (volume empty or not created yet)

echo.
echo [!] Make sure you have uploaded:
echo     - Pretrained checkpoint  to /weights/checkpoint.pth
echo     - GAICD dataset          to /data/GAIC/
echo     - GAICD annotations      to /data/GAIC/annotations/instances_test.json
echo.
set /p CONFIRM="Proceed with evaluation? (y/n): "
if /i "%CONFIRM%" NEQ "y" (
    echo Cancelled.
    exit /b 0
)

echo.
echo [Running] Submitting job to Modal...
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
modal run 034-run-results\modal_app.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo  Job submitted successfully!
    echo  Monitor logs: modal app logs 034-unic-run-results
    echo  Expected metrics (Table 1, paper Section 4.2):
    echo    GAICD: IoU=0.801, Disp=0.048
    echo    GAICD: Acc_1/5(e=0.90)=23.2, Acc_1/10(e=0.85)=72.8
    echo ============================================================
) else (
    echo.
    echo ERROR: modal run failed. Check the error above.
)
pause
