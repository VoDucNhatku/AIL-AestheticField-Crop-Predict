@echo off
REM ============================================================
REM  setup_modal.bat  —  One-time Modal setup (run ONCE)
REM  Paper 034: UNIC — Unbounded Image Composition
REM ============================================================
REM  Prerequisites: Python 3.10+ must be installed and on PATH
REM ============================================================

echo.
echo [1/3] Installing Modal Python package...
pip install modal
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Authenticating with Modal (opens browser)...
modal setup
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: modal setup failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying Modal CLI...
modal --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: modal CLI not found after install.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete! Next steps:
echo.
echo  1. Upload GAICD dataset to Modal Volume:
echo     modal volume put 034-unic-data  ^<local_GAIC_dir^>  /data/GAIC
echo.
echo  2a. For run-results: upload pretrained checkpoint:
echo      modal volume put 034-unic-weights  checkpoint.pth  /weights/checkpoint.pth
echo.
echo  2b. For reproduce: optionally upload DETR init weights:
echo      modal volume put 034-unic-weights  detr-r50-e632da11.pth  /weights/detr-r50-init.pth
echo.
echo  Then run:
echo     run_034_run_results.bat   — evaluation only (~1h, ~$0.59)
echo     run_034_reproduce.bat     — full training   (~10h, ~$11.80)
echo ============================================================
pause
