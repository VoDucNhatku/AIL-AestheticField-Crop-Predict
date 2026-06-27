@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  run_034_full_auto.bat
REM  Auto-script: upload GAICD + run UNIC on Modal
REM
REM  Dataset path: C:\Users\DELL\Downloads\GAIC  (hardcoded)
REM  Choose mode when prompted: (1) run-results or (2) reproduce
REM ============================================================

set GAIC_PATH=C:\Users\DELL\Downloads\GAIC
set SCRIPT_DIR=%~dp0

echo.
echo ============================================================
echo  UNIC (Paper 034) — Modal Auto-Runner
echo ============================================================
echo  Dataset: %GAIC_PATH%
echo ============================================================

REM -- Step 0: Check Modal CLI
echo.
echo [0/4] Checking Modal CLI...
modal --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  Modal not found. Installing now...
    pip install modal
    if %ERRORLEVEL% NEQ 0 (
        echo  ERROR: pip install modal failed. Install Python first.
        pause & exit /b 1
    )
    echo  Modal installed. Running first-time setup...
    modal setup
    if %ERRORLEVEL% NEQ 0 (
        echo  ERROR: modal setup failed.
        pause & exit /b 1
    )
)
echo  [OK] Modal ready.

REM -- Step 1: Check dataset folder exists
echo.
echo [1/4] Checking dataset at %GAIC_PATH% ...
if not exist "%GAIC_PATH%" (
    echo  ERROR: Folder not found: %GAIC_PATH%
    echo  Download GAICD from:
    echo    https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view
    echo  Then place it at: %GAIC_PATH%
    pause & exit /b 1
)
echo  [OK] Dataset folder found.

REM -- Step 2: Choose mode
echo.
echo ============================================================
echo  Choose run mode:
echo    1  run-results  (evaluation with pretrained weights)
echo                    GPU: T4 x1   Cost: ~$0.59   Time: ~1h
echo    2  reproduce    (full training from scratch)
echo                    GPU: T4 x2   Cost: ~$11.80  Time: ~10h
echo ============================================================
echo.
set /p MODE_CHOICE="Enter 1 or 2: "

if "%MODE_CHOICE%"=="1" (
    set RUN_MODE=run-results
    set APP_DIR=034-run-results
    set APP_NAME=034-unic-run-results
    set NEED_CHECKPOINT=1
) else if "%MODE_CHOICE%"=="2" (
    set RUN_MODE=reproduce
    set APP_DIR=034-reproduce
    set APP_NAME=034-unic-reproduce
    set NEED_CHECKPOINT=0
) else (
    echo  Invalid choice. Exiting.
    pause & exit /b 1
)
echo  Mode selected: %RUN_MODE%

REM -- Step 3: Create volumes (idempotent — safe to run again)
echo.
echo [2/4] Creating Modal Volumes (if not exist)...
modal volume create 034-unic-data    >nul 2>&1
modal volume create 034-unic-weights >nul 2>&1
modal volume create 034-unic-output  >nul 2>&1
echo  [OK] Volumes ready.

REM -- Upload dataset
echo.
echo [2b/4] Uploading dataset to Modal Volume 034-unic-data ...
echo  Source : %GAIC_PATH%
echo  Dest   : /data/GAIC
echo  (This may take several minutes on first upload; subsequent runs skip unchanged files.)
echo.
modal volume put 034-unic-data "%GAIC_PATH%" /data/GAIC --force
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Dataset upload failed.
    pause & exit /b 1
)
echo  [OK] Dataset uploaded.

REM -- Step 3b: Upload checkpoint if run-results mode
if "%NEED_CHECKPOINT%"=="1" (
    echo.
    echo [2b/4] Checking pretrained checkpoint for run-results mode...
    modal volume ls 034-unic-weights /weights/checkpoint.pth >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo  [OK] Checkpoint already in Volume. Skipping upload.
    ) else (
        echo  Checkpoint not found in Volume.
        echo  Download from Google Drive:
        echo    https://drive.google.com/file/d/1owvtFQBCC4uxd5f6tRsh7Fgktuj8MzZy/view
        echo  Save it as: C:\Users\DELL\Downloads\checkpoint.pth
        echo.
        set CKPT_LOCAL=C:\Users\DELL\Downloads\checkpoint.pth
        if exist "!CKPT_LOCAL!" (
            echo  Found checkpoint at !CKPT_LOCAL! — uploading...
            modal volume put 034-unic-weights "!CKPT_LOCAL!" /weights/checkpoint.pth
            if %ERRORLEVEL% NEQ 0 (
                echo  WARNING: Checkpoint upload failed. run-results will error at runtime.
            ) else (
                echo  [OK] Checkpoint uploaded.
            )
        ) else (
            echo  WARNING: Checkpoint not found at !CKPT_LOCAL!
            echo  The job will fail unless you upload the checkpoint first:
            echo    modal volume put 034-unic-weights ^<checkpoint.pth^> /weights/checkpoint.pth
        )
    )
)

REM -- Step 4: Run Modal job
echo.
echo [3/4] Verifying Volume contents...
echo  /data/GAIC:
modal volume ls 034-unic-data /data/GAIC/
echo.

echo [4/4] Submitting Modal job: %RUN_MODE% ...
echo  App : %APP_NAME%
echo  File: %SCRIPT_DIR%%APP_DIR%\modal_app.py
echo.
cd /d "%SCRIPT_DIR%"
powershell -NoProfile -Command "$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; modal run '%APP_DIR%\modal_app.py'"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo  Job submitted successfully!
    echo.
    echo  Monitor live logs:
    echo    modal app logs %APP_NAME%
    echo.
    if "%RUN_MODE%"=="reproduce" (
        echo  After training completes, retrieve checkpoints:
        echo    modal volume get 034-unic-output /output/train .\local_output\
        echo.
        echo  Then run evaluation with the new checkpoint:
        echo    (update checkpoint_path in 034-run-results\modal_app.py)
        echo    run_034_full_auto.bat  -^> choose option 1
    ) else (
        echo  Expected metrics (Table 1, paper Section 4.2):
        echo    GAICD: IoU=0.801, Disp=0.048
        echo    GAICD: Acc_1/5(e=0.90)=23.2
    )
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo  ERROR: Modal job failed. Common causes:
    echo    - Dataset missing: modal volume ls 034-unic-data /data/
    echo    - Checkpoint missing (run-results): modal volume ls 034-unic-weights /weights/
    echo    - Not authenticated: modal setup
    echo    - Timeout: check modal_apps\%APP_DIR%\modal_app.py
    echo ============================================================
)

pause
endlocal
