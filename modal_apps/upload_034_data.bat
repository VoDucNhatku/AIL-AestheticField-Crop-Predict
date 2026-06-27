@echo off
REM ============================================================
REM  upload_034_data.bat
REM  Upload GAICD dataset + checkpoint to Modal Volumes
REM  Run this ONCE before run_034_run_results.bat or
REM  run_034_reproduce.bat
REM ============================================================

echo.
echo ============================================================
echo  Upload data to Modal Volumes for Paper 034 (UNIC)
echo ============================================================

modal --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Modal not found. Run setup_modal.bat first.
    pause
    exit /b 1
)

echo.
echo This script uploads:
echo   1. GAICD dataset folder    -> Modal Volume: 034-unic-data
echo   2. Pretrained checkpoint   -> Modal Volume: 034-unic-weights  (run-results)
echo   3. DETR init checkpoint    -> Modal Volume: 034-unic-weights  (reproduce, optional)
echo.
echo ============================================================
echo  STEP 1: Upload GAICD dataset
echo ============================================================
echo.
echo Expected local folder structure:
echo   GAIC\
echo     images\        (GAICD image files)
echo     annotations\
echo       instances_train.json
echo       instances_test.json
echo.
set /p GAIC_PATH="Enter path to local GAIC folder (e.g. C:\Downloads\GAIC): "
if not exist "%GAIC_PATH%" (
    echo ERROR: Folder not found: %GAIC_PATH%
    echo Download from: https://drive.google.com/file/d/1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR/view
    pause
    exit /b 1
)

echo.
echo Uploading GAIC dataset to Modal Volume 034-unic-data ...
modal volume put 034-unic-data "%GAIC_PATH%" /data/GAIC
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Upload failed.
    pause
    exit /b 1
)
echo [OK] Dataset uploaded.

echo.
echo ============================================================
echo  STEP 2: Upload pretrained checkpoint (for run-results)
echo ============================================================
echo.
echo Download from Google Drive first:
echo   https://drive.google.com/file/d/1owvtFQBCC4uxd5f6tRsh7Fgktuj8MzZy/view
echo.
set /p CKPT_PATH="Enter path to checkpoint .pth file (or press Enter to skip): "
if "%CKPT_PATH%"=="" (
    echo Skipped pretrained checkpoint.
) else (
    if not exist "%CKPT_PATH%" (
        echo WARNING: File not found: %CKPT_PATH% — skipping.
    ) else (
        echo Uploading checkpoint...
        modal volume put 034-unic-weights "%CKPT_PATH%" /weights/checkpoint.pth
        if %ERRORLEVEL% EQU 0 (
            echo [OK] Checkpoint uploaded to /weights/checkpoint.pth
        ) else (
            echo WARNING: Checkpoint upload failed.
        )
    )
)

echo.
echo ============================================================
echo  STEP 3: Upload DETR-R50 init weights (optional, for reproduce)
echo ============================================================
echo.
echo Download from Facebook Research:
echo   https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
echo.
set /p DETR_PATH="Enter path to detr-r50 .pth file (or press Enter to skip): "
if "%DETR_PATH%"=="" (
    echo Skipped DETR init weights.
) else (
    if not exist "%DETR_PATH%" (
        echo WARNING: File not found: %DETR_PATH% — skipping.
    ) else (
        echo Uploading DETR init weights...
        modal volume put 034-unic-weights "%DETR_PATH%" /weights/detr-r50-init.pth
        if %ERRORLEVEL% EQU 0 (
            echo [OK] DETR weights uploaded to /weights/detr-r50-init.pth
        ) else (
            echo WARNING: Upload failed.
        )
    )
)

echo.
echo ============================================================
echo  Verifying uploads...
echo ============================================================
echo.
echo [034-unic-data] /data/GAIC:
modal volume ls 034-unic-data /data/GAIC/
echo.
echo [034-unic-weights] /weights:
modal volume ls 034-unic-weights /weights/

echo.
echo ============================================================
echo  Done. You can now run:
echo    run_034_run_results.bat   (evaluation, ~$0.59)
echo    run_034_reproduce.bat     (full training, ~$11.80)
echo ============================================================
pause
