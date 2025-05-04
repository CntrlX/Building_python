@echo off
SETLOCAL EnableDelayedExpansion

REM ========== Mask-Aware Semi-Supervised Object Detection Training Script ==========

echo ===================================================================
echo Mask-Aware Semi-Supervised Object Detection Model - Training Script
echo ===================================================================

REM Parse command-line arguments
SET mode=train
SET device=cuda
SET label_percentage=1.0
SET use_synthetic=
SET resume=
SET batch_size=
SET model_path=
SET image_path=
SET test_mode=
SET debug=

:parse_args
IF "%~1"=="" GOTO execute
IF /I "%~1"=="--mode" (
    SET mode=%~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--device" (
    SET device=%~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--label-percentage" (
    SET label_percentage=%~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--batch-size" (
    SET batch_size=--batch-size %~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--resume" (
    SET resume=--resume
    GOTO next_arg
)
IF /I "%~1"=="--use-synthetic" (
    SET use_synthetic=--use-synthetic
    GOTO next_arg
)
IF /I "%~1"=="--model-path" (
    SET model_path=--model-path %~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--image-path" (
    SET image_path=--image-path %~2
    SHIFT
    GOTO next_arg
)
IF /I "%~1"=="--test" (
    SET test_mode=1
    GOTO next_arg
)
IF /I "%~1"=="--debug" (
    SET debug=--debug
    GOTO next_arg
)
IF /I "%~1"=="help" (
    GOTO show_help
)
IF /I "%~1"=="/?" (
    GOTO show_help
)

echo Unknown argument: %~1
GOTO show_help

:next_arg
SHIFT
GOTO parse_args

:show_help
echo.
echo Usage: train.bat [options]
echo.
echo Options:
echo   --mode [train^|val^|test]       Mode to run (default: train)
echo   --device [cuda^|cpu]           Device to use (default: cuda)
echo   --label-percentage VALUE     Percentage of labeled data to use (default: 1.0)
echo   --batch-size VALUE           Batch size for training
echo   --resume                     Resume training from last checkpoint
echo   --use-synthetic              Use synthetic annotations when annotations are missing
echo   --model-path PATH            Path to model for validation/testing
echo   --image-path PATH            Path to image for testing
echo   --test                       Run synthetic annotation test before training
echo   --debug                      Enable debug output
echo   help, /?                     Display this help message
echo.
echo Examples:
echo   train.bat --mode train --device cuda --label-percentage 0.1 --use-synthetic
echo   train.bat --mode val --model-path output/models/best_model.pth
echo   train.bat --mode test --image-path path/to/image.jpg
echo.
GOTO :EOF

:execute
REM Check if we want to run the synthetic annotation test first
IF DEFINED test_mode (
    echo Running synthetic annotation test...
    cd ..
    python -m odm.test_synthetic
    echo Synthetic annotation test completed.
    echo.
)

REM Set default values for empty parameters
IF "%mode%"=="" SET mode=train
IF "%device%"=="" SET device=cuda
IF "%label_percentage%"=="" SET label_percentage=1.0

REM Construct the command
SET cmd=python -m odm.main --mode %mode% --device %device% --label-percentage %label_percentage% %use_synthetic% %resume% %batch_size% %model_path% %image_path% %debug%

echo Executing: %cmd%
cd ..
%cmd%

REM Check for errors
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: The command failed with exit code %ERRORLEVEL%
    echo Please check the logs for more information.
    exit /b %ERRORLEVEL%
)

echo.
echo Process completed successfully.
ENDLOCAL 