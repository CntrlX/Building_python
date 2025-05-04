# PowerShell script for training and testing the Mask-Aware Semi-Supervised Object Detection model

# ========== Mask-Aware Semi-Supervised Object Detection Training Script ==========

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "Mask-Aware Semi-Supervised Object Detection Model - Training Script" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# Default parameter values
$mode = "train"
$device = "cuda"
$labelPercentage = 1.0
$useSynthetic = $false
$resume = $false
$batchSize = $null
$modelPath = $null
$imagePath = $null
$testMode = $false
$debug = $false

# Function to show help
function Show-Help {
    Write-Host "`nUsage: .\train.ps1 [options]`n" -ForegroundColor Yellow
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Mode [train|val|test]       Mode to run (default: train)"
    Write-Host "  -Device [cuda|cpu]           Device to use (default: cuda)"
    Write-Host "  -LabelPercentage VALUE       Percentage of labeled data to use (default: 1.0)"
    Write-Host "  -BatchSize VALUE             Batch size for training"
    Write-Host "  -Resume                      Resume training from last checkpoint"
    Write-Host "  -UseSynthetic                Use synthetic annotations when annotations are missing"
    Write-Host "  -ModelPath PATH              Path to model for validation/testing"
    Write-Host "  -ImagePath PATH              Path to image for testing"
    Write-Host "  -Test                        Run synthetic annotation test before training"
    Write-Host "  -Debug                       Enable debug output"
    Write-Host "  -Help                        Display this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\train.ps1 -Mode train -Device cuda -LabelPercentage 0.1 -UseSynthetic"
    Write-Host "  .\train.ps1 -Mode val -ModelPath output/models/best_model.pth"
    Write-Host "  .\train.ps1 -Mode test -ImagePath path/to/image.jpg"
    Write-Host ""
    exit
}

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "-Mode" {
            $mode = $args[++$i]
            if ($mode -notin @("train", "val", "test")) {
                Write-Host "Invalid mode: $mode. Must be train, val, or test." -ForegroundColor Red
                Show-Help
            }
        }
        "-Device" {
            $device = $args[++$i]
            if ($device -notin @("cuda", "cpu")) {
                Write-Host "Invalid device: $device. Must be cuda or cpu." -ForegroundColor Red
                Show-Help
            }
        }
        "-LabelPercentage" {
            $labelPercentage = [double]$args[++$i]
        }
        "-BatchSize" {
            $batchSize = [int]$args[++$i]
        }
        "-Resume" {
            $resume = $true
        }
        "-UseSynthetic" {
            $useSynthetic = $true
        }
        "-ModelPath" {
            $modelPath = $args[++$i]
        }
        "-ImagePath" {
            $imagePath = $args[++$i]
        }
        "-Test" {
            $testMode = $true
        }
        "-Debug" {
            $debug = $true
        }
        "-Help" {
            Show-Help
        }
        default {
            Write-Host "Unknown argument: $($args[$i])" -ForegroundColor Red
            Show-Help
        }
    }
}

# Set working directory to the script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptPath

# Construct command arguments
$cmdArgs = @("--mode", $mode, "--device", $device, "--label-percentage", $labelPercentage)

if ($useSynthetic) {
    $cmdArgs += "--use-synthetic"
}

if ($resume) {
    $cmdArgs += "--resume"
}

if ($batchSize) {
    $cmdArgs += "--batch-size"
    $cmdArgs += $batchSize
}

if ($modelPath) {
    $cmdArgs += "--model-path"
    $cmdArgs += $modelPath
}

if ($imagePath) {
    $cmdArgs += "--image-path"
    $cmdArgs += $imagePath
}

if ($debug) {
    $cmdArgs += "--debug"
}

# Check if we want to run the synthetic annotation test first
if ($testMode) {
    Write-Host "Running synthetic annotation test..." -ForegroundColor Cyan
    
    # Move up one directory to execute the Python module
    Set-Location -Path (Join-Path $scriptPath "..")
    
    try {
        python -m odm.test_synthetic
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error running test script!" -ForegroundColor Red
            exit 1
        } else {
            Write-Host "Synthetic annotation test completed." -ForegroundColor Green
            Write-Host ""
        }
    } catch {
        Write-Host "An error occurred during testing: $_" -ForegroundColor Red
        exit 1
    }
}

# Execute the main command
Write-Host "Executing: python -m odm.main $($cmdArgs -join ' ')" -ForegroundColor Cyan

# Move up one directory to execute the Python module
Set-Location -Path (Join-Path $scriptPath "..")

try {
    & python -m odm.main $cmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nError: The command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Please check the logs for more information." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn error occurred during execution: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nProcess completed successfully." -ForegroundColor Green 