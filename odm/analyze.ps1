# PowerShell script for analyzing annotations in the Mask-Aware Semi-Supervised Object Detection model
Write-Host "Annotation Analysis Tool for Mask-Aware Semi-Supervised Object Detection" -ForegroundColor Green

# Set working directory to the script location
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptPath

# Process command line arguments
$mode = "both"
$numSamples = 100
$outputDir = $null

# Parse arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--mode" -and $i+1 -lt $args.Count) {
        $mode = $args[$i+1]
        $i++
    }
    elseif ($args[$i] -eq "--num-samples" -and $i+1 -lt $args.Count) {
        $numSamples = $args[$i+1]
        $i++
    }
    elseif ($args[$i] -eq "--output-dir" -and $i+1 -lt $args.Count) {
        $outputDir = $args[$i+1]
        $i++
    }
    elseif ($args[$i] -eq "--help" -or $args[$i] -eq "-h") {
        Write-Host "Usage: .\analyze.ps1 [options]"
        Write-Host "Options:"
        Write-Host "  --mode <mode>         Analysis mode: real, synthetic, both, or compare (default: both)"
        Write-Host "  --num-samples <num>   Number of samples to analyze (default: 100)"
        Write-Host "  --output-dir <dir>    Directory to save analysis results"
        Write-Host "  --help, -h            Show this help message"
        exit 0
    }
}

# Move up one directory to execute the Python module
Set-Location -Path (Join-Path $scriptPath "..")

# Build command
$command = "python -m odm.analyze_synthetic --mode $mode --num-samples $numSamples"
if ($outputDir) {
    $command += " --output-dir `"$outputDir`""
}

# Display command
Write-Host "Running: $command" -ForegroundColor Cyan

# Execute command
try {
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running analysis!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Analysis completed successfully." -ForegroundColor Green
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    exit 1
}

# Display help message for viewing results
if (-not $outputDir) {
    $defaultOutputDir = Join-Path (Join-Path $scriptPath "..") "output\analysis"
    Write-Host "Results saved to: $defaultOutputDir" -ForegroundColor Yellow
    Write-Host "You can view the analysis reports and visualizations in this directory." -ForegroundColor Yellow
} else {
    Write-Host "Results saved to: $outputDir" -ForegroundColor Yellow
    Write-Host "You can view the analysis reports and visualizations in this directory." -ForegroundColor Yellow
} 