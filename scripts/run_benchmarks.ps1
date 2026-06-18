<#
.SYNOPSIS
Runs the evaluation script for the Adaptive Evidence-Aware RAG models on benchmark datasets.

.DESCRIPTION
This script sets up the environment and kicks off the python evaluation suite.
The output will be piped to a timestamped log file in the logs/ directory.
#>

$ErrorActionPreference = "Stop"

# Navigate to project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $ProjectRoot

# Activate Virtual Environment (adjust if using a different venv path)
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & ".\.venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Warning "No virtual environment found. Assuming system-wide installation."
}

# Ensure logs directory exists
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = ".\logs\evaluation_log_$Timestamp.txt"

Write-Host "Starting Benchmark Evaluation..."
Write-Host "Evaluation results will be logged to: $LogFile"
Write-Host "Please be patient as loading datasets and models can take some time.`n"

# Run the evaluation script
# We start with a max-samples of 100 to ensure everything works without waiting hours.
# Feel free to change max-samples to 500 or 1000 for a full rigorous benchmark.
python -m src.evaluate --max-samples 100 2>&1 | Tee-Object -FilePath $LogFile

Write-Host "`nEvaluation complete! Check $LogFile for details."
