# Setup environment for Adaptive Evidence-Aware RAG (Windows)
# Usage: .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Environment Setup Script (Windows)"     -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# Resolve project root (parent of scripts/)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

Push-Location $ProjectRoot

try {
    # Check Python version
    $PythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
                 elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
                 else { Write-Error "Python not found. Install Python 3.9+ first."; exit 1 }

    $PythonVersion = & $PythonCmd --version 2>&1
    Write-Host "Python version: $PythonVersion"

    # Create virtual environment
    $VenvDir = Join-Path $ProjectRoot "venv"
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment..."
        & $PythonCmd -m venv $VenvDir
    } else {
        Write-Host "Virtual environment already exists at $VenvDir"
    }

    # Activate
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    Write-Host "Activating virtual environment..."
    & $ActivateScript

    # Upgrade pip
    Write-Host "Upgrading pip..."
    & python -m pip install --upgrade pip

    # Detect CUDA
    $HasCuda = $false
    try {
        $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($NvidiaSmi) {
            & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $HasCuda = $true }
        }
    } catch {}

    if ($HasCuda) {
        Write-Host "CUDA detected! Installing PyTorch with CUDA support..." -ForegroundColor Green
        & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "No CUDA detected. Installing CPU-only PyTorch..." -ForegroundColor Yellow
        & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }

    # Install package in editable mode (uses pyproject.toml)
    Write-Host "Installing package in editable mode..."
    & pip install -e .

    Write-Host ""
    Write-Host "========================================"  -ForegroundColor Green
    Write-Host "  Setup complete!"                         -ForegroundColor Green
    Write-Host "========================================"  -ForegroundColor Green
    Write-Host ""
    Write-Host "Activate environment:"
    Write-Host "  .\venv\Scripts\Activate.ps1"             -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  evidence-rag --help"                     -ForegroundColor Cyan
    Write-Host "  evidence-rag info"                       -ForegroundColor Cyan
    Write-Host "  evidence-rag demo"                       -ForegroundColor Cyan
    Write-Host "  evidence-rag query `"Your question`""    -ForegroundColor Cyan
    Write-Host ""

} finally {
    Pop-Location
}
