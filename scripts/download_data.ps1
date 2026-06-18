# Download datasets for Adaptive Evidence-Aware RAG (Windows)
# Usage: .\scripts\download_data.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Dataset Download Script (Windows)"      -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# Resolve project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$DataDir = Join-Path $ProjectRoot "data"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

# Inline Python download script
$DownloadScript = @"
import os
import sys

os.chdir(r'$DataDir')

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)

datasets_to_download = [
    ('fever', 'v1.0', 'train', 100000),
    ('hotpot_qa', 'distractor', 'train', 50000),
    ('trivia_qa', 'unfiltered.nocontext', 'train', 50000),
    ('musique', 'full', 'train', 25000),
]

print("Starting dataset downloads...")
print("This may take a while depending on your internet connection.\n")

success = 0
for name, config, split, max_samples in datasets_to_download:
    try:
        print(f"Downloading {name} ({config}) ...")
        ds = load_dataset(name, config, split=split, trust_remote_code=True)
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        save_path = name.replace('/', '_')
        ds.save_to_disk(save_path)
        print(f"  Saved to {save_path} ({len(ds)} samples)")
        success += 1
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDownloaded {success}/{len(datasets_to_download)} datasets")
print(f"Data location: {os.path.abspath('.')}")

print("\nNote: Natural Questions requires manual download or academic access.")
print("Visit: https://ai.google.com/research/NaturalQuestions")
"@

Write-Host "Data directory: $DataDir"
Write-Host ""

# Run the download script
python -c $DownloadScript

Write-Host ""
Write-Host "Dataset download complete!" -ForegroundColor Green
Write-Host "Location: $DataDir"
