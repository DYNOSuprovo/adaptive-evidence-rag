#!/bin/bash
# Download datasets for Adaptive Evidence-Aware RAG

set -e

echo "========================================"
echo "Dataset Download Script"
echo "========================================"

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Create Python script for downloading
cat > "$DATA_DIR/download.py" << 'EOF'
import os
from datasets import load_dataset
from tqdm import tqdm

def download_dataset(name, config=None, split='train', max_samples=10000):
    """Download a dataset from HuggingFace."""
    try:
        print(f"\nDownloading {name}...")
        if config:
            ds = load_dataset(name, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
        
        # Limit samples if needed
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        
        # Save to disk
        save_path = f"./{name.replace('/', '_')}"
        ds.save_to_disk(save_path)
        print(f"Saved to {save_path} ({len(ds)} samples)")
        return True
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return False

# Download all datasets
datasets_to_download = [
    ('fever', 'v1.0', 'train', 100000),
    ('hotpot_qa', 'distractor', 'train', 50000),
    ('trivia_qa', 'unfiltered.nocontext', 'train', 50000),
    ('musique', 'full', 'train', 25000),
]

print("Starting dataset downloads...")
print("This may take a while depending on your internet connection.")

success_count = 0
for name, config, split, max_samples in datasets_to_download:
    if download_dataset(name, config, split, max_samples):
        success_count += 1

print(f"\n========================================")
print(f"Downloaded {success_count}/{len(datasets_to_download)} datasets")
print(f"Data location: {os.path.abspath('.')}")
print(f"========================================")

# Note about Natural Questions
print("\nNote: Natural Questions requires manual download or academic access.")
print("Visit: https://ai.google.com/research/NaturalQuestions")
print("Or use: huggingface.co/datasets/google-research-datasets/natural_questions")
EOF

# Run download script
cd "$DATA_DIR"
python3 download.py

echo ""
echo "Dataset download complete!"
echo "Location: $DATA_DIR"
