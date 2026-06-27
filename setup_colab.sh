#!/bin/bash
# Setup script for Google Colab — installs Python deps only (HuggingFace backend)
echo "=== Setting up verse-uefn-tuning on Colab ==="

cd /content/verse-uefn-tuning

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5
pip install -r requirements.txt 2>&1 | tail -5

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run generation with:"
echo "  python3 scripts/generate_synthetic.py \\
    --backend huggingface \\
    --model Qwen/Qwen3.6-27B-MTP \\
    --workers $(nproc 2>/dev/null || nproc) \\
    --target-samples 5000"

