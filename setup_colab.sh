#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Run this on a Colab GPU instance (T4 / A100 / L4) before generate_synthetic.py
#
# Prerequisites: NVIDIA driver + CUDA toolkit (Colab provides these)
# Expected VRAM: ~18 GB for Q4_K_M (T4 has 16GB, use Q3_K_M; A100/L4 have 24GB+)
# ============================================================
set -e

echo "=== Step 1: Clone and build llama.cpp with CUDA ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_SERVER=ON
cmake --build build --config Release -j$(nproc)
echo "llama.cpp built: $(ls build/bin/llama-server)"
cd ..

echo ""
echo "=== Step 2: Download Qwen3.6-27B-Q4_K_M GGUF ==="
mkdir -p models

# Check if model already exists (save bandwidth on resume)
if [ ! -f "models/Qwen3.6-27B-Q4_K_M.gguf" ]; then
    echo "Downloading Qwen3.6-27B-Q4_K_M from HuggingFace..."
    huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF \
        Qwen3.6-27B-Q4_K_M.gguf \
        --local-dir models/
else
    echo "Model already downloaded, skipping."
fi

echo ""
echo "=== Step 3: Verify GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Auto-select model quant based on VRAM
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | tr -d ' MB' | cut -d. -f1)
if [ "$TOTAL_VRAM" -ge 20000 ]; then
    MODEL_FILE="models/Qwen3.6-27B-Q4_K_M.gguf"
    echo "VRAM: ${TOTAL_VRAM}MB — using Q4_K_M (~18GB)"
elif [ "$TOTAL_VRAM" -ge 15000 ]; then
    # Try UD quant (more efficient for MTP) if available, otherwise use Q3_K_M
    MODEL_FILE="models/Qwen3.6-27B-Q4_K_M.gguf"
    echo "VRAM: ${TOTAL_VRAM}MB — using Q4_K_M (~18GB), may need CPU offload"
else
    MODEL_FILE="models/Qwen3.6-27B-Q4_K_M.gguf"
    echo "VRAM: ${TOTAL_VRAM}MB — tight fit, consider upgrading to A100/L4"
fi

echo ""
echo "=== Setup complete! ==="
echo "Run with:"
echo "  python3 scripts/generate_synthetic.py \\"
echo "    --llama-server llama.cpp/build/bin/llama-server \\"
echo "    --model-q4 $MODEL_FILE \\"
echo "    --workers $(nproc) \\"
echo "    --target-samples 5000"
