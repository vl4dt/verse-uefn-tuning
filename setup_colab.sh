#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Works on Google Colab (T4 / A100 / L4) with GPU runtime enabled
# ============================================================
set -e

echo "=== Step 1: Install CUDA toolkit ==="
if ! dpkg -l | grep -q cuda-cudart-dev; then
    echo "Installing CUDA toolkit..."
    # Add NVIDIA package repo (Colab uses Ubuntu 20.04/22.04)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    
    # Detect GPU and install appropriate CUDA version
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    
    # Use latest stable CUDA (works on Colab)
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        libcuda1-dev \
        cuda-cudart-dev \
        libcublas-dev \
        libcufft-dev
    
    echo "CUDA toolkit installed: $(nvcc --version 2>/dev/null | tail -1)"
else
    echo "CUDA toolkit already installed."
fi

echo ""
echo "=== Step 2: Clone and build llama.cpp with CUDA ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
export CUDACXX=$(which nvcc)
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_SERVER=ON \
    -DGGML_CUDA_MOCK=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc 2>/dev/null || nproc)

echo "llama.cpp built: $(ls build/bin/llama-server)"
cd ..

echo ""
echo "=== Step 3: Download Qwen3.6-27B-Q4_K_M GGUF model ==="
mkdir -p models

if [ ! -f "models/Qwen3.6-27B-Q4_K_M.gguf" ]; then
    echo "Downloading from HuggingFace (may take 5–10 min)..."
    pip install -q huggingface_hub
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.6-27B-MTP-GGUF', 'Qwen3.6-27B-Q4_K_M.gguf', local_dir='models/')"
else
    echo "Model already downloaded, skipping."
fi

echo ""
echo "=== Step 4: Verify GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Auto-select quant based on VRAM
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | tr -d ' MB' | cut -d. -f1)
if [ "$TOTAL_VRAM" -ge 20000 ]; then
    MODEL_FILE="models/Qwen3.6-27B-Q4_K_M.gguf"
    echo "VRAM: ${TOTAL_VRAM}MB — using Q4_K_M (~18GB)"
else
    MODEL_FILE="models/Qwen3.6-27B-Q4_K_M.gguf"
    echo "VRAM: ${TOTAL_VRAM}MB — using Q4_K_M (tight fit, may use CPU offload)"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run generation with:"
echo "  python3 scripts/generate_synthetic.py \\"
echo "    --llama-server llama.cpp/build/bin/llama-server \\"
echo "    --model-q4 $MODEL_FILE \\"
echo "    --workers $(nproc 2>/dev/null || nproc) \\"
echo "    --target-samples 5000"
echo ""
echo "To resume an interrupted run:"
echo "  python3 scripts/generate_synthetic.py --resume"
