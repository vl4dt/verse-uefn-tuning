#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Specifically designed for Google Colab environment
# Uses conda-based CUDA installation (more reliable than apt)
# ============================================================
set -e

echo "=== Step 1: Install CUDA toolkit via conda ==="
if ! command -v nvcc &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    
    echo "Installing CUDA toolkit via conda..."
    conda install -y -c nvidia cuda-toolkit=12.4 cuda-cccl=12.4
    
    # Verify installation
    if command -v nvcc &> /dev/null; then
        echo "CUDA installed: $(nvcc --version 2>/dev/null | tail -1)"
    else
        echo "ERROR: CUDA installation failed"
        exit 1
    fi
else
    echo "CUDA already installed."
fi

echo ""
echo "=== Step 2: Clone and build llama.cpp with CUDA ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
export PATH="$HOME/miniconda/bin:$PATH"
export CUDACXX=$(which nvcc)
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_SERVER=ON \
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
echo "=== Step 4: Verify setup ==="
if command -v nvcc &> /dev/null; then
    echo "CUDA: $(nvcc --version 2>/dev/null | tail -1)"
else
    echo "WARNING: CUDA not found in PATH after build"
fi

# Check VRAM if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Note: nvidia-smi not available (expected in Colab container)"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run generation with:"
echo "  python3 scripts/generate_synthetic.py \\"
echo "    --llama-server llama.cpp/build/bin/llama-server \\"
echo "    --model-q4 models/Qwen3.6-27B-Q4_K_M.gguf \\"
echo "    --workers $(nproc 2>/dev/null || nproc) \\"
echo "    --target-samples 5000"
