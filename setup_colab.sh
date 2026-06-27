#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Specifically designed for Google Colab environment
# Uses conda-forge channel (no ToS issues)
# ============================================================
set -e

echo "=== Step 1: Install CUDA toolkit via conda ==="
if ! command -v nvcc &> /dev/null; then
    echo "Installing Miniconda..."
    if [ ! -d "$HOME/miniconda3" ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda3
    else
        echo "Miniconda already installed, reusing."
    fi
    
    # Export conda to PATH (handle different install paths)
    if [ -d "$HOME/miniconda3" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    elif [ -d "$HOME/miniconda" ]; then
        export PATH="$HOME/miniconda/bin:$PATH"
    fi
    
    # Initialize conda (required for some commands)
    if ! command -v conda &> /dev/null; then
        $HOME/miniconda3/bin/conda init bash || true 2>/dev/null
    fi
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    echo "Accepting Conda Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    
    echo "Installing CUDA toolkit via conda-forge..."
    # Use conda-forge (no ToS issues) + nvidia channel for CUDA packages
    conda install -y -c conda-forge -c nvidia \
        cuda-toolkit=12.4 \
        cuda-cccl=12.4 \
        cmake \
        ninja
    
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
export PATH="$HOME/miniconda3/bin:$PATH"
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

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run generation with:"
echo "  python3 scripts/generate_synthetic.py \\"
echo "    --llama-server llama.cpp/build/bin/llama-server \\"
echo "    --model-q4 models/Qwen3.6-27B-Q4_K_M.gguf \\\"
echo "    --workers $(nproc 2>/dev/null || nproc) \\"
echo "    --target-samples 5000"
