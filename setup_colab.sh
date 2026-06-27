#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Specifically designed for Google Colab environment
# Prefers system CUDA (Colab ships with /usr/local/cuda) over conda
# ============================================================
set -e

echo "=== Step 1: Set up CUDA ==="

# Prefer system CUDA (Colab ships with it at /usr/local/cuda)
if [ -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "Using system CUDA: $(nvcc --version 2>/dev/null | tail -1)"
else
    # Colab may have put nvcc somewhere else — try common locations
    for cand in /usr/local/cuda-12.4/bin/nvcc \
                /usr/local/cuda-12.3/bin/nvcc \
                /usr/local/cuda-12.2/bin/nvcc \
                /root/miniconda3/envs/colab/bin/nvcc; do
        if [ -x "$cand" ]; then
            export PATH="$(dirname $cand):$PATH"
            echo "Using system CUDA: $(nvcc --version 2>/dev/null | tail -1)"
            break
        fi
    done

    # Fallback: install via conda (only if no system CUDA found)
    if ! command -v nvcc &> /dev/null; then
        echo "No system CUDA found. Installing Miniconda + CUDA toolkit..."
        if [ ! -d "$HOME/miniconda3" ]; then
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            bash miniconda.sh -b -p $HOME/miniconda3
        fi
        export PATH="$HOME/miniconda3/bin:$PATH"
        
        echo "Installing CUDA toolkit via conda-forge..."
        conda install -y -c conda-forge -c nvidia \
            cuda-toolkit=12.4 \
            cuda-cccl=12.4 \
            cmake \
            ninja

        if command -v nvcc &> /dev/null; then
            echo "CUDA installed via conda: $(nvcc --version 2>/dev/null | tail -1)"
        else
            echo "ERROR: CUDA installation failed"
            exit 1
        fi
    fi
fi

echo ""
echo "=== Step 2: Clone and build llama.cpp with CUDA ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
export CUDACXX=$(which nvcc)
echo "Building with: $CUDACXX"

# Tell cmake where CUDA lives explicitly (avoids conda path confusion)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -d "$CUDA_HOME" ] && command -v nvcc &> /dev/null; then
    # Derive from nvcc location: /usr/local/cuda/bin/nvcc -> /usr/local/cuda
    CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi
echo "Using CUDA_HOME=$CUDA_HOME"

cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_PATH="$CUDA_HOME" \
    -DCUDACXX="$CUDACXX" \
    -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" \
    -DCMAKE_CUDA_FLAGS="-I${CUDA_HOME}/include"
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
echo "    --model-q4 models/Qwen3.6-27B-Q4_K_M.gguf \\"
echo "    --workers $(nproc 2>/dev/null || nproc) \\"
echo "    --target-samples 5000"
