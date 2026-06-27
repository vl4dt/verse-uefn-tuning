#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Google Colab workflow: use conda env with CUDA activated
# This is the standard approach that works on Colab
# ============================================================
set -e

echo "=== Step 1: Set up conda environment with CUDA ==="

# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# Create and activate conda environment with CUDA toolkit
echo "Creating conda env with CUDA..."
conda create -y -n colab-cuda python=3.11 cuda-toolkit cmake ninja -c nvidia -c conda-forge || {
    echo "Trying without explicit channels..."
    conda create -y -n colab-cuda python=3.11 cuda-toolkit cmake ninja
}

# Activate the environment — this puts headers in the right place
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate colab-cuda

echo "Using CUDA from activated conda env:"
nvcc --version 2>/dev/null | tail -1
echo "CUDA_HOME=$CONDA_PREFIX"

echo ""
echo "=== Step 2: Clone and build llama.cpp ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
export CUDACXX=$(which nvcc)
echo "Building with: $CUDACXX (CUDA_HOME=$CONDA_PREFIX)"

# Build with conda env's CUDA paths
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_PATH="$CONDA_PREFIX" \
    -DCUDACXX="$CUDACXX"

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
echo "=== Setup complete! ==="
echo ""
echo "Run generation with:"
echo "  python3 scripts/generate_synthetic.py \\"
echo "    --llama-server llama.cpp/build/bin/llama-server \\"
echo "    --model-q4 models/Qwen3.6-27B-Q4_K_M.gguf \\"
echo "    --workers $(nproc 2>/dev/null || nproc) \\"
echo "    --target-samples 5000"

# Deactivate conda env when script ends (optional, Colab usually doesn't need it)
# conda deactivate
