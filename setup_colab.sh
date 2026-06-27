#!/bin/bash
# ============================================================
# setup_colab.sh — Build llama.cpp + download Qwen3.6-27B model
# Specifically designed for Google Colab environment
# Handles both system CUDA and conda-installed CUDA with proper include paths
# ============================================================
set -e

echo "=== Step 1: Set up CUDA ==="

find_system_cuda() {
    # Search common Colab/system CUDA locations
    local candidates=(
        /usr/local/cuda/bin/nvcc
        /usr/local/cuda-13.3/bin/nvcc
        /usr/local/cuda-13/bin/nvcc
        /usr/local/cuda-12.6/bin/nvcc
        /usr/local/cuda-12.4/bin/nvcc
        /usr/local/cuda-12.3/bin/nvcc
    )

    # Also try any cuda-X.Y pattern in /usr/local
    for cand in /usr/local/cuda-[0-9]*; do
        if [ -x "$cand/bin/nvcc" ]; then
            candidates+=("$cand/bin/nvcc")
        fi
    done

    # Deduplicate while preserving order
    local seen=()
    for cand in "${candidates[@]}"; do
        if [[ ! " ${seen[*]} " =~ " $cand " ]]; then
            seen+=("$cand")
        fi
    done

    echo "${seen[@]}"
}

find_cuda_headers() {
    local nvcc_dir="$1"  # e.g., /usr/local/cuda/bin or /root/miniconda3/bin
    
    # Try common include layouts
    local candidates=(
        "${nvcc_dir}/../include"
        "${nvcc_dir}/../../include"
        "${nvcc_dir}/../targets/x86_64-linux/include"
        "/usr/local/cuda-13.3/targets/x86_64-linux/include"
        "/usr/local/cuda-13/targets/x86_64-linux/include"
        "/usr/local/cuda/include"
    )

    for cand in "${candidates[@]}"; do
        if [ -f "$cand/cuda_runtime.h" ]; then
            echo "$cand"
            return 0
        fi
    done

    # Last resort: search system CUDA installation
    for cand in /usr/local/cuda-13.3/include /usr/local/cuda-13/include /usr/local/cuda/include; do
        if [ -f "$cand/cuda_runtime.h" ]; then
            echo "$cand"
            return 0
        fi
    done

    # Conda-specific: check conda pkgs directory for cuda headers
    local miniconda_dir="/root/miniconda3"
    if [ -d "$miniconda_dir/pkgs" ]; then
        find "$miniconda_dir/pkgs" -name "cuda_runtime.h" 2>/dev/null | head -1
        return $?
    fi

    echo ""
    return 1
}

# Find system CUDA first
echo "Searching for system CUDA..."
SYSTEM_NVCCS=($(find_system_cuda))

USE_SYSTEM_CUDA=false
for nvcc in "${SYSTEM_NVCCS[@]}"; do
    if [ -x "$nvcc" ]; then
        echo "Found system nvcc: $nvcc"
        export PATH="$(dirname $nvcc):$PATH"
        
        # Find headers for this nvcc's CUDA installation
        NVCC_DIR=$(dirname $(dirname "$nvcc"))
        HEADERS=$(find_cuda_headers "$(dirname $nvcc)")
        
        if [ -n "$HEADERS" ]; then
            export CUDA_HOME="$NVCC_DIR"
            export CUDACXX="$nvcc"
            USE_SYSTEM_CUDA=true
            echo "Using system CUDA: $(nvcc --version 2>/dev/null | tail -1)"
            echo "CUDA_HOME=$CUDA_HOME (headers at $HEADERS)"
            break
        else
            echo "Warning: nvcc found but headers not accessible"
        fi
    fi
done

if [ "$USE_SYSTEM_CUDA" = false ]; then
    # Fallback to conda install
    echo "No usable system CUDA. Installing Miniconda + CUDA toolkit..."
    
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
        
        # Find where conda put the headers
        CONDA_NVCC_DIR=$(dirname $(which nvcc))
        HEADERS=$(find_cuda_headers "$CONDA_NVCC_DIR")
        
        if [ -n "$HEADERS" ]; then
            export CUDA_HOME="/root/miniconda3"  # conda root
            export CUDACXX="$(which nvcc)"
            echo "Using conda CUDA with headers at: $HEADERS"
        else
            echo "ERROR: Could not find CUDA headers after conda install"
            exit 1
        fi
    else
        echo "ERROR: CUDA installation failed"
        exit 1
    fi
fi

echo ""
echo "=== Step 2: Clone and build llama.cpp with CUDA ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
export CUDACXX=$(which nvcc)
echo "Building with: $CUDACXX (CUDA_HOME=$CUDA_HOME)"

# Find include directory for this CUDA installation
HEADERS=$(find_cuda_headers "$(dirname $(dirname "$CUDACXX"))")
if [ -z "$HEADERS" ]; then
    # Fallback to standard locations
    HEADERS="${CUDA_HOME}/include"
    if [ ! -d "$HEADERS" ] && [ -d "${CUDA_HOME}/targets/x86_64-linux/include" ]; then
        HEADERS="${CUDA_HOME}/targets/x86_64-linux/include"
    fi
fi

echo "Using CUDA headers: $HEADERS"

# Build llama.cpp with explicit include paths
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_SERVER=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_PATH="$CUDA_HOME" \
    -DCUDACXX="$CUDACXX" \
    -DCMAKE_C_FLAGS="-I$HEADERS" \
    -DCMAKE_CXX_FLAGS="-I$HEADERS" \
    -DCMAKE_CUDA_FLAGS="-I$HEADERS"

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
