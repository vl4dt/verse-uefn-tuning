#!/usr/bin/env bash
# verse-uefn-tuning-s87: Verify Qwen3.6 MTP inference works locally
# Run this after killing LM Studio to free VRAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS="$PROJECT_DIR/data/models/inference_check.json"
LOG_FILE="$PROJECT_DIR/data/models/llama-server.log"

LLAMA_SERVER="/home/vl4dt/LLM-AI-Tooling/llama.cpp/build/bin/llama-server"
MODEL_Q4="/home/vl4dt/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf"
MODEL_UD="/home/vl4dt/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-UD-Q4_K_XL.gguf"

# Pick model (prefer UD variant if available, fallback to Q4)
if [[ -f "$MODEL_UD" ]]; then
    MODEL="$MODEL_UD"
    echo "Using UD-Q4_K_XL variant"
elif [[ -f "$MODEL_Q4" ]]; then
    MODEL="$MODEL_Q4"
    echo "Using Q4_K_M variant"
else
    echo "ERROR: No GGUF found. Run 'unsloth studio run' first."
    exit 1
fi

echo "Model: $MODEL"
echo ""

# Kill any existing llama-server or LM Studio backends
echo "[1/5] Killing existing llama-server processes..."
pkill -f "llama-server.*Qwen3.6" 2>/dev/null || true
sleep 2

# Check VRAM availability (strip non-numeric suffix)
echo "[2/5] Checking VRAM..."
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | grep -oP '^\d+')
echo "  Free VRAM: ${FREE_VRAM} MiB"
if (( FREE_VRAM < 16000 )); then
    echo "  WARNING: Less than 16 GB free. You may need to kill LM Studio."
    echo "  Try: pkill -f lmstudio || killall llama.cpp-linux-x86_64-nvidia-cuda"
fi

# Start llama-server with MTP settings and explicit alias matching the API request
echo "[3/5] Starting llama-server (MTP, ~20 GPU layers)..."
$LLAMA_SERVER \
    -m "$MODEL" \
    --alias Qwen3.6-27B-MTP \
    -ngl 99 \
    --flash-attn on \
    --cont-batching \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --ctx-size 65536 \
    --parallel 1 \
    --temp 0.7 \
    --top-p 0.80 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.2 \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    --spec-draft-n-min 1 \
    --spec-draft-p-split 0.10 \
    --spec-draft-p-min 0.05 \
    --spec-draft-ngl all \
    --spec-draft-type-k q4_0 \
    --spec-draft-type-v q4_0 \
    --cache-reuse 256 \
    --host 127.0.0.1 \
    --port 18080 \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready (health endpoint + model loaded)
echo "[4/5] Waiting for server to start..."
for i in $(seq 1 120); do
    HEALTH=$(curl -s http://127.0.0.1:18080/health 2>/dev/null || echo "")
    if [[ "$HEALTH" == *"ok"* ]]; then
        echo "  Server ready after ${i}s"
        # Give model a few more seconds to fully load into memory
        sleep 3
        break
    fi
    if (( i % 15 == 0 )); then
        echo "  ... still waiting (${i}s)"
    fi
    sleep 1
done

# Verify health endpoint
if ! curl -s http://127.0.0.1:18080/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start. Check $LOG_FILE"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run test inference with a Verse prompt
echo "[5/5] Running test inference..."
curl -s http://127.0.0.1:18080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.6-27B-MTP",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Verse programmer for Unreal Editor for Fortnite (UEFN). Generate correct, idiomatic Verse code."
            },
            {
                "role": "user",
                "content": "Write a Verse class that extends creative_device and tracks player kills. It should have an event that fires when a kill happens, and a function to get the total kill count for a player."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }' > "$RESULTS" 2>&1

# Check if response is valid JSON
if ! python3 -c "import json; json.load(open('$RESULTS'))" 2>/dev/null; then
    echo "ERROR: Server returned non-JSON response:"
    head -5 "$RESULTS"
    echo ""
    echo "Check server log: $LOG_FILE"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Parse and display results
python3 << 'PYEOF'
import json, sys

with open("data/models/inference_check.json") as f:
    data = json.load(f)

usage = data.get("usage", {})
choices = data.get("choices", [])

print()
print("=" * 60)
print("INFERENCE CHECK RESULTS")
print("=" * 60)
print(f"Status: {'OK' if choices else 'FAILED'}")
if choices:
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    print(f"Output tokens: {usage.get('completion_tokens', '?')}")
    print(f"Total tokens:   {usage.get('total_tokens', '?')}")
    print()
    print("--- Generated Verse Code (first 80 lines) ---")
    for line in content.split("\n")[:80]:
        print(line)
    if len(content.split("\n")) > 80:
        print(f"... ({len(content.split('\n')) - 80} more lines)")

# Check for timing info
prompt_tokens = usage.get("prompt_tokens", 0)
completion_tokens = usage.get("completion_tokens", 0)
if prompt_tokens and completion_tokens:
    print()
    print("(Check llama-server.log for tok/s metrics)")
PYEOF

echo ""
echo "Server still running on port 8080 (PID $SERVER_PID)"
echo "Kill it with: kill $SERVER_PID"
echo "Full results saved to: $RESULTS"
echo "Server log: $LOG_FILE"
