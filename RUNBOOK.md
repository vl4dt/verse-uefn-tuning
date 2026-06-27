# Runbook: Inference Check + Synthetic Generation

## Step 1 — Free VRAM and start llama-server

```bash
# Kill LM Studio (frees ~24 GB VRAM)
pkill -f lmstudio || true
pkill -f "llama.cpp-linux-x86_64-nvidia-cuda" || true
sleep 3

# Verify VRAM is free
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# Run the inference check (starts llama-server, runs test prompt, reports results)
bash scripts/run_inference_check.sh
```

The script will:
1. Kill any existing llama-server processes
2. Start Qwen3.6-27B-MTP on port 8080 (~20 GPU layers, MTP speculative decoding)
3. Send a Verse coding prompt and save results to `data/models/inference_check.json`
4. Print tok/s speed and generated code quality

**Leave the server running after this step.** It stays warm for synthetic generation.

## Step 2 — Run synthetic data generation (after confirming inference works)

```bash
# Default: 5000 samples, temperatures 0.3/0.5/0.7/0.9
python3 scripts/generate_synthetic.py

# Custom target
python3 scripts/generate_synthetic.py --target-samples 8000

# Resume interrupted run
python3 scripts/generate_synthetic.py --resume
```

Output: `data/seeds/synthetic_raw.jsonl` (Alpaca format)

## Step 3 — When done, kill the server

```bash
pkill -f "llama-server.*Qwen3.6" || true
```
