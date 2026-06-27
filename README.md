# Verse Synthesis — Synthetic Data Generation for UEFN Verse Fine-tuning

Fine-tune a small LLM to understand and generate **Verse**, Epic Games' programming language for [UEFN](https://www.epicgames.com/fortnite/creator) (Unreal Editor for Fortnite).

## Pipeline Overview

```
Seed Data → Synthetic Generation → Quality Filtering → QLoRA Fine-tuning
(1560 seeds)   via Qwen3.6-27B          Dedup + length       Unsloth SFT
```

### What's in this repo

| File | Purpose |
|------|---------|
| `data/seeds/unified_seeds.jsonl` | 1,562 curated Verse code seeds (from Book of Verse, GitHub repos, community content) |
| `scripts/generate_synthetic.py` | Main generator — produces instruction-response pairs via llama.cpp server API |
| `scripts/consolidate_seeds.py` | Merge scraped sources into unified seed format (already ran locally) |
| `setup_colab.sh` | One-command setup for Google Colab GPU instances |

### Seed Sources (1,562 total)

- **GitHub Verse files** — 702 real `.verse` source files from UEFN community repos
- **Book of Verse** — 1,035 code examples scraped from the official language reference
- **MD snippets** — 37 curated examples from awesome-verse and Versus community book

---

## Quick Start: Google Colab

### 1. Set up GPU runtime
- Colab → Runtime → Change runtime type → **T4 GPU** (free) or A100/L4 (paid)
- Minimum VRAM needed: ~18 GB for Q4_K_M quantization

### 2. Clone and run setup

```python
# Cell 1 — Clone repo
!git clone https://github.com/YOUR_USERNAME/verse-uefn-tuning.git
%cd verse-uefn-tuning

# Cell 2 — Build llama.cpp + download model (~5 min)
!bash setup_colab.sh
```

### 3. Run synthetic generation

```python
# Cell 3 — Generate 5000 samples (est. 30–60 min on T4, ~10 min on A100)
!python3 scripts/generate_synthetic.py \
    --llama-server llama.cpp/build/bin/llama-server \
    --model-q4 models/Qwen3.6-27B-Q4_K_M.gguf \
    --workers 4 \
    --target-samples 5000

# Or resume an interrupted run:
!python3 scripts/generate_synthetic.py --resume
```

Output: `data/seeds/synthetic_raw.jsonl` (Alpaca-format instruction-response pairs)

---

## Architecture Details

### Teacher Model: Qwen3.6-27B MTP
The [Qwen3.6-27B-MTP](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) model uses **Multi-Token Prediction** speculative decoding baked into the GGUF — no extra config needed for ~1.4–2.2× faster inference.

### Generation Strategies (4 methods, weighted sampling)

| Strategy | Weight | What it does |
|----------|--------|-------------|
| NL-to-Code | 40% | Natural language → Verse code (primary fine-tuning target) |
| Self-Instruct | 25% | Expand seed examples into diverse variants |
| Evol-Instruct | 20% | Mutate instructions to increase complexity |
| Code-to-Explanation | 15% | Explain existing Verse code in plain English |

### System Prompt
The generator is fed a comprehensive Verse syntax cheat sheet as the system prompt covering:
- Assignment (`:=` immutable, `var`+`set` mutable)
- Failable expressions `[...]#`, must-succeed `(())`
- Effect specifiers (`<computes>`, `<reads>`, `<writes>`, etc.)
- Structured concurrency (`sync:`, `race:`, `rush:`, `branch:`)
- Module imports, events, classes, interfaces

### Few-Shot Context
Each generation call includes 2–3 randomly sampled seeds from `unified_seeds.jsonl` as few-shot demonstrations — this is critical for preventing Verse syntax hallucination.

---

## Colab Configuration

### llama.cpp Server Arguments (auto-applied by script)

```bash
--n-gpu-layers 99 --flash-attn on --cont-batching \
--cache-type-k q4_0 --cache-type-v q4_0 \
--ctx-size 65536 --parallel <workers> \
--temp 0.7 --top-p 0.80 --top-k 20 \
--spec-type draft-mtp --spec-draft-n-max 4 \
--spec-draft-p-split 0.10
```

### VRAM Requirements

| GPU | Available VRAM | Quantization | Notes |
|-----|---------------|--------------|-------|
| T4 (free) | 16 GB | Q3_K_M / Q4_K_M (tight) | Use fewer parallel workers |
| A100/L4 (paid) | 24 GB | Q4_K_M comfortable | Recommended for fastest generation |

---

## Post-Generation: Quality Filtering

After generation completes, run filtering to clean the dataset:

```python
# Deduplication + length filtering + syntax validation
# See RESEARCH.md Phase 3 for full pipeline pseudocode
```

Expected output after filtering: ~3K–4K high-quality samples from 5K raw.

---

## Fine-tuning (Phase 4 — not in this repo)

After filtering, fine-tune a student model with Unsloth + QLoRA:

| Student Model | Size | Recommended For |
|---------------|------|-----------------|
| Qwen2.5-Coder-7B | 7B | Best starting point (strong code priors) |
| Llama-3.1-8B-Instruct | 8B | Strong general instruction following |
| Mistral-Nemo-12B | 12B | Larger capacity if you have enough data |

See `RESEARCH.md` Phase 4 for the full Unsloth training script and hyperparameters.

---

## References

- [Book of Verse](https://verselang.github.io/book/) — Official language reference
- [awesome-verse](https://github.com/spilth/awesome-verse) — Community resource list
- [Qwen3.6-27B-MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) — Teacher model quantization
- [llama.cpp MTP guide](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/)
