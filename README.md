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

### 2. Clone and install dependencies

```python
# Cell 1 — Clone repo + install deps
!git clone https://github.com/YOUR_USERNAME/verse-uefn-tuning.git
%cd verse-uefn-tuning
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
!pip install -r requirements.txt -q
```

### 3. Run synthetic generation (HuggingFace backend)

```python
# Generate 5000 samples (~30–60 min on T4, ~10 min on A100)
import subprocess, os
os.chdir('/content/verse-uefn-tuning')
subprocess.run([
    'python3', 'scripts/generate_synthetic.py',
    '--backend', 'huggingface',
    '--workers', str(os.cpu_count() or 2),
    '--target-samples', '5000'
], check=True)

# Or resume an interrupted run:
subprocess.run(['python3', 'scripts/generate_synthetic.py', '--resume'], check=True)
```

Output: `output/synthetic_data.jsonl` (Alpaca-format instruction-response pairs)

---

## Architecture Details

### Teacher Model: Qwen3.6-27B MTP (HuggingFace)
The [Qwen3.6-27B-MTP](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) model is loaded via `transformers` + `accelerate`. The HuggingFace backend uses standard PyTorch inference — no separate server needed, no quantization required (full precision).

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

### HuggingFace Backend Notes
- Uses `transformers` + `accelerate` — no separate server needed
- Full precision (no quantization) — model loaded into GPU VRAM via PyTorch
- Auto-selects GPU if available, falls back to CPU

### VRAM Requirements

| GPU | Available VRAM | Notes |
|-----|---------------|-------|
| T4 (free) | 16 GB | Full precision ~52 GB won't fit — use `--max-workers 1` or switch to QLoRA fine-tuning later |
| A100/L4 (paid) | 24–80 GB | Comfortable for full precision with multiple workers |

> **Note:** The HuggingFace backend runs at full precision (~52 GB for Qwen3.6-27B). On T4 (16 GB), you'll need to use fewer workers or rely on CPU offloading via `accelerate`. For production generation, an A100 with 80 GB is recommended.

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
- [Qwen3.6-27B-MTP](https://huggingface.co/Qwen/Qwen3.6-27B-MTP) — Teacher model (HuggingFace)
