# Verse Coding Model: Research & Plan

Fine-tune an open-weight model to understand and generate **Verse** — Epic Games' programming language for Unreal Editor for Fortnite (UEFN).

---

## Hardware Profile

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | NVIDIA RTX 3090, 24 GB VRAM | Enough for QLoRA SFT of 7–8B models; ~20 layers of Qwen3.6-27B Q4 |
| CPU | AMD Ryzen 9 5950X (16C/32T) | Strong offload partner for llama.cpp |
| RAM | 62 GB DDR4 | Plenty for model offloading + data processing |

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│  PHASE 1: Seed Collection (manual, ~50–100 examples)           │
│  Scrape real Verse code from repos, docs, and community books   │
└──────────────────────┬────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  PHASE 2: Synthetic Data Generation (Qwen3.6-27B MTP)         │
│  Run locally via llama.cpp with MTP speculative decoding       │
│  Self-Instruct + Evol-Instruct → 5K–20K instruction pairs      │
└──────────────────────┬────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  PHASE 3: Quality Filtering                                    │
│  Deduplication, length filtering, basic syntax checks          │
└──────────────────────┬────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  PHASE 4: Fine-tuning (Unsloth + QLoRA SFT)                   │
│  Student model: Llama-3.1-8B or Qwen2.5-Coder-7B              │
│  Format: Alpaca JSONL → Unsloth FastLanguageModel             │
└───────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Seed Data Sources

### Primary Source: Your Verse Skill ⭐

Your pi-agent skill at `~/.pi/agent/skills/verse-uefn/` is the **highest-quality seed source** available. It contains:

| File | Content | Synthetic Value |
|------|---------|----------------|
| **SKILL.md** | Language cheat sheet, workflows, key patterns, best practices checklist | System prompt material + few-shot templates for Qwen |
| **REFERENCE.md** | Full language reference (12 sections: primitives → persistence) | Authoritative syntax ground truth — prevents hallucination |
| **EXAMPLES.md** | 12 complete working code examples | Direct seed examples for Self-Instruct expansion |

#### Examples inventory (from EXAMPLES.md)

1. Custom Device Template (`creative_device` with `@editable`, lifecycle hooks)
2. Game Manager with Player Tracking (multiplayer join/leave, `map<int, player>`)
3. Score System (`map<int, int>`, win condition, point tracking)
4. Round-Based Game Loop (warmup → gameplay → cooldown phases, `branch:` loop)
5. Health/Damage System (`damagable_entity`, `<decides>`, block init)
6. Inventory System (item stacking, weight limits, `struct<computes>`)
7. Timer/Countdown Device (visual countdown, completion event)
8. Spawn Point Manager (cycle through spawn points, teleporter_device)
9. Custom NPC Behavior (`creative_npc_agent`, patrol loop)
10. Event Bus Pattern (decoupled communication, `enum<computes>`)
11. Persistable Save System (`struct<persistable>`, save/load lifecycle)
12. Concurrent Initialization (`sync:` parallel init pattern)

**These 12 examples alone cover**: failable expressions, speculative execution, structured concurrency (`sync`/`race`/`rush`/`branch`), effect system (`<computes>`/`<reads>`/`<writes>`/`<transacts>`/`<decides>`/`<suspends>`), mutability (`var`/`set`), containers (arrays, maps, optionals), classes/interfaces, `@editable`, multiplayer patterns, and persistence.

### Secondary Sources (supplemental)

| Source | URL | Content |
|--------|-----|---------|
| **awesome-verse** | <https://github.com/spilth/awesome-verse> | Curated list of repos, docs, Discord servers, YouTube channels |
| **Book of Verse** | <https://verselang.github.io/book/> | Full language reference with code examples (primitives, containers, classes, concurrency, effects) |
| **Versus** | <https://github.com/glinesbdev/versus> | Community-driven learning book for Verse + UEFN |
| **uefn** | <https://github.com/kbfngg/uefn> | UEFN API knowledge base + public API diffs over time |
| **VerseProgrammingLanguage** | <https://github.com/UnrealVerseGuru/VerseProgrammingLanguage> | Snippets, official resources aggregation |
| **Epic Verse Docs** | <https://dev.epicgames.com/documentation/en-us/unreal-editor-for-fortnite/verse-language-reference> | Official language reference (API surface) |

### What to extract from each source

- Code snippets with surrounding natural-language context
- Function signatures and class definitions
- API usage patterns (Gameplay, UI, Networking, Persistence)
- Error handling / failable expression examples
- Concurrency patterns (`sync`, `race`, `rush`, `branch`)

---

## Phase 2 — Synthetic Data Generation

### Teacher Model: Qwen3.6-27B MTP

Qwen3.6 is Alibaba's hybrid-thinking model with **Multi-Token Prediction (MTP)** built in. Key advantages for our use case:

- **1.4–2.2× faster inference** via speculative decoding (no extra config needed — it's baked into the GGUF)
- Strong coding capabilities at 27B scale
- Runs on our hardware at Q4 quantization (~18 GB total memory, ~20 layers on GPU + CPU offload)

### Quantization & Memory Requirements

| Quant | Total Memory | VRAM Needed (est.) | Speed (est. tok/s) |
|-------|-------------|-------------------|--------------------|
| Q3_K_M | 15 GB | ~14 GB | Fast, slight quality loss |
| **Q4_K_M** | **18 GB** | **~17 GB** | **Good balance** ← recommended |
| Q6_K | 24 GB | ~22 GB | Best quality, tight on VRAM |

### Running via llama.cpp

```bash
# Build with CUDA support
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_SERVER=ON
cmake --build build --config Release -j32

# Download GGUF (Unsloth Dynamic 2.0 quant recommended)
# Source: HuggingFace → unsloth/Qwen3.6-27B-Instruct-* or TheBloke/Marcello mirrors

# Start server with MTP enabled
./build/bin/llama-server \
  -m qwen3.6-27b-instruct-q4_k_m.gguf \
  --n-gpu-layers 20 \
  --ctx-size 32768 \
  --temp 0.7 \
  --top-p 0.95 \
  -c 16384
```

### Generation Strategies

#### Skill Integration: How Your Verse Skill Improves Quality

Your skill provides **authoritative ground truth** that prevents Qwen from hallucinating Verse syntax. Three integration modes:

**Mode 1 — System Prompt Injection (primary)**

Feed the SKILL.md cheat sheet + key REFERENCE.md sections as the system prompt to llama.cpp. This gives Qwen in-context knowledge of real Verse syntax, effects, and patterns before it generates anything.

```
System: You are an expert Verse programmer for Unreal Editor for Fortnite (UEFN).

## Language Rules
[SKILL.md cheat sheet content — immutable-by-default, failable [], effects <computes>/<reads>/<writes>/<transacts>/<decides>/<suspends>, structured concurrency sync/race/rush/branch]

## Key Patterns
[SKILL.md key patterns section — failable expressions, speculative execution, event-driven]

## Best Practices
[SKILL.md best practices checklist]
```

**Mode 2 — Few-Shot Examples (per generation call)**

Attach 2–3 relevant examples from EXAMPLES.md as few-shot demonstrations in each prompt:

```
User: Here are working Verse examples for reference:

Example 1 — Game Manager with Player Tracking:
[excerpt from EXAMPLES.md]

Example 2 — Score System:
[excerpt from EXAMPLES.md]

Now generate 5 new instruction-response pairs covering similar multiplayer patterns but with different mechanics.
```

**Mode 3 — Post-Generation Validation (quality gate)**

After Qwen generates a sample, use pi-agent **with the verse-uefn skill loaded** to review it:

```
pi-agent: Review this generated Verse code for correctness. Check:
1. Does it use valid Verse syntax?
2. Are effect specifiers correct?
3. Are failable expressions handled properly?
4. Does it follow the best practices from SKILL.md?
```

This catches hallucinated APIs or wrong effect combinations before they enter the dataset.

#### A. Self-Instruct Expansion

Take a seed example, generate diverse variants:

```json
{
  "system": "You are an expert Verse programmer for Unreal Editor for Fortnite (UEFN). Generate diverse coding instructions and solutions in Verse.",
  "user": "Here is a seed example of Verse code. Generate 5 new instruction-response pairs that cover similar concepts but with different scenarios, edge cases, and complexity levels.\n\nSeed:\n{seed_example}",
}
```

#### B. Evol-Instruct (Difficulty Mutation)

Increase complexity of existing instructions:

```json
{
  "user": "Evolve this instruction to be more complex: add constraints, require error handling with failable expressions, or combine with another Verse API concept.\n\nOriginal: {original_instruction}"
}
```

#### C. Code-to-Explanation (Bidirectional)

Generate natural-language explanations from code:

```json
{
  "user": "Explain what this Verse code does in plain English, including the UEFN context and which APIs are used:\n\n{verse_code_snippet}"
}
```

#### D. NL-to-Code (Primary Fine-tuning Target)

Natural language → Verse code generation:

```json
{
  "user": "Write a Verse function that {natural_language_description}. Use the {specific_API} API.",
}
```

### Expected Output Volume

With your skill as the seed source, we have a strong foundation of 12 complete examples + full language reference.

| Strategy | Seeds × Multiplier | Est. Samples |
|----------|-------------------|-------------|
| Self-Instruct (5×) from EXAMPLES.md | 12 examples × 5 variants × 3 temps | ~180 |
| Evol-Instruct (3 rounds, difficulty ramp) | 180 × 3 | ~540 |
| NL-to-Code from REFERENCE.md sections | 12 reference sections × 5 prompts each × 3 temps | ~180 |
| Code-to-Explanation (bidirectional) | All code blocks from skill (~60 snippets) | ~60 |
| API surface expansion (from REFERENCE.md UEFN patterns) | creative_device, NPC, game_instance, persistence × variants | ~200 |
| **Total per pass** | | **~1,160** |

Run 5–8 passes with different temperatures (0.3, 0.5, 0.7, 0.9) and random seeds → **5K–10K samples before filtering**.

### Skill as Quality Gate

Your skill enables a validation step that pure web-scraped approaches can't match:

```
Generated Sample ──► pi-agent (verse-uefn skill loaded)
                      │
                      ├─ Valid syntax? ✓/✗
                      ├─ Correct effects? ✓/✗  
                      ├─ Follows best practices? ✓/✗
                      └─ Score ≥ threshold? → Keep / Discard
```

This catches hallucinated APIs, wrong effect combinations (e.g., `<computes><transacts>`), and missing failable handling — the most common failure modes when generating code for an unfamiliar language.

---

## Phase 3 — Quality Filtering

### Pipeline

```python
# Pseudocode for filtering pipeline
def filter_dataset(raw_samples: list[dict]) -> list[dict]:
    # 1. Deduplication (MinHash or simple text similarity)
    deduplicated = remove_duplicates(raw_samples, threshold=0.85)

    # 2. Length filtering
    filtered = [s for s in deduplicated
                if 20 < len(s["instruction"]) < 500
                and 30 < len(s["output"]) < 4000]

    # 3. Basic syntax checks (if Verse parser available, otherwise regex)
    valid = [s for s in filtered if contains_verse_syntax(s["output"])]

    # 4. Remove near-duplicates of output code
    final = remove_code_duplicates(valid, threshold=0.9)

    return final
```

### Key Filters

- **Deduplication**: MinHash with Jaccard similarity ≥ 0.85 → collapse clusters
- **Length bounds**: Skip trivially short or excessively long samples
- **Code presence**: Output must contain recognizable Verse syntax (`fn`, `class`, `var`, `use`, `if`, `for`)
- **Instruction quality**: Must be a complete, actionable prompt (not fragments)

---

## Phase 4 — Fine-tuning with Unsloth

### Student Model Options

| Model | Size | Why Consider | VRAM for QLoRA SFT |
|-------|------|-------------|-------------------|
| **Llama-3.1-8B-Instruct** | 8B | Strong general instruction following, well-supported by Unsloth | ~6–7 GB |
| **Qwen2.5-Coder-7B** | 7B | Already code-specialized, may need less Verse-specific data | ~5–6 GB |
| **Mistral-Nemo-12B** | 12B | Larger capacity if we have enough data | ~9–10 GB |

**Recommendation**: Start with **Qwen2.5-Coder-7B** — it already has strong code priors, so the Verse-specific fine-tuning can focus on language syntax and API patterns rather than teaching coding from scratch.

### Dataset Format: Alpaca JSONL

Unsloth's `FastLanguageModel` works with Alpaca-style instruction data:

```jsonl
{"instruction": "Create a Verse class that tracks player score with increment and reset methods", "input": "", "output": "class ScoreTracker {\n  var m_Score: i64 = 0;\n\n  fn Increment(this, amount: i64) {\n    this.m_Score += amount;\n  }\n\n  fn Reset(this) {\n    this.m_Score = 0;\n  }\n}"}
```

Or ShareGPT conversation format for multi-turn:

```jsonl
{"conversations": [
  {"from": "human", "value": "How do I use failable expressions in Verse?"},
  {"from": "gpt", "value": "Failable expressions use the `?` operator..."}
]}
```

### Unsloth Fine-tuning Script (Skeleton)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",  # or Llama-3.1-8B
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported

dataset = load_dataset("json", data_files="verse_synthetic.jsonl", split="train")

# Alpaca formatting via Unsloth's chat template
def format_alpaca(example):
    return {
        "text": tokenizer.apply_chat_template([
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ], tokenize=False, add_generation_prompt=False)
    }

dataset = dataset.map(format_alpaca)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,          # adjust based on dataset size
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,
    ),
)

trainer.train()
model.save_pretrained("verse-lora")
model.save_pretrained_gguf("verse-gguf", tokenizer)
```

### Hyperparameter Notes

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (LoRA rank) | 16 | Standard for domain adaptation; bump to 32 if data > 10K |
| `lora_alpha` | 16 | = r is standard; some use 2×r (32) for stronger adaptation |
| `learning_rate` | 2e-4 | Standard QLoRA LR |
| `max_seq_length` | 4096 | Covers most Verse functions + explanations |
| `gradient_accumulation` | 4 | Effective batch size = 8, fits in VRAM |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Verse is niche** — limited real code exists | Low-quality synthetic data if seeds are weak | **MITIGATED**: Your skill provides authoritative ground truth (12 examples + full reference) as system prompt + few-shot context |
| **Qwen3.6 may hallucinate Verse syntax** | Synthetic data contains invalid code | **REDUCED**: Skill in system prompt constrains generation; post-generation validation via pi-agent with skill catches remaining errors |
| **Small dataset → overfitting** | Model memorizes instead of generalizing | Use small LoRA rank (r=8–16), early stopping, keep epochs ≤ 3 |
| **MTP GGUF availability** | May need to convert from non-MTP format | Unsloth provides Dynamic 2.0 GGUFs; check HuggingFace first |

---

## Open Questions

- [ ] Should we include **preference data** (chosen/rejected pairs) for DPO after SFT?
- [ ] Is there a Verse linter/parser we can use for automated syntax validation of generated code?
- [ ] Should the student model be **instruct-tuned or base**? Instruct is safer — it already follows instructions, so we only teach Verse domain.
- [ ] How many epochs? With ~5K samples and 7B model, 2–3 epochs is probably right.

---

## Your Verse Skill (Primary Seed Source)

Location: `~/.pi/agent/skills/verse-uefn/` (local, not yet on GitHub)

Package: `@vl4dt/verse-uefn-skill` v1.0.0

Files:
- `skills/verse-uefn/SKILL.md` — Core cheat sheet, workflows, patterns, best practices
- `skills/verse-uefn/REFERENCE.md` — Full language reference (12 sections)
- `skills/verse-uefn/EXAMPLES.md` — 12 complete working code examples
- `package.json` — Pi skill package manifest

**Role in pipeline**: System prompt content + few-shot examples for Qwen3.6 data generation, plus post-generation validation via pi-agent with the skill loaded.

## References
- Fine-tuning Guide: <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>
- Datasets Guide: <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide>
- Qwen3.6 Local Setup: <https://unsloth.ai/docs/models/qwen3.6>

### Synthetic Data Research
- OpenCodeInstruct (NVIDIA, 2025): <https://arxiv.org/html/2504.04030v1> — 5M synthetic code instruction pairs
- Self-Instruct (Wang et al., 2022): Bootstrap from seeds → expand via model itself
- Evol-Instruct (Xu et al., 2024): Mutate instructions for difficulty diversity
- Eugene Yan's Synthetic Data Guide: <https://eugeneyan.com/writing/synthetic/>

### Verse Language Resources
- Book of Verse: <https://verselang.github.io/book/>
- Versus (community book): <https://github.com/glinesbdev/versus>
- awesome-verse: <https://github.com/spilth/awesome-verse>
- UEFN API KB: <https://github.com/kbfngg/uefn>

### llama.cpp MTP for Qwen3.6
- GitHub Gist (MTP setup): <https://gist.github.com/eeshansrivastava89/85797104af34181944bfd1360d69e8af>
- Mer.vin guide: <https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/>
