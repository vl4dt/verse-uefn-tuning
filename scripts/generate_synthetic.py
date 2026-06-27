#!/usr/bin/env python3
"""Synthetic Verse data generation via llama.cpp server API.

Starts and stops its own llama-server with matching --parallel/--workers.
Uses unified seed data as few-shot context, then generates instruction-response
pairs through four strategies: Self-Instruct, Evol-Instruct, NL-to-Code, and
Code-to-Explanation.

Usage:
    python3 scripts/generate_synthetic.py --target-samples 2000 --workers 4

Output: data/seeds/synthetic_raw.jsonl (Alpaca format)
Checkpoint: saved after every batch so interrupted runs don't lose data.
Ctrl+C: saves current progress and exits cleanly on first press.
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# Path resolution: check multiple locations for llama-server and model files
def _resolve_path(*candidates: str) -> str:
    """Return first existing path from candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    # Last candidate is the fallback (will cause clear error if missing)
    return candidates[-1]


def _find_llama_server() -> str:
    """Search common locations for llama-server binary."""
    candidates = [
        os.environ.get("LLAMA_SERVER", ""),  # explicit override
        "./llama.cpp/build/bin/llama-server",
        "/home/vl4dt/LLM-AI-Tooling/llama.cpp/build/bin/llama-server",
        os.path.expanduser("~/LLM-AI-Tooling/llama.cpp/build/bin/llama-server"),
    ]
    return _resolve_path(*[c for c in candidates if c])


DEFAULT_LLAMA_SERVER = _find_llama_server()
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", DEFAULT_LLAMA_SERVER)
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MODEL_Q4 = os.environ.get(
    "MODEL_Q4",
    f"{HF_CACHE}/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/"
    f"snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf",
)
MODEL_UD = os.environ.get(
    "MODEL_UD",
    f"{HF_CACHE}/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/"
    f"snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-UD-Q4_K_XL.gguf",
)

SEEDS_FILE = Path("data/seeds/unified_seeds.jsonl")
OUTPUT_FILE = Path("data/seeds/synthetic_raw.jsonl")
DEBUG_LOG  = Path("data/seeds/generation_debug.log")

# Shared state (GIL makes simple ints thread-safe)
shutdown_requested = False
executor_shutting_down = False
task_counter_lock = threading.Lock()
task_id_counter = 0
total_completed = 0
pending_count = 0
# Task timing: maps task_id -> {"start": ts, "rate": samples/sec}
task_timing_lock = threading.Lock()
task_timings: dict[int, dict] = {}
# Recent per-task sample rates (samples/sec) for ETA estimation
task_rate_history: list[float] = []
debug_log_lock = threading.Lock()


def handle_sigint(signum, frame):
    """On Ctrl+C: save what we have and exit cleanly.

    First press sets the flag so the main loop can break and checkpoint.
    Second press force-exits with os._exit to avoid raising SystemExit
    inside threading._python_exit.join() which causes
    'Exception ignored on threading shutdown' during interpreter teardown.
    """
    global shutdown_requested, executor_shutting_down
    if shutdown_requested:
        os._exit(1)  # force exit; safe in signal handler, avoids atexit/threading cleanup
    shutdown_requested = True
    executor_shutting_down = True


signal.signal(signal.SIGINT, handle_sigint)


def dlog(msg: str):
    """Append to debug log file (thread-safe). Survives stdout/stderr capture."""
    with debug_log_lock:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(line)


def p(*args, **kwargs):
    """Print with flush=True to avoid stdout buffering."""
    print(*args, flush=True, **kwargs)


# Server management

_server_proc: subprocess.Popen | None = None


def start_llama_server(host: str, port: int, parallel: int) -> subprocess.Popen:
    """Start llama-server with MTP tuning matching Windows config."""
    model = MODEL_UD if Path(MODEL_UD).exists() else MODEL_Q4
    if not Path(model).exists():
        p(f"ERROR: Model not found at {model}", file=sys.stderr)
        sys.exit(1)

    variant = "UD-Q4_K_XL" if "UD" in model else "Q4_K_M"
    p(f"Model:       {Path(model).name} ({variant})")

    cmd = [
        LLAMA_SERVER, "-m", model, "--alias", "Qwen3.6-27B-MTP",
        "-ngl", "99", "--flash-attn", "on", "--cont-batching",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--ctx-size", "65536", "--parallel", str(parallel),
        "--temp", "0.7", "--top-p", "0.80", "--top-k", "20",
        "--min-p", "0.0",
        # MTP speculative decoding (tuned from Windows config)
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
        "--spec-draft-n-min", "1", "--spec-draft-p-split", "0.10",
        "--spec-draft-p-min", "0.05", "--spec-draft-ngl", "all",
        "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
        "--cache-reuse", "256",
        "--host", host, "--port", str(port),
    ]

    p(f"Starting llama-server (--parallel {parallel})...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    global _server_proc
    _server_proc = proc

    # Wait for server to be ready
    url = f"http://{host}:{port}"
    for i in range(120):
        try:
            req = Request(f"{url}/health")
            with urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    p(f"Server ready after {i + 1}s ({parallel} slots)")
                    time.sleep(2)  # let model warm up
                    return proc
        except Exception:
            pass
        if (i + 1) % 30 == 0:
            p(f"  ... waiting for server ({i + 1}s)")
        time.sleep(1)

    p("ERROR: Server failed to start within 120s", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def stop_llama_server():
    """Kill the llama-server subprocess."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        p("Stopping llama-server...")
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
            _server_proc.wait()
        p("Server stopped.")


# System prompt (authoritative Verse knowledge)

SYSTEM_PROMPT = """You are an expert Verse programmer for Unreal Editor for Fortnite (UEFN). Generate correct, idiomatic Verse code that follows the language's unique syntax and semantics.

## Key Verse Syntax Rules
- Assignment uses `:=` (immutable) or `var` + `set` (mutable)
- Failable expressions use square brackets: `Result := SomeFunction[Args]` — if it fails, execution stops
- Must-succeed calls use parentheses: `SomeFunction(Args)` — panics on failure
- Effect specifiers on functions: `<computes>`, `<reads>`, `<writes>`, `<transacts>`, `<decides>`, `<suspends>`
- Structs can have effect modifiers: `struct<computes>`, `struct<persistable>`
- Enums with computes: `enum<computes>`
- Classes extend base classes and implement interfaces: `class MyClass extends BaseClass implements Interface`
- Access modifiers: `public`, `protected`, `private` (default)
- Module imports: `using { /Verse.org/Simulation }` or `using { /Fortnite.com/Game }`
- Structured concurrency: `sync:` (parallel), `race:` (first wins), `rush:` (best effort), `branch:` (conditional parallel)
- Speculative execution: `if (X := expr): ... else: ...` — binds variable only in the true branch
- Arrays use `[]`, maps use `map<K, V>`, optionals use `optional<T>`
- Events for decoupled communication: `event MyEvent : void` + `OnMyEvent(this)`"""

# Generation strategies

def load_seeds(path: Path, max_samples: int = 200) -> list[dict]:
    """Load unified seeds, shuffle, and cap."""
    if not path.exists():
        p(f"ERROR: {path} not found. Run consolidate_seeds.py first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        seeds = [json.loads(line) for line in f]

    verse_seeds = [s for s in seeds if s["metadata"]["has_verse_syntax"]]
    other_seeds = [s for s in seeds if not s["metadata"]["has_verse_syntax"]]

    random.shuffle(verse_seeds)
    random.shuffle(other_seeds)

    selected = verse_seeds[:max_samples] + other_seeds[:20]
    p(f"Loaded {len(selected)} seeds ({len(verse_seeds)} with Verse syntax)")
    return selected


def api_chat(messages: list[dict], temperature: float, max_tokens: int = 2048) -> str | None:
    """Call llama.cpp OpenAI-compatible chat endpoint."""
    payload = {
        "model": "Qwen3.6-27B-MTP",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,  # Increased from 4096 to allow full generation of complex Verse prompts
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{CONFIG['server']}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    raw = ""
    try:
        with urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
            result = json.loads(raw)

            # Log the full response structure for diagnosis (first time only)
            if not hasattr(api_chat, "_logged_structure"):
                api_chat._logged_structure = True
                dlog(f"api_chat: response keys={list(result.keys())}")
                choices = result.get("choices", [])
                dlog(f"api_chat: {len(choices)} choice(s)")
                if choices:
                    msg = choices[0].get("message", {})
                    dlog(f"api_chat: message keys={list(msg.keys())}")
                    usage = result.get("usage", {})
                    dlog(f"api_chat: usage={usage}")

            content = result["choices"][0]["message"]["content"]
            if not content or len(content.strip()) == 0:
                # Log the raw JSON so we can see what's actually in there
                dlog(f"api_chat: EMPTY response (temp={temperature}), full_json={raw[:500]}")
            else:
                preview = content[:200].replace("\n", "\\n")
                tail = content[-150:].replace("\n", "\\n") if len(content) > 350 else ""
                dlog(f"api_chat OK: {len(content)} chars, temp={temperature}, preview=...{preview}")
            return content
    except TimeoutError:
        msg = f"api_chat: TIMEOUT (300s), temp={temperature}"
        p(f"  ERROR: {msg}", file=sys.stderr)
        dlog(msg)
        return None
    except URLError as e:
        msg = f"api_chat: CONNECTION FAILED: {e}, temp={temperature}"
        p(f"  ERROR: {msg}", file=sys.stderr)
        dlog(msg)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        preview = raw[:300].replace("\n", "\\n")
        msg = f"api_chat: PARSE ERROR ({type(e).__name__}: {e}), temp={temperature}, raw=...{preview}"
        p(f"  ERROR: Response parse error: {e}", file=sys.stderr)
        dlog(msg)
        return None


def pick_few_shot(seeds: list[dict], n: int = 3) -> str:
    """Format n random seeds as few-shot examples."""
    examples = random.sample(seeds, min(n, len(seeds)))
    parts = []
    for i, seed in enumerate(examples, 1):
        code = seed["code"][:600]
        source_type = seed["source"]["type"]
        parts.append(f"Example {i} ({source_type}):\n```\n{code}\n```")
    return "\n\n".join(parts)


# Strategy A: Self-Instruct Expansion

SELF_INSTRUCT_PROMPTS = [
    "Here are working Verse examples for reference:\n\n{few_shot}\n\nGenerate {n} new instruction-response pairs covering similar concepts but with different scenarios, edge cases, and complexity levels. Format each as JSON: {{\"instruction\": \"...\", \"output\": \"...\"}}. Separate pairs with a blank line.",
    "Study these Verse code patterns:\n\n{few_shot}\n\nNow create {n} diverse coding tasks that test understanding of the same concepts (effects, failable expressions, concurrency). Each task should have a clear instruction and a complete Verse code solution. Output JSON array: [{{\"instruction\": \"...\", \"output\": \"...\"}}]",
    "Based on these Verse examples:\n\n{few_shot}\n\nGenerate {n} new problems that combine 2-3 of the demonstrated concepts in novel ways. Vary difficulty from beginner to advanced. Output as JSON array.",
]


def generate_self_instruct(seeds: list[dict], temperature: float, batch_size: int = 5) -> list[dict]:
    """Expand seed examples into new instruction-response pairs."""
    prompt_template = random.choice(SELF_INSTRUCT_PROMPTS)
    few_shot = pick_few_shot(seeds, n=3)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompt_template.format(few_shot=few_shot, n=batch_size),
        },
    ]

    result = api_chat(messages, temperature, max_tokens=2048)  # self_instruct: batch of 5 samples
    if not result:
        return []

    samples = parse_json_from_response(result)
    for s in samples:
        s["strategy"] = "self_instruct"
        s["temperature"] = temperature
    return samples


# Strategy B: Evol-Instruct (Difficulty Mutation)

EVOL_OPERATIONS = [
    "Make this instruction require error handling with failable expressions ([] syntax)",
    "Add a concurrency requirement using sync:, race:, or rush:",
    "Require the solution to use structured effects (<computes>, <reads>, <writes>)",
    "Combine this task with player state management and persistence (struct<persistable>)",
    "Make it multiplayer-aware: handle join/leave events and networked state",
    "Add constraints: must use @editable fields, implement an interface, and fire custom events",
    "Evolve into a full device class extending creative_device with lifecycle hooks",
]


def generate_evol_instruct(seeds: list[dict], temperature: float) -> list[dict]:
    """Mutate existing instructions to increase complexity."""
    seed = random.choice([s for s in seeds if s["metadata"]["has_verse_syntax"]])

    base_instruction = f"Write Verse code that does what this example demonstrates:\n```\n{seed['code'][:400]}\n```"
    evol_op = random.choice(EVOL_OPERATIONS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"I have this base instruction:\n{base_instruction}\n\n"
                f"{evol_op}. Generate 3 evolved (harder) versions of this instruction, "
                f"each with a complete Verse code solution. Output as JSON array: "
                f"[{{\"instruction\": \"...\", \"output\": \"...\"}}]"
            ),
        },
    ]

    result = api_chat(messages, temperature, max_tokens=1024)  # evol_instruct: 3 evolved versions
    if not result:
        return []

    samples = parse_json_from_response(result)
    for s in samples:
        s["strategy"] = "evol_instruct"
        s["temperature"] = temperature
    return samples


# Strategy C: NL-to-Code (Primary Fine-tuning Target)

NL_TO_CODE_PROMPTS = [
    "Write a Verse function that {task}. Use the {api} API.",
    "Create a Verse class extending creative_device that {task}.",
    "Implement a Verse system for {task} using structured concurrency.",
    "Build a Verse device that {task}, with @editable configuration fields.",
]

VERSE_TASKS = [
    ("tracks player score and displays it on screen", "/Fortnite.com/UI"),
    ("manages spawn points and teleports players to random locations", "/Fortnite.com/Characters"),
    ("handles round-based game flow (warmup, gameplay, cooldown)", "/Verse.org/Simulation"),
    ("creates a health bar that updates when damage is taken", "/UnrealEngine.com/Temporary/UI"),
    ("manages an inventory system with item stacking and weight limits", "/Fortnite.com/Game"),
    ("spawns NPCs that patrol between waypoints in a loop", "/Fortnite.com/Characters"),
    ("saves and loads player progress using persistable structs", "/Verse.org/Persistence"),
    ("creates a timer device with visual countdown and completion event", "/UnrealEngine.com/Temporary/UI"),
    ("handles multiplayer join/leave events and updates team scores", "/Fortnite.com/Game"),
    ("builds an event bus for decoupled communication between devices", "/Verse.org/Simulation"),
    ("creates a damage system with armor reduction and critical hits", "/Fortnite.com/Game"),
    ("manages power-ups that spawn randomly and expire after a duration", "/UnrealEngine.com/Temporary/SpatialMath"),
    ("builds a checkpoint system that saves player position on death", "/Verse.org/Persistence"),
    ("creates a voting system for players to choose game modes", "/Fortnite.com/UI"),
    ("implements a wave-based enemy spawner with increasing difficulty", "/Fortnite.com/Characters"),
]


def generate_nl_to_code(seeds: list[dict], temperature: float) -> list[dict]:
    """Generate natural-language-to-code instruction pairs."""
    task, api = random.choice(VERSE_TASKS)
    prompt_template = random.choice(NL_TO_CODE_PROMPTS)

    few_shot = pick_few_shot(seeds, n=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here are reference examples of correct Verse code:\n\n{few_shot}\n\n"
                f"Now write a complete solution for this task. Include all necessary imports, "
                f"class definitions, and event handlers. The code should be production-ready for UEFN.\n\n"
                f"Task: {prompt_template.format(task=task, api=api)}"
            ),
        },
    ]

    result = api_chat(messages, temperature, max_tokens=1536)  # nl_to_code: single code sample
    if not result:
        return []

    instruction = prompt_template.format(task=task, api=api)
    output = re.sub(r"^```(?:verse)?\s*\n?", "", result.strip(), flags=re.MULTILINE)
    output = re.sub(r"\n?```\s*$", "", output, flags=re.MULTILINE)

    return [{
        "instruction": instruction,
        "output": output,
        "strategy": "nl_to_code",
        "temperature": temperature,
    }]


# Strategy D: Code-to-Explanation (Bidirectional)

def generate_code_explanation(seeds: list[dict], temperature: float) -> list[dict]:
    """Generate natural-language explanations from code."""
    seed = random.choice([s for s in seeds if s["metadata"]["has_verse_syntax"]])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Explain what this Verse code does in plain English. Cover:\n"
                f"1. What the class/function does at a high level\n"
                f"2. Which UEFN APIs are used and why\n"
                f"3. How effects (<computes>, <reads>, etc.) work here\n"
                f"4. Any concurrency patterns (sync, race, rush, branch)\n"
                f"5. Edge cases or error handling\n\n"
                f"Code:\n```\n{seed['code'][:800]}\n```"
            ),
        },
    ]

    result = api_chat(messages, temperature, max_tokens=2048)
    if not result:
        return []

    instruction = f"Explain what this Verse code does:\n```\n{seed['code'][:600]}\n```"
    output = re.sub(r"^```\s*\n?", "", result.strip(), flags=re.MULTILINE)
    output = re.sub(r"\n?```\s*$", "", output, flags=re.MULTILINE)

    return [{
        "instruction": instruction,
        "output": output,
        "strategy": "code_to_explanation",
        "temperature": temperature,
    }]


# Parsing helpers

def parse_json_from_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response (handles markdown fences, partial JSON)."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)

    match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if not match:
        objects = re.findall(r"\{[^{}]*\"instruction\"[^{}]*\"output\"[^{}]*\}", text, flags=re.DOTALL)
        samples = []
        for obj in objects:
            try:
                samples.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
        if not samples:
            preview = text[:200].replace("\n", "\\n")
            dlog(f"parse_json: no array found, no objects matched. preview=...{preview}")
        return samples

    try:
        array = json.loads(match.group(1))
        if isinstance(array, list):
            valid = [item for item in array if "instruction" in item and "output" in item]
            dlog(f"parse_json: extracted {len(valid)}/{len(array)} valid items from array")
            return valid
    except json.JSONDecodeError as e:
        preview = match.group(1)[:200].replace("\n", "\\n")
        dlog(f"parse_json: JSON decode failed ({e}). bracket_match_preview=...{preview}")

    preview = text[:300].replace("\n", "\\n")
    dlog(f"parse_json: FAILED entirely. full_text_preview=...{preview}")
    return []


# Task factory (creates one generation job)

def make_task(seeds: list[dict], temps: list[float]) -> tuple[str, list[dict]]:
    """Pick a strategy and temperature, run it. Returns (strategy_label, samples)."""
    global task_id_counter
    with task_counter_lock:
        task_id_counter += 1
        tid = task_id_counter
    with task_timing_lock:
        task_timings[tid] = {"start": time.time(), "label": "pending"}

    roll = random.random()
    if roll < 0.40:
        label = "nl_to_code"
    elif roll < 0.65:
        label = "self_instruct"
    elif roll < 0.85:
        label = "evol_instruct"
    else:
        label = "code_to_explanation"

    temp = random.choice(temps)
    p(f"[{time.strftime('%H:%M:%S')}] #{tid:>3} START [{label:<20s}] t={temp}")
    start = time.time()

    if label == "nl_to_code":
        samples = generate_nl_to_code(seeds, temp)
    elif label == "self_instruct":
        samples = generate_self_instruct(seeds, temp, batch_size=5)
    elif label == "evol_instruct":
        samples = generate_evol_instruct(seeds, temp)
    else:
        samples = generate_code_explanation(seeds, temp)

    elapsed = time.time() - start
    count = len(samples) if samples else 0
    rate = count / elapsed if elapsed > 0 else 0
    with task_timing_lock:
        task_timings[tid] = {
            "start": start,
            "label": label,
            "elapsed": elapsed,
            "count": count,
            "rate": rate,  # samples/sec for this task
        }
    status = f"+{count}" if count > 0 else "+0 (FAILED)"
    p(f"[{time.strftime('%H:%M:%S')}] #{tid:>3} DONE  [{label:<20s}] t={temp} "
      f"| {elapsed:6.1f}s | {status}")

    # Log every task result for post-mortem (especially +0 failures)
    if count == 0:
        dlog(f"TASK #{tid} FAILED [{label}] t={temp}: returned 0 samples after {elapsed:.1f}s")
    else:
        dlog(f"TASK #{tid} OK [{label}] t={temp}: +{count} samples in {elapsed:.1f}s")

    return (label, samples)


# File I/O helpers

def save_samples(samples: list[dict], path: Path):
    """Append samples to output file (checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def write_final(samples: list[dict], path: Path):
    """Write complete output (overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


# Main generation loop

CONFIG = {"server": os.environ.get("LLAMA_HOST", "http://127.0.0.1:18080")}


def main():
    global total_completed, pending_count, shutdown_requested, executor_shutting_down

    parser = argparse.ArgumentParser(description="Generate synthetic Verse data")
    parser.add_argument("--target-samples", type=int, default=5000, help="Target number of samples")
    parser.add_argument("--temps", default="0.3,0.5,0.7,0.9", help="Comma-separated temperatures")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (server --parallel)")
    parser.add_argument("--resume", action="store_true", help="Append to existing output instead of overwriting")
    parser.add_argument("--llama-server", default=None, help="Path to llama-server binary")
    parser.add_argument("--model-q4", default=None, help="Path to Qwen3.6-27B-Q4_K_M.gguf")
    args = parser.parse_args()

    target = args.target_samples
    temps = [float(t.strip()) for t in args.temps.split(",")]

    if args.llama_server:
        global LLAMA_SERVER
        LLAMA_SERVER = args.llama_server
    if args.model_q4:
        global MODEL_Q4
        MODEL_Q4 = args.model_q4

    p(f"Temperatures: {temps}")
    p(f"Workers:      {args.workers} (server --parallel)")
    p(f"Target:       {target} samples")
    p()

    # Start llama-server
    host, port = "127.0.0.1", 18080
    start_llama_server(host, port, args.workers)

    try:
        _run_generation(target, temps, args.workers, args.resume)
    finally:
        stop_llama_server()


def _run_generation(target: int, temps: list[float], workers: int, resume: bool):
    """Core generation loop (server is already running)."""
    global total_completed, pending_count

    # Load seeds
    seeds = load_seeds(SEEDS_FILE)

    # Load existing samples if resuming
    all_samples = []
    if resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            all_samples = [json.loads(line) for line in f]
        p(f"Resumed with {len(all_samples)} existing samples")

    # Start time
    start_time = time.time()
    strategy_counts = {"self_instruct": 0, "evol_instruct": 0, "nl_to_code": 0, "code_to_explanation": 0}
    in_flight_task_ids = set()

    p(f"\nStarting generation... (Ctrl+C to stop and save)\n")
    dlog(f"=== RUN START === target={target} workers={workers} temps={temps} resume={resume}")
    dlog(f"Seeds loaded: {len(seeds)}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        while len(all_samples) < target or futures:
            if shutdown_requested:
                break

            # Submit new tasks up to worker count and target
            while (len(all_samples) + pending_count < target
                   and len(futures) < workers):
                future = executor.submit(make_task, seeds, temps)
                futures[future] = task_id_counter  # track which task this is
                pending_count += 1

            # Wait for at least one task to complete
            if not futures:
                break

            done_iterable = as_completed(futures, timeout=300)
            try:
                finished_future = next(done_iterable)
            except StopIteration:
                break

            try:
                label, samples = finished_future.result()
            except Exception as e:
                p(f"  Task failed: {e}", file=sys.stderr)
                samples = []
                label = "error"

            finished_future_tid = futures.pop(finished_future, None)
            if finished_future_tid is not None:
                in_flight_task_ids.discard(finished_future_tid)
            pending_count -= 1

            count = len(samples) if samples else 0
            task_rate = (task_timings.get(finished_future_tid, {}).get("rate", 0.0)
                         if finished_future_tid is not None else 0.0)
            with task_timing_lock:
                task_rate_history.append(task_rate)
                # Keep last ~200 entries to bound memory for long runs
                while len(task_rate_history) > 200:
                    task_rate_history.pop(0)

            if samples:
                all_samples.extend(samples)
                strategy_counts[label] += count
                total_completed += count

                # Save checkpoint immediately after every batch
                save_samples(samples, OUTPUT_FILE)

            # ETA: account for parallel workers + in-flight work
            elapsed = time.time() - start_time
            num_in_flight = len(in_flight_task_ids)

            # Compute average samples-per-batch from recently completed tasks
            recent_rates = task_rate_history[-20:]  # last ~20 task completions
            if recent_rates:
                avg_samples_per_batch = sum(recent_rates) / len(recent_rates)
            else:
                avg_samples_per_batch = 1.0  # fallback: assume 1 sample per batch

            # Effective throughput: completed rate + in-flight contribution
            # (each in-flight task is expected to produce ~avg_samples_per_batch more)
            completed_rate = total_completed / elapsed if elapsed > 0 else 0
            inflight_contribution = num_in_flight * avg_samples_per_batch / max(elapsed, 1.0)
            effective_rate = completed_rate + inflight_contribution

            remaining = max(target - len(all_samples), 0)
            eta_secs = remaining / effective_rate if effective_rate > 0 else float('inf')

            if eta_secs < 60:
                eta_str = f"{eta_secs:.0f}s"
            elif eta_secs < 3600:
                eta_str = f"{eta_secs/60:.1f}m"
            elif eta_secs == float('inf'):
                eta_str = "N/A (no samples yet)"
            else:
                eta_str = f"{eta_secs/3600:.1f}h"

            temp_val = samples[0].get("temperature", "?") if samples else "?"
            p(f"[{time.strftime('%H:%M:%S')}] +{count:>2} [{label:<20s}] t={temp_val} | "
              f"{len(all_samples)}/{target} | {effective_rate:.2f}/s ({num_in_flight} in-flight) | ETA {eta_str}")

    # Shutdown handling
    elapsed = time.time() - start_time
    p(f"\n{'='*60}")

    if shutdown_requested:
        executor.shutdown(wait=True)
        dlog(f"=== RUN STOPPED (Ctrl+C) === samples={len(all_samples)} tasks_done={task_id_counter} elapsed={elapsed:.0f}s")
        p("Stopped by user (Ctrl+C).")
        p(f"Saved {len(all_samples)} samples to {OUTPUT_FILE}")
        p("Re-run with --resume to continue.")
    else:
        # Final overwrite with complete set (removes any checkpoint duplicates)
        write_final(all_samples, OUTPUT_FILE)
        dlog(f"=== RUN COMPLETE === samples={len(all_samples)} tasks_done={task_id_counter} elapsed={elapsed:.0f}s")

        p("Generation complete!")
        p(f"Total samples: {len(all_samples)}")
        if len(all_samples) > 0:
            p(f"Time: {elapsed/60:.1f} minutes ({elapsed/len(all_samples):.1f}s/sample)")
        p(f"\nBy strategy:")
        for strat, count in sorted(strategy_counts.items()):
            pct = count / len(all_samples) * 100 if len(all_samples) > 0 else 0
            p(f"  {strat}: {count} ({pct:.0f}%)")
        p(f"\nWritten to: {OUTPUT_FILE}")

    # Always show debug log location for post-mortem
    if DEBUG_LOG.exists():
        size = DEBUG_LOG.stat().st_size
        p(f"\nDebug log:  {DEBUG_LOG} ({size:,} bytes)")


if __name__ == "__main__":
    main()
