#!/usr/bin/env python3
"""Consolidate all seed data into a unified seeds.jsonl for synthetic generation.

Normalizes schemas, deduplicates by code content, filters noise, and tags metadata.
"""

import hashlib
import html
import json
import re
import sys
from pathlib import Path

SEEDS_DIR = Path("data/seeds")
OUTPUT = SEEDS_DIR / "unified_seeds.jsonl"

# Minimum viable code snippet (filter out noise like "* module B")
MIN_CODE_CHARS = 30
MIN_CODE_LINES = 2


def code_hash(code: str) -> str:
    """Stable hash of normalized code for dedup."""
    normalized = re.sub(r"\s+", " ", code).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def clean_html_entities(text: str) -> str:
    """Decode HTML entities and strip residual CSS/JS noise from MkDocs context."""
    text = html.unescape(text)
    # Remove CSS variable declarations
    text = re.sub(r":root\{[^}]*\}", "", text)
    # Remove JS assignments that leaked into context
    text = re.sub(r"__md_\w+\s*=\s*[^\n;]+;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_verse_syntax(code: str) -> bool:
    """Heuristic: does this look like Verse code?"""
    verse_indicators = [
        r"using\s*\{",           # module imports
        r":=",                    # assignment
        r"\[.*\]\s*#",           # effect brackets with comment
        r"class\s+\w+",          # class definition
        r"func\s+\w+",           # function definition
        r"event\s+\w+",          # event declaration
        r"public\s+",            # access modifier
        r"protected\s+",         # access modifier
        r"private\s+",           # access modifier
        r"static\s+",            # static keyword
        r"override\s+",          # override keyword
        r"implements\s+",        # interface implementation
        r"extends\s+",           # class extension
        r"<\w+>\s*:",            # generic type parameter
        r"/Fortnite\.com/",      # Fortnite module path
        r"/Verse\.org/",         # Verse stdlib path
        r"/UnrealEngine\.com/",  # UE module path
    ]
    return any(re.search(pattern, code) for pattern in verse_indicators)


def normalize_book_of_verse(entry: dict) -> dict | None:
    """Normalize a Book of Verse entry."""
    code = entry.get("code", "").strip()
    if len(code) < MIN_CODE_CHARS or code.count("\n") < MIN_CODE_LINES:
        return None

    context = clean_html_entities(entry.get("context", ""))[:500]

    return {
        "id": f"book-{code_hash(code)}",
        "source": {
            "type": "book_of_verse",
            "page": entry.get("page", ""),
            "section": entry.get("section", ""),
        },
        "code": code,
        "context": context,
        "metadata": {
            "lines": code.count("\n") + 1,
            "chars": len(code),
            "has_verse_syntax": has_verse_syntax(code),
        },
    }


def normalize_md_scraped(entry: dict) -> dict | None:
    """Normalize a scraped markdown entry."""
    code = entry.get("code", "").strip()
    if len(code) < MIN_CODE_CHARS or code.count("\n") < MIN_CODE_LINES:
        return None

    context = clean_html_entities(entry.get("context", ""))[:500]

    return {
        "id": f"md-{code_hash(code)}",
        "source": {
            "type": "github_md",
            "repo": entry.get("source", "").replace("github:", ""),
            "file": entry.get("file", ""),
            "section": entry.get("section", ""),
        },
        "code": code,
        "context": context,
        "metadata": {
            "lines": code.count("\n") + 1,
            "chars": len(code),
            "has_verse_syntax": has_verse_syntax(code),
        },
    }


def normalize_verse_file(entry: dict) -> dict | None:
    """Normalize a .verse file snippet."""
    code = entry.get("code", "").strip()
    if len(code) < MIN_CODE_CHARS or code.count("\n") < 1:
        return None

    # Skip pure comment blocks (ASCII art, license headers)
    non_comment_lines = [l for l in code.split("\n") if not l.strip().startswith("#")]
    if not non_comment_lines:
        return None

    return {
        "id": f"verse-{code_hash(code)}",
        "source": {
            "type": "github_verse_file",
            "repo": entry.get("source", "").replace("github:", ""),
            "file": entry.get("file", ""),
        },
        "code": code,
        "context": "",  # Verse files are self-contained
        "metadata": {
            "lines": code.count("\n") + 1,
            "chars": len(code),
            "has_verse_syntax": has_verse_syntax(code),
            "total_file_lines": entry.get("total_lines", 0),
        },
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    normalizers = {
        "book_of_verse.jsonl": normalize_book_of_verse,
        "md_scraped.jsonl": normalize_md_scraped,
        "verse_files.jsonl": normalize_verse_file,
    }

    all_seeds: list[dict] = []
    seen_hashes: set[str] = set()
    stats = {
        "total_raw": 0,
        "filtered_short": 0,
        "filtered_comments_only": 0,
        "deduped": 0,
        "by_source": {},
        "with_verse_syntax": 0,
    }

    for filename, normalize in normalizers.items():
        path = SEEDS_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found)")
            continue

        source_count = 0
        with open(path) as f:
            for line in f:
                stats["total_raw"] += 1
                entry = json.loads(line)
                normalized = normalize(entry)

                if normalized is None:
                    # Determine why it was filtered
                    code = entry.get("code", "").strip()
                    if len(code) < MIN_CODE_CHARS or code.count("\n") < MIN_CODE_LINES:
                        stats["filtered_short"] += 1
                    else:
                        stats["filtered_comments_only"] += 1
                    continue

                # Dedup by code hash (across all sources)
                h = code_hash(normalized["code"])
                if h in seen_hashes:
                    stats["deduped"] += 1
                    continue
                seen_hashes.add(h)

                source_count += 1
                if normalized["metadata"]["has_verse_syntax"]:
                    stats["with_verse_syntax"] += 1
                all_seeds.append(normalized)

        stats["by_source"][filename] = source_count
        print(f"  {filename}: {source_count} seeds")

    # Sort: verse files first (highest quality), then book, then MD
    priority = {"github_verse_file": 0, "book_of_verse": 1, "github_md": 2}
    all_seeds.sort(key=lambda s: (priority.get(s["source"]["type"], 99), -s["metadata"]["chars"]))

    # Write unified output
    with open(OUTPUT, "w") as f:
        for seed in all_seeds:
            f.write(json.dumps(seed) + "\n")

    print(f"\n{'='*60}")
    print(f"Raw inputs:          {stats['total_raw']}")
    print(f"Filtered (too short):{stats['filtered_short']}")
    print(f"Filtered (comments): {stats['filtered_comments_only']}")
    print(f"Deduplicated:        {stats['deduped']}")
    print(f"Final unified seeds: {len(all_seeds)}")
    print(f"With Verse syntax:   {stats['with_verse_syntax']} ({stats['with_verse_syntax']/max(len(all_seeds),1)*100:.0f}%)")
    print(f"\nBy source:")
    for src, count in stats["by_source"].items():
        print(f"  {src}: {count}")
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
