#!/usr/bin/env python3
"""Scrape .md and .verse files from cloned repos for Verse code examples."""

import json
import re
import sys
from pathlib import Path

REPOS_DIR = Path("data/repos")
OUTPUT_MD = Path("data/seeds/md_scraped.jsonl")
OUTPUT_VERSE = Path("data/seeds/verse_files.jsonl")

def extract_code_blocks_from_md(text):
    """Extract fenced code blocks from markdown, keeping language hints."""
    blocks = []
    # Match ```lang ... ``` or just ``` ... ```
    pattern = re.compile(r'```(\w*)\s*\n(.*?)```', re.DOTALL)
    for match in pattern.finditer(text):
        lang = match.group(1).strip() or "verse"
        code = match.group(2).rstrip()
        if len(code) > 5:
            blocks.append({"lang": lang, "code": code})
    return blocks

def extract_context_around_block(md_text, block_start_pos, chars=300):
    """Get surrounding prose context for a code block."""
    start = max(0, block_start_pos - chars)
    end = min(len(md_text), block_start_pos + chars)
    # Strip other code blocks from context
    ctx = re.sub(r'```[\s\S]*?```', ' [code] ', md_text[start:end])
    ctx = re.sub(r'#.*$', '', ctx, flags=re.MULTILINE)  # Remove headers
    ctx = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', ctx)  # Resolve links to text
    ctx = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', ctx)  # Remove images
    ctx = re.sub(r'\s+', ' ', ctx).strip()
    return ctx

def process_md_file(filepath, repo_name):
    """Process a single .md file and extract code blocks with context."""
    try:
        text = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}", file=sys.stderr)
        return []

    seeds = []
    pattern = re.compile(r'```(\w*)\s*\n(.*?)```', re.DOTALL)
    for match in pattern.finditer(text):
        lang = match.group(1).strip() or "verse"
        code = match.group(2).rstrip()
        if len(code) < 5:
            continue

        # Get section title (nearest preceding header)
        before = text[:match.start()]
        headers = re.findall(r'^(#{1,6})\s+(.+)$', before, flags=re.MULTILINE)
        section = headers[-1][1] if headers else "unknown"

        context = extract_context_around_block(text, match.start())

        seeds.append({
            "source": f"github:{repo_name}",
            "file": str(filepath.relative_to(REPOS_DIR)),
            "section": section,
            "lang": lang,
            "context": context[:500],
            "code": code
        })
    return seeds

def process_verse_file(filepath, repo_name):
    """Process a .verse file - treat entire file as one sample."""
    try:
        text = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}", file=sys.stderr)
        return []

    if len(text.strip()) < 10:
        return []

    # Try to split into logical units (functions, classes, etc.)
    lines = text.split('\n')
    seeds = []

    # Simple heuristic: split on blank-line-separated blocks of >3 lines
    current_block = []
    for line in lines:
        if line.strip():
            current_block.append(line)
        else:
            if len(current_block) >= 3:
                block_text = '\n'.join(current_block).strip()
                seeds.append({
                    "source": f"github:{repo_name}",
                    "file": str(filepath.relative_to(REPOS_DIR)),
                    "code": block_text,
                    "total_lines": len(lines),
                    "block_lines": len(current_block)
                })
            current_block = []
    # Don't forget last block
    if len(current_block) >= 3:
        seeds.append({
            "source": f"github:{repo_name}",
            "file": str(filepath.relative_to(REPOS_DIR)),
            "code": '\n'.join(current_block).strip(),
            "total_lines": len(lines),
            "block_lines": len(current_block)
        })

    # If no blocks found, treat whole file as one sample
    if not seeds and text.strip():
        seeds.append({
            "source": f"github:{repo_name}",
            "file": str(filepath.relative_to(REPOS_DIR)),
            "code": text.strip(),
            "total_lines": len(lines),
            "block_lines": len(lines)
        })

    return seeds

def main():
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    all_md_seeds = []
    all_verse_seeds = []

    # Skip repos that are too large or irrelevant
    SKIP_REPOS = {"UnrealEngine"}  # Main engine source, thousands of non-Verse MD files

    for repo_dir in sorted(REPOS_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name
        if repo_name in SKIP_REPOS:
            print(f"\nSkipping: {repo_name} (too large / irrelevant)")
            continue
        print(f"\nProcessing: {repo_name}")

        # Find .md files (skip node_modules, .git, etc.)
        md_files = list(repo_dir.rglob("*.md"))
        md_files = [f for f in md_files if '/node_modules/' not in str(f) and '/.git/' not in str(f)]
        print(f"  Found {len(md_files)} .md files")

        for md_file in md_files:
            seeds = process_md_file(md_file, repo_name)
            all_md_seeds.extend(seeds)

        # Find .verse files
        verse_files = list(repo_dir.rglob("*.verse"))
        verse_files = [f for f in verse_files if '/node_modules/' not in str(f) and '/.git/' not in str(f)]
        print(f"  Found {len(verse_files)} .verse files")

        for vf in verse_files:
            seeds = process_verse_file(vf, repo_name)
            all_verse_seeds.extend(seeds)

    # Write outputs
    with open(OUTPUT_MD, 'w') as f:
        for seed in all_md_seeds:
            f.write(json.dumps(seed) + '\n')

    with open(OUTPUT_VERSE, 'w') as f:
        for seed in all_verse_seeds:
            f.write(json.dumps(seed) + '\n')

    print(f"\n{'='*60}")
    print(f"MD code blocks extracted: {len(all_md_seeds)}")
    print(f"Verse file snippets extracted: {len(all_verse_seeds)}")
    print(f"Total seed samples: {len(all_md_seeds) + len(all_verse_seeds)}")
    print(f"Written to: {OUTPUT_MD}")
    print(f"Written to: {OUTPUT_VERSE}")

if __name__ == "__main__":
    main()
