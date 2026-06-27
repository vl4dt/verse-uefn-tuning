#!/usr/bin/env python3
"""Scrape Book of Verse (MkDocs) for code examples via sitemap."""

import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request

BASE = "https://verselang.github.io/book/"
OUTPUT = Path("data/seeds/book_of_verse.jsonl")

def fetch(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")

def get_sections():
    """Fetch sitemap.xml to discover all pages."""
    try:
        xml = fetch(BASE + "sitemap.xml")
        locs = re.findall(r"<loc>([^<]+)</loc>", xml)
        # Filter to content pages (exclude assets, search, etc.)
        sections = [u.replace(BASE, "") for u in sorted(locs) if "/book/" in u and not any(x in u for x in ["assets/", "search", ".css", ".js"])]
        return sections
    except Exception as e:
        print(f"  sitemap.xml failed: {e}")
        return []

def extract_code_blocks(html):
    """Extract code blocks from MkDocs HTML. Handles highlight spans."""
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)

    blocks = []
    # MkDocs uses <div class="highlight"><pre><span class=...>code</span></pre></div>
    # Also catch bare <pre><code> and just <pre>
    pre_matches = re.findall(r"<pre(?:\s[^>]*)?>(.*?)</pre>", html, flags=re.DOTALL)
    for match in pre_matches:
        code = re.sub(r"<[^>]+>", "", match).strip()
        if len(code) > 10:
            blocks.append(code)
    return blocks

def extract_text_context(html):
    """Extract surrounding text context (excluding code blocks)."""
    html_no_code = re.sub(r"<pre[^>]*>.*?</pre>", " [CODE] ", html, flags=re.DOTALL)
    html_no_tags = re.sub(r"<[^>]+>", " ", html_no_code)
    text = re.sub(r"\s+", " ", html_no_tags).strip()
    return text[:600]

def extract_title(html):
    """Extract page title from <h1> or <title> tag."""
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.DOTALL)
    if h1_match:
        return re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
    title_match = re.search(r"<title>(.*?)</title>", html)
    if title_match:
        raw = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
        # Remove " — Book of Verse" suffix
        return raw.replace(" — Book of Verse", "").replace("| Book of Verse", "").strip()
    return ""

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    seeds = []

    print("Fetching sitemap...")
    sections = get_sections()
    if not sections:
        print("  No sections found!", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(sections)} pages")

    for section_path in sections:
        url = BASE + section_path
        try:
            html = fetch(url)
            code_blocks = extract_code_blocks(html)
            context = extract_text_context(html)
            title = extract_title(html) or section_path

            for i, code in enumerate(code_blocks):
                seeds.append({
                    "source": "book_of_verse",
                    "page": section_path,
                    "section": title,
                    "context": context,
                    "code": code,
                    "snippet_index": i
                })

            if code_blocks:
                print(f"  {section_path}: {len(code_blocks)} snippets")
        except Exception as e:
            print(f"  ERROR {section_path}: {e}", file=sys.stderr)

    # Write output
    with open(OUTPUT, "w") as f:
        for seed in seeds:
            f.write(json.dumps(seed) + "\n")

    print(f"\nDone! Wrote {len(seeds)} code snippets to {OUTPUT}")

if __name__ == "__main__":
    main()
