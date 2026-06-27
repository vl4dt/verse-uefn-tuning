#!/bin/bash
cd /home/vl4dt/LLM-AI-Tooling/verse-uefn-tuning

echo "=== Running Book of Verse scraper ==="
python3 scripts/scrape_book_of_verse.py 2>&1

echo ""
echo "=== Cloning secondary Verse repos ==="
mkdir -p data/repos

# Known Verse codebases to scrape
declare -a REPOS=(
    "https://github.com/VerseLanguage/verse.git"
    "https://github.com/epicgames/UnrealEngine.git"  # Has Verse samples in Plugins/Runtime/Verse
)

for repo in "${REPOS[@]}"; do
    name=$(basename "$repo" .git)
    if [ -d "data/repos/$name" ]; then
        echo "  $name already cloned, skipping"
    else
        echo "  Cloning $repo (shallow)..."
        git clone --depth=1 "$repo" "data/repos/$name" 2>&1 | tail -3 || true
    fi
done

echo ""
echo "=== Finding .verse files ==="
find data/repos -name "*.verse" -type f 2>/dev/null | head -20 || echo "No .verse files found yet"
