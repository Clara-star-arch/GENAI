#!/usr/bin/env bash
# Runs the data pipeline end to end. Works from any folder.
#   bash run.sh            clean -> verify -> chunk -> index -> smoke search  (uses existing crawl)
#   bash run.sh --crawl    re-crawl the site first
#   bash run.sh --eval     also run the answer evaluation (~4 min, a few cents) and make the charts
set -euo pipefail
cd "$(dirname "$0")"                       # always run from the project folder

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

if [[ " $* " == *" --crawl "* ]] || [ ! -f data/raw/manifest.jsonl ]; then
  echo "== crawl";  python -m scraper.crawl
fi
echo "== clean";    python -m pipeline.clean
echo "== verify";   python -m pipeline.verify_facts
echo "== chunk";    python -m pipeline.chunk
echo "== index";    python -m pipeline.index
echo "== search";   python -m pipeline.smoke_search -q
echo "== retrieval eval"; python -m eval.retrieval_eval
if [[ " $* " == *" --eval "* ]]; then
  echo "== answer eval"; python -m eval.answer_eval
  echo "== charts";      python -m eval.make_charts
fi
