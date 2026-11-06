#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
python3 -u scripts/merge_adapters.py \
  --base checkpoints/qwen3-8b-8x2-half \
  --adapter runs/wikitext-20steps/adapter.pt \
  --output checkpoints/qwen3-8b-8x2-half-cpt20 \
  2>&1 | tee logs/merge-cpt20.log
python3 -u scripts/evaluate.py \
  --model checkpoints/qwen3-8b-8x2-half-cpt20 \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl \
  --output runs/eval/moe-cpt20-merged.json --blocks 8 --block 64 \
  2>&1 | tee logs/eval-merged.log
