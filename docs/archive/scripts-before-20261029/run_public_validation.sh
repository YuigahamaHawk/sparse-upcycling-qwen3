#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
mkdir -p logs
python3 -u scripts/train_adapters.py \
  --model checkpoints/qwen3-8b-8x2-half \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/train.jsonl \
  --output runs/wikitext-20steps --steps 20 --block 64 --rank 4 \
  --max-memory 28GiB 2>&1 | tee logs/train-20.log
python3 -u scripts/evaluate.py \
  --model checkpoints/qwen3-8b-8x2-half \
  --adapter runs/wikitext-20steps/adapter.pt \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl \
  --output runs/eval/moe-20steps.json --blocks 8 --block 64 \
  2>&1 | tee logs/eval-moe-20steps.log
