#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
mkdir -p logs runs/eval
DATA=/home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl
python3 -u scripts/evaluate.py --model /home/wurendi/models/Qwen-3-8B --dense \
  --data "$DATA" --output runs/eval/dense.json --blocks 8 --block 64 2>&1 | tee logs/eval-dense.log
python3 -u scripts/evaluate.py --model checkpoints/qwen3-8b-8x2-half \
  --data "$DATA" --output runs/eval/moe-initial.json --blocks 8 --block 64 2>&1 | tee logs/eval-moe-initial.log
python3 -u scripts/evaluate.py --model checkpoints/qwen3-8b-8x2-half \
  --adapter runs/wikitext-smoke/adapter.pt \
  --data "$DATA" --output runs/eval/moe-smoke.json --blocks 8 --block 64 2>&1 | tee logs/eval-moe-smoke.log
