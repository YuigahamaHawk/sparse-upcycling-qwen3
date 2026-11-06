#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
mkdir -p logs
python3 -u scripts/train_adapters.py \
  --model checkpoints/qwen3-8b-8x2-half \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/train.jsonl \
  --output runs/wikitext-smoke --steps 3 --block 64 --rank 4 \
  --max-memory 28GiB 2>&1 | tee logs/smoke.log
