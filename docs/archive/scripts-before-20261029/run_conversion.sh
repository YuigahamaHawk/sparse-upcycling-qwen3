#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
mkdir -p logs checkpoints
python3 -u scripts/upcycle.py \
  --source /home/wurendi/models/Qwen-3-8B \
  --output checkpoints/qwen3-8b-8x2-half \
  --experts 8 --top-k 2 --layers alternating \
  2>&1 | tee logs/convert.log
