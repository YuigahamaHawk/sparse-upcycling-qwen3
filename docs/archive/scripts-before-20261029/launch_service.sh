#!/usr/bin/env bash
# Run INSIDE the user-started container, during an explicitly approved lease.
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
mkdir -p logs runs/serving-20261028
set -o noclobber
python3 -u scripts/serve_models.py \
  --lease configs/serving/lease-20261028.json \
  --registry configs/serving/models.json --model moe-sft \
  --host 127.0.0.1 --port 18082 \
  --api-key-file runs/serving-20261028/api.key \
  "$@"
