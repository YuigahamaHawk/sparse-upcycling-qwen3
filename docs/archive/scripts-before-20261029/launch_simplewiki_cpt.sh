#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
DATA=/home/wurendi/datasets/simplewiki_cpt_20260901
mkdir -p logs

echo "Waiting for SimpleWiki preparation at $(date -Is)"
for ((attempt=0; attempt<240; attempt++)); do
  if [[ -s "$DATA/manifest.json" ]]; then
    break
  fi
  if ! ps -eo args | grep -Eq '[p]ython3 -u (scripts/)?prepare_simplewiki.py'; then
    echo "Dataset preparation ended without a manifest; see logs/prepare-simplewiki.log" >&2
    exit 1
  fi
  sleep 30
done
[[ -s "$DATA/manifest.json" ]] || { echo "Timed out waiting for dataset" >&2; exit 1; }

echo "Tokenizing SimpleWiki at $(date -Is)"
python3 -u scripts/tokenize_simplewiki.py 2>&1 | tee logs/tokenize-simplewiki.log
[[ -s "$DATA/train.uint32.bin" && -s "$DATA/tokens_manifest.json" ]]
echo "Starting CPT at $(date -Is)"
bash scripts/run_long.sh
