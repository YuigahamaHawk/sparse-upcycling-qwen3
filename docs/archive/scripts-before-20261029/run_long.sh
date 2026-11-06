#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
RUN=runs/simplewiki-cpt-1m
BASE=checkpoints/qwen3-8b-8x2-half-cpt20
FINAL=checkpoints/qwen3-8b-8x2-half-cpt-simplewiki-1m
mkdir -p "$RUN" logs
printf 'training\n' > "$RUN/pipeline.status"
python3 -u scripts/train_long.py \
  --model "$BASE" \
  --tokens /home/wurendi/datasets/simplewiki_cpt_20260901/train.uint32.bin \
  --eval-data /home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl \
  --output "$RUN" --steps 1000 --block 256 --grad-accum 4 \
  --rank 4 --lr 3e-5 --min-lr 3e-6 --warmup 20 \
  --save-every 25 --eval-every 100 --max-memory 28GiB \
  2>&1 | tee logs/simplewiki-cpt-1m.log
printf 'merging\n' > "$RUN/pipeline.status"
python3 -u scripts/merge_adapters.py --base "$BASE" --adapter "$RUN/adapter.pt" \
  --output "$FINAL" 2>&1 | tee logs/simplewiki-merge-1m.log
python3 -u scripts/evaluate.py --model "$FINAL" \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl \
  --output "$RUN/merged_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/simplewiki-merged-eval-1m.log
printf 'complete\n' > "$RUN/pipeline.status"
echo "Standalone CPT checkpoint ready: $FINAL"
