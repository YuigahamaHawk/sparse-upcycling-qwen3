#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3
RUN=runs/simplewiki-normalized-cpt-1m
BASE=checkpoints/qwen3-8b-8x2-half-normalized-cpt20
FINAL=checkpoints/qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m
DATA=/home/wurendi/datasets/simplewiki_cpt_20260901/train.uint32.bin
VALID=/home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl
test -s "$RUN/resume.pt"
test -s "$DATA"
test -s "$BASE/model.safetensors.index.json"
mkdir -p logs
trap 'code=$?; printf "failed\n" > "$RUN/pipeline.status"; exit "$code"' ERR
printf 'training\n' > "$RUN/pipeline.status"
echo "Resuming normalized SimpleWiki CPT at $(date -Is)" | tee -a logs/normalized-simplewiki-cpt-1m.log
python3 -u scripts/train_long.py --model "$BASE" --tokens "$DATA" \
  --eval-data "$VALID" --output "$RUN" --steps 1000 --block 256 \
  --grad-accum 4 --rank 4 --lr 3e-5 --min-lr 3e-6 --warmup 20 \
  --save-every 25 --eval-every 100 --max-memory 28GiB --resume \
  2>&1 | tee -a logs/normalized-simplewiki-cpt-1m.log
printf 'merging\n' > "$RUN/pipeline.status"
python3 -u scripts/merge_adapters.py --base "$BASE" --adapter "$RUN/adapter.pt" \
  --output "$FINAL" 2>&1 | tee logs/normalized-merge-1m.log
printf 'evaluating\n' > "$RUN/pipeline.status"
python3 -u scripts/evaluate.py --model "$FINAL" --data "$VALID" \
  --output "$RUN/merged_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/normalized-merged-eval.log
printf 'complete\n' > "$RUN/pipeline.status"
chown -R 1021:1022 "$RUN" "$FINAL"
echo "Standalone normalized CPT checkpoint ready: $FINAL"
