#!/usr/bin/env bash
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3

INITIAL=checkpoints/qwen3-8b-8x2-half-normalized
WIKI=checkpoints/qwen3-8b-8x2-half-normalized-cpt20
FINAL=checkpoints/qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m
SMOKE_RUN=runs/wikitext-normalized-20steps
RUN=runs/simplewiki-normalized-cpt-1m
VALID=/home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl
mkdir -p "$RUN" logs
trap 'code=$?; printf "failed\n" > "$RUN/pipeline.status"; exit "$code"' ERR

printf 'converting\n' > "$RUN/pipeline.status"
python3 -u scripts/upcycle.py --source /home/wurendi/models/Qwen-3-8B \
  --output "$INITIAL" --experts 8 --top-k 2 --layers alternating \
  --normalize-topk 2>&1 | tee logs/normalized-convert.log

printf 'initial_evaluation\n' > "$RUN/pipeline.status"
python3 -u scripts/evaluate.py --model "$INITIAL" --data "$VALID" \
  --output "$RUN/initial_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/normalized-initial-eval.log

printf 'wikitext_cpt\n' > "$RUN/pipeline.status"
python3 -u scripts/train_adapters.py --model "$INITIAL" \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/train.jsonl \
  --output "$SMOKE_RUN" --steps 20 --block 64 --rank 4 --max-memory 28GiB \
  2>&1 | tee logs/normalized-wikitext-cpt20.log

printf 'wikitext_merging\n' > "$RUN/pipeline.status"
python3 -u scripts/merge_adapters.py --base "$INITIAL" \
  --adapter "$SMOKE_RUN/adapter.pt" --output "$WIKI" \
  2>&1 | tee logs/normalized-merge-cpt20.log
python3 -u scripts/evaluate.py --model "$WIKI" --data "$VALID" \
  --output "$RUN/cpt20_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/normalized-cpt20-eval.log

printf 'training\n' > "$RUN/pipeline.status"
python3 -u scripts/train_long.py --model "$WIKI" \
  --tokens /home/wurendi/datasets/simplewiki_cpt_20260901/train.uint32.bin \
  --eval-data "$VALID" --output "$RUN" --steps 1000 --block 256 \
  --grad-accum 4 --rank 4 --lr 3e-5 --min-lr 3e-6 --warmup 20 \
  --save-every 25 --eval-every 100 --max-memory 28GiB \
  2>&1 | tee logs/normalized-simplewiki-cpt-1m.log

printf 'merging\n' > "$RUN/pipeline.status"
python3 -u scripts/merge_adapters.py --base "$WIKI" \
  --adapter "$RUN/adapter.pt" --output "$FINAL" \
  2>&1 | tee logs/normalized-merge-1m.log
printf 'evaluating\n' > "$RUN/pipeline.status"
python3 -u scripts/evaluate.py --model "$FINAL" --data "$VALID" \
  --output "$RUN/merged_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/normalized-merged-eval.log

printf 'complete\n' > "$RUN/pipeline.status"
chown -R 1021:1022 "$RUN" "$INITIAL" "$WIKI" "$FINAL" "$SMOKE_RUN"
echo "Standalone normalized CPT checkpoint ready: $FINAL"
