#!/usr/bin/env bash
# Domain-model pipeline on the upcycled MoE checkpoint:
# data download -> SFT -> merge -> DPO -> merge -> evaluation -> generation.
# Run only inside the user-started container after explicit approval of this experiment and GPU IDs.
set -euo pipefail
cd /home/wurendi/projects/sparse_upcycling_qwen3

BASE=checkpoints/qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m
DATA=/home/wurendi/datasets/domain_sft_dpo_20260930
WIKI=/home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl
SFT_RUN=runs/sft-alpaca
SFT_CKPT=checkpoints/qwen3-8b-8x2-norm-sft-alpaca-merged
DPO_RUN=runs/dpo-hh
DPO_CKPT=checkpoints/qwen3-8b-8x2-norm-dpo-hh-merged
STATUS=runs/domain_pipeline.status

status() { printf '%s\n' "$1" > "$STATUS"; mkdir -p "$SFT_RUN" "$DPO_RUN" logs; }

status data
python3 -u scripts/prepare_domain_data.py --output "$DATA" 2>&1 | tee logs/domain-prepare.log

status sft-training
python3 -u scripts/train_sft.py \
  --model "$BASE" \
  --data "$DATA/alpaca_train.jsonl" \
  --eval-data "$DATA/alpaca_val.jsonl" \
  --wiki-eval "$WIKI" \
  --output "$SFT_RUN" \
  --epochs 2 --max-len 640 --grad-accum 8 --rank 8 \
  --lr 1e-4 --min-lr 1e-5 --warmup 20 \
  --save-every 25 --eval-every 50 --max-memory 18GiB \
  2>&1 | tee logs/domain-sft.log

status sft-merging
python3 -u scripts/merge_adapters.py --base "$BASE" --adapter "$SFT_RUN/adapter.pt" \
  --output "$SFT_CKPT" 2>&1 | tee logs/domain-sft-merge.log
python3 -u scripts/evaluate.py --model "$SFT_CKPT" --data "$WIKI" \
  --output "$SFT_RUN/merged_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/domain-sft-eval.log

status dpo-training
python3 -u scripts/train_dpo.py \
  --model "$SFT_CKPT" \
  --data "$DATA/hh_pairs.jsonl" \
  --eval-data "$DATA/hh_val.jsonl" \
  --output "$DPO_RUN" \
  --epochs 1 --beta 0.1 --max-len 640 --grad-accum 8 --rank 8 \
  --lr 5e-5 --min-lr 5e-6 --warmup 10 \
  --save-every 20 --eval-every 20 --max-memory 18GiB \
  2>&1 | tee logs/domain-dpo.log

status dpo-merging
python3 -u scripts/merge_adapters.py --base "$SFT_CKPT" --adapter "$DPO_RUN/adapter.pt" \
  --output "$DPO_CKPT" 2>&1 | tee logs/domain-dpo-merge.log
python3 -u scripts/evaluate.py --model "$DPO_CKPT" --data "$WIKI" \
  --output "$DPO_RUN/merged_eval.json" --blocks 8 --block 64 \
  2>&1 | tee logs/domain-dpo-eval.log

status generating
python3 -u scripts/generate_samples.py --model "$SFT_CKPT" \
  --output "$SFT_RUN/generations.jsonl" --max-memory 18GiB 2>&1 | tee logs/domain-sft-generate.log
python3 -u scripts/generate_samples.py --model "$DPO_CKPT" \
  --output "$DPO_RUN/generations.jsonl" --max-memory 18GiB 2>&1 | tee logs/domain-dpo-generate.log

status complete
echo "Domain pipeline complete: $DPO_CKPT"
