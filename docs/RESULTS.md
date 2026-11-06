# Qwen3 sparse upcycling: public-data validation

Date: 2026-09-27. All model operations ran inside the `corex:4.3.6` Docker container on `icse`, with `/dev/iluvatar4`–`15` mapped. Host tmux session: `sparse-upcycling`.

## Configuration

- Source: `/home/wurendi/models/Qwen-3-8B` (Qwen3-8B-Base, 36 layers, hidden 4096, FFN 12288).
- Conversion: odd-numbered layers 1, 3, ..., 35 become MoE; eight exact FFN copies per converted layer; top-2 token routing; router `N(0, 0.02²)`; unnormalized top-k probabilities, matching the original paper's language setup. Other weights copied.
- Sharded initial MoE checkpoint: `checkpoints/qwen3-8b-8x2-half` (54.43 GB in tensor payload, about 27.2B total and 10.9B active parameters).
- CPT smoke: 20 steps, one 64-token block per step, rank-4 expert LoRA plus trainable routers, auxiliary load balancing coefficient 0.01. This is parameter-efficient validation, **not** the paper's full-weight pretraining.
- Training data: WikiText-2 raw (English), downloaded inside Docker from `https://cosmo.zip/pub/datasets/wikitext-2-raw/` to `/home/wurendi/datasets/sparse_upcycling_wikitext2`.
- Raw SHA-256: `wiki.train.raw` = `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`; `wiki.valid.raw` = `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`.

## Held-out result

The same first eight 64-token blocks of WikiText-2 validation were evaluated for every model. The sample is only 512 tokens, so these numbers confirm pipeline behavior rather than model quality.

| Model | Mean NLL | Perplexity |
|---|---:|---:|
| Original dense Qwen3-8B | 2.3651 | 10.64 |
| Initial upcycled MoE | 2.7889 | 16.26 |
| MoE after 3 CPT steps, adapter loaded | 2.7836 | 16.18 |
| MoE after 20 CPT steps, adapter loaded | 2.7239 | 15.24 |
| MoE after 20 CPT steps, standalone merged checkpoint | 2.7277 | 15.30 |

The small difference between the loaded adapter and merged checkpoint is from merging FP32 LoRA updates into BF16 expert weights. The merged checkpoint was loaded independently and evaluated successfully.

## Saved artifacts

- **Benchmark-ready complete CPT checkpoint:** `checkpoints/qwen3-8b-8x2-half-cpt20/` (config, tokenizer, 37 safetensors shards, index, manifest).
- Continue-training adapter: `runs/wikitext-20steps/adapter.pt`.
- Per-step metrics: `runs/wikitext-20steps/metrics.jsonl`.
- Evaluation JSON: `runs/eval/`.
- Command logs: `logs/`.

The original paper's SuperGLUE/ImageNet figures used T5/ViT, so they cannot be numerically reproduced from Qwen3-8B. A strong Qwen3 reproduction still needs a large continuation corpus, a longer full-weight training run, and dense-continuation and from-scratch controls matched by compute, followed by a held-out benchmark suite. The present run establishes conversion, training, saving and independent checkpoint loading on this server.

## SimpleWiki CPT continuation (completed 2026-09-30)

- Corpus: 240,343 Simple English Wikipedia training articles, 112,248,359 tokens; archive and token hashes in `/home/wurendi/datasets/simplewiki_cpt_20260901/`.
- Objective: 1,000 steps × 1,024 sampled tokens/step, expert rank-4 LoRA and routers, starting from the merged `cpt20` checkpoint. Base weights remain frozen.
- The first run stopped after step 240 due to an invalid expert index emitted by the Corex/Transformers active-expert `nonzero()` path. The last complete checkpoint was step 225. `train_adapters.py` now iterates over the eight experts with Python integer indices; a CPU equivalence test gave exactly matching output and input/router gradients.
- The resumed run in `wrd-sparse-upcycling-2gpu` (physical GPUs 4 and 5) completed all 1,000 steps in about 1.95 h. `runs/simplewiki-cpt-1m/complete.json` records 1,024,000 trained tokens; final in-training eval perplexity 12.233.
- Merged standalone checkpoint: `checkpoints/qwen3-8b-8x2-half-cpt-simplewiki-1m/`, scoring 12.162 perplexity on the fixed 512-token WikiText-2 validation sample (`runs/simplewiki-cpt-1m/merged_eval.json`).

## Normalized-routing comparison (completed 2026-09-30)

- Same 8-expert, top-2, alternating-layer conversion from the original dense model, with `norm_topk_prob=true`, trained in `wrd-sparse-upcycling-normalized` on physical GPUs 6 and 7 with identical data and hyperparameters; the unnormalized run proceeded separately on GPUs 4 and 5.
- Initial normalized MoE: 10.6992 perplexity on the fixed 512-token validation sample, close to the original dense 10.64 and markedly below the unnormalized initial MoE 16.26.
- The 20-step WikiText CPT merged checkpoint scores 10.4323. The 1,000-step SimpleWiki continuation completed in about 3.46 h (slower than the unnormalized run, which shared the host concurrently). Merged checkpoint `checkpoints/qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m/` scores 10.5073 (`runs/simplewiki-normalized-cpt-1m/merged_eval.json`).

## Final comparison (2026-09-30)

Fixed 512-token WikiText-2 validation sample; both variants consumed the same 1,025,280 training tokens (20-step WikiText CPT + 1,000-step SimpleWiki CPT).

| Model | Perplexity |
|---|---:|
| Original dense Qwen3-8B | 10.64 |
| Initial MoE, unnormalized top-k | 16.26 |
| Initial MoE, normalized top-k | 10.70 |
| Unnormalized, after cpt20 + SimpleWiki 1M | 12.16 |
| Normalized, after cpt20 + SimpleWiki 1M | 10.51 |

- Normalized routing recovers the dense function at conversion time and stays within about 0.15 perplexity of dense through continued pretraining; its validation curve bottoms at 10.29 near step 400, then drifts up to about 10.59 as weights move toward the SimpleWiki distribution.
- Unnormalized routing starts 5.6 perplexity above dense and improves steadily (15.30 → 12.23 in-training), flattening near step 800, but does not close the gap after 1M tokens.
- `summarize_results.py` verified both runs (pipeline status, 1,000 steps) and wrote `runs/comparison_summary.json` and `runs/validation_curve.csv` with the full step-wise curves.
- The auxiliary load-balancing loss stays near 2.0 in both runs, suggesting uneven expert utilization; routing analysis is a candidate follow-up.
- The small-sample caveat still applies: 512 tokens confirm trends, not benchmark quality.
