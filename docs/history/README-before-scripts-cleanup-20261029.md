> 历史快照：以下脚本路径和运行时段已过期。当前入口见项目 README 和 scripts/README.md；不要直接执行历史命令。

> 资源规则：容器由用户启动；GPU 实验先申请并获得明确批准。运行状态以本次有效授权和实际进程为准，不沿用历史授权。
> 脚本入口已迁入 `scripts/`，完整模型位于 `checkpoints/`，`models/` 用于模型定义代码；详见 [目录说明](docs/PROJECT_LAYOUT.md)。

# Sparse upcycling on Qwen3-8B-Base

## 代码结构

模型形状和架构适配在 `models/backbones/`，MoE/LoRA 的 forward 在 `models/components/`；
`BasicTrainer` 和 CPT/SFT/DPO 训练器在 `models/trainers/`。转换、合并、评估、推理/API 和运行保护也各自归入 `models/` 子模块。
`scripts/` 保留 CLI 与流程入口，原命令和 checkpoint/adapter 格式继续兼容。
文件导航见 [模型实现说明](models/README.md) 和 [项目目录](docs/PROJECT_LAYOUT.md)。

This project adapts [Komatsuzaki et al., Sparse Upcycling (2023)](https://arxiv.org/abs/2212.05055) to the local Qwen3-8B-Base checkpoint. The paper itself used T5 and ViT, so this is an architecture adaptation, not a numerical reproduction of its SuperGLUE or ImageNet results.

## Configurations

`upcycle.py` defaults to 8 identical experts, top-2 routing and alternating sparse layers (18 of Qwen3's 36 layers). Attention, norms, embeddings and remaining MLPs are bitwise copies. The router is initialized with `N(0, 0.02²)`. `--normalize-topk` enables normalized routing for an exact initial dense function; the original language experiment did not use this normalization, so leave it off for the canonical configuration. A full-layer version is available with `--layers all`. The report's widened FFN and random expert are later method variants and are not implemented here.

The output is a standard Hugging Face `Qwen3MoeForCausalLM` sharded checkpoint. The converter streams by layer, avoiding a second model-sized CPU allocation.

`train_adapters.py` is a practical smoke/limited-budget continuation: it freezes base weights and trains only expert LoRA matrices and routers, with auxiliary router loss 0.01. This does **not** constitute the paper's full-parameter continual pretraining. Its JSONL input must contain a `text` field. The WikiText-2 preparation script provides pipeline data only, not the original dense model's pretraining mixture.

The public WikiText-2 raw files are downloaded from the [cosmo.zip mirror](https://cosmo.zip/pub/datasets/wikitext-2-raw/) inside Docker and kept under `/home/wurendi/datasets/sparse_upcycling_wikitext2`. The download is for a small functional validation only.

## In the Docker container

```bash
cd /home/wurendi/projects/sparse_upcycling_qwen3
python3 check_conversion.py
python3 scripts/upcycle.py --source /home/wurendi/models/Qwen-3-8B \
  --output /home/wurendi/projects/sparse_upcycling_qwen3/checkpoints/qwen3-8b-8x2-half \
  --experts 8 --top-k 2 --layers alternating
python3 scripts/prepare_wikitext.py
python3 scripts/train_adapters.py \
  --model checkpoints/qwen3-8b-8x2-half \
  --data /home/wurendi/datasets/sparse_upcycling_wikitext2/train.jsonl \
  --output runs/wikitext-smoke --steps 20 --block 128
bash scripts/run_eval.sh
bash scripts/run_public_validation.sh
bash scripts/run_merge.sh
```

Inspect `runs/*/metrics.jsonl` and the tmux pane for progress. A research comparison still needs a specified, sufficiently large continuation corpus, full-weight or equivalent training budget, dense-continuation and from-scratch MoE controls at matched compute, and held-out benchmarks.

容器仅由用户启动；GPU 卡号和本次实验必须事先获批，详见 [AGENTS.md](AGENTS.md)。

The benchmark-ready, standalone 20-step CPT checkpoint is written to `checkpoints/qwen3-8b-8x2-half-cpt20` by `run_merge.sh`. It contains merged expert weights and the updated routers, plus a `cpt_manifest.json`. The unmerged adapter is retained in `runs/wikitext-20steps/adapter.pt` for further training. This 1,280-token run validates the pipeline only.

## SimpleWiki continuation

`launch_simplewiki_cpt.sh` runs inside the persistent Docker container in the host tmux session `sparse-upcycling:cpt`. It waits for `prepare_simplewiki.py` (in the `data` window), tokenizes the public [Simple English Wikipedia dump](https://dumps.wikimedia.org/simplewiki/latest/) into `/home/wurendi/datasets/simplewiki_cpt_20260901`, and launches `run_long.sh` from the saved `cpt20` model. The source archive SHA256 and train/validation split are recorded in `manifest.json`; token counts and hashes are recorded in `tokens_manifest.json`.

The prepared corpus contains 240,343 training articles (112,248,359 tokens) and 2,465 validation articles (1,097,935 tokens). The SHA256 of the compressed source is `6832fd106ae0e4734a349d6351709fc219ccebad5850234214a57e653f7812f8`. The pre-continuation `cpt20` checkpoint scores 15.2971 perplexity on the 8 × 64-token WikiText-2 validation setting used for the final comparison (`runs/simplewiki-cpt-1m/baseline_eval.json`).

The continuation is 1,000 optimizer steps × 4 sequences × 256 tokens = 1,024,000 sampled training tokens. It updates rank-4 expert LoRA matrices and routers while keeping copied base weights frozen. This is a budgeted CPT validation, not full-weight reproduction of the paper. The run writes `runs/simplewiki-cpt-1m/metrics.jsonl`, `status.json`, and `resume.pt` every 25 steps. A stopped run can resume with `train_long.py --resume` using the same arguments as `run_long.sh`.

After training, `run_long.sh` merges the adapters into the standalone Hugging Face checkpoint `checkpoints/qwen3-8b-8x2-half-cpt-simplewiki-1m` and evaluates it on the held-out WikiText-2 validation blocks. To inspect progress from the host, use `tmux attach -t sparse-upcycling` and select window `cpt`, or run inside Docker:

```bash
cd /home/wurendi/projects/sparse_upcycling_qwen3
python3 scripts/monitor_long.py
```

## 各阶段模型推理服务（2026-10-28）

部署、HTTP API、模型切换和独立推理详见 [部署与调用说明](docs/DEPLOYMENT.md)。
默认服务模型为 moe-sft；七个阶段/分支共享一个模型实例，DPO 使用 SFT 基础权重加原始 adapter 和路由器。
容器内 tmux 会话 llm-serving，监听 127.0.0.1:18082，接口需本次 API key。
2026-10-28 用户已续约至 22:00；仅使用已授权物理 GPU 2、3，21:55 停止自己的 GPU 子进程。续约验证后当前加载 moe-dpo；后续启动须有当次有效授权。
