# 命令入口

只保留当前可复用入口。模型、forward 和训练器实现请到 ../models/ 查看。
同名 CLI 与实现类是调用关系：例如 training/cpt.py 解析参数后调用 models.trainers.cpt.CPTTrainer.run()，不再写一套训练循环。
合并审计的实现也在 models/evaluation/merged_weights.py；本目录只保留参数入口。
所有命令从项目根目录执行；需要 GPU 的命令须取得本次批准，并在用户启动的容器及登记时段内运行。
目录整理和 --help 检查不启动容器、不使用 GPU。

| 目录 | 入口 | 用途 |
|---|---|---|
| data/ | prepare_wikitext.py | WikiText-2 小规模验证数据 |
| data/ | prepare_simplewiki.py、tokenize_simplewiki.py | SimpleWiki 清理与 token 化 |
| data/ | prepare_domain_data.py | SFT/DPO 公共数据准备 |
| conversion/ | upcycle.py | dense FFN 复制为 MoE 专家 |
| conversion/ | merge_adapters.py | 合并各阶段 LoRA 与路由器 |
| training/ | smoke_cpt.py、cpt.py | 小规模 CPT、token 流 CPT/断点恢复 |
| training/ | sft.py、dpo.py | 监督微调、偏好对齐 |
| evaluation/ | evaluate.py、generate_samples.py | loss/PPL、生成样例 |
| evaluation/ | diagnose_forward.py、check_merged_weights.py | forward 诊断、合并张量审计 |
| deployment/ | serve_models.py、launch_service.sh | 带有效 lease 的 HTTP 服务 |
| deployment/ | infer.py | 带有效 lease 的独立推理 |
| deployment/ | call_model.py、verify_service.py | HTTP 调用、服务检查 |

## 查看参数（不加载模型）

```bash
python3 scripts/conversion/upcycle.py --help
python3 scripts/training/cpt.py --help
python3 scripts/training/sft.py --help
python3 scripts/training/dpo.py --help
python3 scripts/deployment/serve_models.py --help
```

部分历史数据准备脚本为直接执行型，没有 --help；执行会准备数据，勿作为参数检查命令调用。
HTTP 客户端本身不加载模型，但向在线模型发起生成、加载或验证请求会让服务使用 GPU，仍须有当前授权。

## 服务入口

以下只展示写法；替换为已确认的本次授权、密钥和状态路径后，在用户已启动的容器内 tmux 中执行：

```bash
bash scripts/deployment/launch_service.sh \
  --lease configs/serving/lease-CURRENT.json \
  --api-key-file runs/serving-CURRENT/api.key \
  --status runs/serving-CURRENT/status.json \
  --model moe-sft --host 127.0.0.1 --port 18082
```

CURRENT 是占位符，不能复用过期授权文件。启动器不设置 GPU 编号，不启动容器；具体映射及截止时间由本次 lease 校验。

## 历史文件

- 已结束实验的 Shell 配方、续跑脚本、过期时段守护、一次性迁移工具：../docs/archive/scripts-before-20261029/。
- gpu_lease.py 和 serving_common.py 的兼容别名已移出 scripts/；直接导入 models.runtime.gpu_lease 和 models.inference.common。
- 数值检查移至 ../tests/check_conversion_cpu.py、../tests/check_sparse_forward_cpu.py。
- 逐文件映射、原文件哈希及备份见 ../docs/maintenance/scripts-cleanup-20261029/。
- 旧的平铺 scripts/<名称>.py 路径不再提供包装层；请按本表更新外部命令。历史报告保留当时命令，不能直接照抄执行。

以后新增入口先检查已有命令能否通过参数覆盖需求；不为每次实验新增 run_*.sh 或 resume_*.sh。
