# 历史脚本归档（2026-10-29）

这里保存已结束实验的配方、过期授权启动器、一次性迁移工具及被正式模块替代的兼容文件，供追溯使用。
归档按原字节保存；其中的路径、GPU、容器名和时段代表当时状态，不是可直接运行的当前入口。
旧 run_alignment_budget.py 仅由历史故障回归测试显式加载函数定义，测试不调用其 main()。
当前可执行入口见项目 scripts/README.md；不要从归档恢复过期的资源授权。

## 路径迁移表

| 原路径 | 新路径 | 原因 |
|---|---|---|
| `scripts/call_model.py` | `scripts/deployment/call_model.py` | retained general CLI, grouped by responsibility |
| `scripts/check_conversion.py` | `tests/check_conversion_cpu.py` | small CPU numerical regression, belongs in tests |
| `scripts/check_merged_weights.py` | `scripts/evaluation/check_merged_weights.py` | retained general CLI, grouped by responsibility |
| `scripts/diagnose_forward.py` | `scripts/evaluation/diagnose_forward.py` | retained general CLI, grouped by responsibility |
| `scripts/evaluate.py` | `scripts/evaluation/evaluate.py` | retained general CLI, grouped by responsibility |
| `scripts/generate_samples.py` | `scripts/evaluation/generate_samples.py` | retained general CLI, grouped by responsibility |
| `scripts/gpu_lease.py` | `docs/archive/scripts-before-20261029/gpu_lease.py` | superseded alias; canonical module is in models/ |
| `scripts/infer.py` | `scripts/deployment/infer.py` | retained general CLI, grouped by responsibility |
| `scripts/launch_service.sh` | `docs/archive/scripts-before-20261029/launch_service.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/launch_simplewiki_cpt.sh` | `docs/archive/scripts-before-20261029/launch_simplewiki_cpt.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/manage_layout.py` | `docs/archive/scripts-before-20261029/manage_layout.py` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/merge_adapters.py` | `scripts/conversion/merge_adapters.py` | retained general CLI, grouped by responsibility |
| `scripts/monitor_long.py` | `docs/archive/scripts-before-20261029/monitor_long.py` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/prepare_domain_data.py` | `scripts/data/prepare_domain_data.py` | retained general CLI, grouped by responsibility |
| `scripts/prepare_simplewiki.py` | `scripts/data/prepare_simplewiki.py` | retained general CLI, grouped by responsibility |
| `scripts/prepare_wikitext.py` | `scripts/data/prepare_wikitext.py` | retained general CLI, grouped by responsibility |
| `scripts/resume_normalized_simplewiki_cpt.sh` | `docs/archive/scripts-before-20261029/resume_normalized_simplewiki_cpt.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/resume_simplewiki_cpt.sh` | `docs/archive/scripts-before-20261029/resume_simplewiki_cpt.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_alignment_budget.py` | `docs/archive/scripts-before-20261029/run_alignment_budget.py` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_conversion.sh` | `docs/archive/scripts-before-20261029/run_conversion.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_domain_pipeline.sh` | `docs/archive/scripts-before-20261029/run_domain_pipeline.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_eval.sh` | `docs/archive/scripts-before-20261029/run_eval.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_long.sh` | `docs/archive/scripts-before-20261029/run_long.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_merge.sh` | `docs/archive/scripts-before-20261029/run_merge.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_normalized_comparison.sh` | `docs/archive/scripts-before-20261029/run_normalized_comparison.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_public_validation.sh` | `docs/archive/scripts-before-20261029/run_public_validation.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/run_smoke.sh` | `docs/archive/scripts-before-20261029/run_smoke.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/serve_models.py` | `scripts/deployment/serve_models.py` | retained general CLI, grouped by responsibility |
| `scripts/serving_common.py` | `docs/archive/scripts-before-20261029/serving_common.py` | superseded alias; canonical module is in models/ |
| `scripts/summarize_results.py` | `docs/archive/scripts-before-20261029/summarize_results.py` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/test_stable_sparse.py` | `tests/check_sparse_forward_cpu.py` | small CPU numerical regression, belongs in tests |
| `scripts/tokenize_simplewiki.py` | `scripts/data/tokenize_simplewiki.py` | retained general CLI, grouped by responsibility |
| `scripts/train_adapters.py` | `scripts/training/smoke_cpt.py` | retained general CLI, grouped by responsibility |
| `scripts/train_dpo.py` | `scripts/training/dpo.py` | retained general CLI, grouped by responsibility |
| `scripts/train_long.py` | `scripts/training/cpt.py` | retained general CLI, grouped by responsibility |
| `scripts/train_sft.py` | `scripts/training/sft.py` | retained general CLI, grouped by responsibility |
| `scripts/upcycle.py` | `scripts/conversion/upcycle.py` | retained general CLI, grouped by responsibility |
| `scripts/verify_service.py` | `scripts/deployment/verify_service.py` | retained general CLI, grouped by responsibility |
| `scripts/wait_reservation_end.py` | `docs/archive/scripts-before-20261029/wait_reservation_end.py` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/docker/README.md` | `docs/archive/scripts-before-20261029/docker/README.md` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/docker/run_docker_sparse_upcycling.sh` | `docs/archive/scripts-before-20261029/docker/run_docker_sparse_upcycling.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/docker/run_docker_sparse_upcycling_2gpu.sh` | `docs/archive/scripts-before-20261029/docker/run_docker_sparse_upcycling_2gpu.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
| `scripts/docker/run_docker_sparse_upcycling_normalized.sh` | `docs/archive/scripts-before-20261029/docker/run_docker_sparse_upcycling_normalized.sh` | historical experiment recipe, dated launcher/guard, or one-time maintenance |
