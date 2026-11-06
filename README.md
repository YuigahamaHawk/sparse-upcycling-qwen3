# Sparse upcycling on Qwen3-8B-Base

本项目把 Qwen3-8B-Base 的部分 dense FFN 扩展为 MoE，并完成预算受限的 CPT、SFT、DPO 与分阶段推理验证。
这是 Sparse Upcycling 的架构适配及小规模验证；CPT 冻结复制的基础权重，只训练专家 LoRA 和路由器，不等同论文的全参数继续预训练或原论文指标复现。

> GPU 任务必须先说明具体物理卡号、最低卡数和依据，获得本次批准；容器由用户启动。历史登记时段已过期，不能视为当前授权。

## GitHub 源码版

本仓库包含实现代码、测试、配置和实验说明，**不包含模型权重、adapter、断点、数据集或密钥**；models/ 是 Python 模型定义目录。
首次上传为 2026-11-06 的本地工作副本快照，服务器后续改动尚需恢复连接后核对。
克隆后的环境、权重路径和授权配置见 [源码仓库说明](docs/REPOSITORY.md)。

## 入口与实现

- [命令导航](scripts/README.md)：19 个通用入口，按 data、conversion、training、evaluation、deployment 五类管理。
- [模型实现](models/README.md)：模型形状、forward、LoRA、BasicTrainer 和 CPT/SFT/DPO，以及合并、评估和推理实现。
- [项目目录](docs/PROJECT_LAYOUT.md)、[资源规则](AGENTS.md)、[部署与 API](docs/DEPLOYMENT.md)。
- checkpoint 在 checkpoints/；adapter、断点及指标在 runs/；models/ 只放实现代码。

## 模型与流程

原模型是 decoder-only Transformer：36 层、hidden size 4096、32 个 attention query heads / 8 个 KV heads。
默认把交替的 18 层 FFN 替换为 8 专家、top-2 路由，attention、归一化、embedding 和其余 FFN 保持原结构。
--normalize-topk 让所选专家的路由权重和为 1；专家刚复制且相同时，这有助于保持 dense 的初始函数。

```text
dense checkpoint → upcycle → CPT adapter → merge → SFT adapter → merge → DPO adapter → evaluation / serving
```

DPO 的默认参考部署使用 SFT 合并权重 + 原始 DPO adapter/路由器；独立 BF16 DPO 合并结果存在已记录的数值一致性告警。
各阶段固定路径见 configs/serving/models.json；不要仅修改登记名就声称支持新架构。

## 常用命令

下列 --help 不加载模型：

```bash
cd /home/wurendi/projects/sparse_upcycling_qwen3
python3 scripts/conversion/upcycle.py --help
python3 scripts/training/cpt.py --help
python3 scripts/training/sft.py --help
python3 scripts/training/dpo.py --help
python3 scripts/conversion/merge_adapters.py --help
python3 scripts/evaluation/evaluate.py --help
python3 scripts/deployment/serve_models.py --help
```

实际实验按本次授权在用户启动的容器及 tmux 内运行。数据保存在 /home/wurendi/datasets/；原始 dense 模型位于 /home/wurendi/models/Qwen-3-8B。
SimpleWiki 两条 CPT 分支各自的 1,000 步预算为 1,000 × 4 × 256 = 1,024,000 token；详细数字以对应 runs/ 配置、指标及阶段报告为准。

## 维护与历史

2026-10-29：scripts/ 的 39 个平铺脚本已整理；18 个通用 Python 入口分组，新增一个无过期默认配置的通用服务启动器。
19 个历史入口以及 scripts/docker/ 的历史文件移至 [归档及路径映射](docs/archive/scripts-before-20261029/README.md)，两个 CPU 数值检查移至 tests/。
旧平铺路径不再保留一套包装脚本；外部命令须使用新路径。完整权重、adapter、断点及指标保持原位置。
原始实验说明和旧命令保存在 [历史 README](docs/history/README-before-scripts-cleanup-20261029.md)；历史命令与授权不是当前入口。
