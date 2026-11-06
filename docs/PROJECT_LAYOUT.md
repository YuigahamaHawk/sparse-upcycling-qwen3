# 项目目录与代码职责

```text
sparse_upcycling_qwen3/
├── AGENTS.md          # 资源授权与开发规则
├── README.md
├── models/            # 模型、forward、BasicTrainer/阶段训练器与推理等实际实现
│   ├── backbones/     # 模型形状、架构适配与 HF 类加载
│   ├── components/    # 专家 LoRA、路由 forward、兼容实现
│   ├── trainers/      # BasicTrainer、SmokeCPT/CPT/SFT/DPO、checkpoint IO
│   ├── conversion/    # dense→MoE 和 adapter 合并
│   ├── evaluation/    # loss/PPL 与 forward 诊断
│   ├── inference/     # 推理引擎、生成、API 与 prompt/registry
│   └── runtime/       # CPU 监督进程、授权时段与 GPU 互斥
├── scripts/           # 通用 CLI，根目录仅保留导航
│   ├── data/          # 语料准备与 token 化
│   ├── conversion/    # upcycle 与 adapter 合并
│   ├── training/      # smoke CPT、CPT、SFT、DPO
│   ├── evaluation/    # loss/PPL、样例、forward/合并审计
│   └── deployment/    # HTTP 服务、独立推理、客户端
├── configs/           # 模型登记、运行配置和授权记录
├── tests/             # CPU 组件与行为回归、CLI 和依赖边界检查
├── checkpoints/       # 完整模型权重
├── runs/              # adapter、resume、实验配置、指标与阶段状态
├── logs/              # 运行日志
└── docs/              # 使用说明、报告、维护记录
```

## 依赖方向

scripts/命令入口 → models/阶段实现 → models/共享组件与架构 → Transformers/PyTorch

models/ 的实际实现不导入训练入口；模型加载器不再通过导入训练脚本获得 LoRA 或 forward 补丁。旧平铺 scripts/ 路径和重复兼容别名已归档，正式实现只有一份。具体文件见 models/README.md。

## 资产与运行

- 完整 checkpoint 保持在 checkpoints/<模型名>/；models/ 里不放权重。
- 基础模型仍为 /home/wurendi/models/Qwen-3-8B；数据仍在 /home/wurendi/datasets/。
- runs/ 中历史 adapter、断点和指标原位保留。docs/maintenance/ 的源码备份与路径仅反映迁移时状态。
- 当前命令见 scripts/README.md，使用 python3 scripts/<分类>/<入口>.py；不为历史路径重复保留包装脚本。
- 过期实验流程、Docker 脚本、一次性迁移工具见 docs/archive/scripts-before-20261029/；它们仅供追溯，不是可执行的当前流程。原始字节、哈希和逐文件迁移表见 docs/maintenance/scripts-cleanup-20261029/。
- 历史数值检查归入 tests/；新增实验优先复用入口和参数，避免继续堆积专用 run/resume 脚本。维护脚本不得用过期模板覆盖本说明或 models/README.md。
- GPU 任务须先说明具体物理卡号、最低卡数及依据，获批后在用户启动的容器中运行。过期授权不得复用。
- CPU 结构整理不启动容器、不加载真实 checkpoint、不运行 GPU 实验。
