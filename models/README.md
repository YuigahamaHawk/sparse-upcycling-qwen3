# 模型与训练实现

本目录保存模型形状、路由与专家 forward、LoRA、训练器、转换/合并、评估、推理和运行保护。完整权重在 checkpoints/，adapter、断点及指标在 runs/。

```text
models/
├── paths.py                         # 项目根目录；无 Torch/GPU 副作用
├── backbones/
│   └── qwen3.py                     # Qwen3Shape、Qwen3Backbone、稀疏层定位与配置构造
├── components/
│   ├── lora.py                      # LoRALinear.forward、专家注入、冻结/训练参数选择
│   └── moe.py                       # MoE routing forward、显式安装 Corex/HF 兼容实现
├── trainers/
│   ├── basic_trainer.py             # BasicTrainer：加载、冻结/LoRA、优化器、原子保存接口
│   ├── checkpoint.py                # 共用原子保存；历史 state/args/optimizer 格式
│   ├── common.py                    # SFT/DPO 共用学习率；CPT/SFT 共用验证 loss
│   ├── smoke_cpt.py                 # SmokeCPTTrainer：JSONL 小规模 CPT
│   ├── cpt.py                       # CPTTrainer：预分词数据、累积、评估与断点续训
│   ├── sft.py                       # SFTTrainer：completion mask、验证与保存
│   └── dpo.py                       # DPOTrainer：序列 log-prob、reference 缓存与偏好损失
├── conversion/
│   ├── upcycle.py                   # Qwen3Upcycler：复制专家与分片输出
│   └── merge.py                     # AdapterMerger：合并 LoRA 与路由器
├── evaluation/
│   ├── language_model.py            # LanguageModelEvaluator：loss/PPL/probe
│   ├── diagnostics.py               # ForwardDiagnostics：逐层有限性诊断
│   └── merged_weights.py            # MergedWeightAuditor：独立 CPU 合并张量审计
├── inference/
│   ├── engine.py                    # InferenceEngine：加载/卸载、adapter 与生成
│   ├── generation.py                # SampleGenerator：固定样例生成
│   ├── api.py                       # HTTP schemas、鉴权、API 路由
│   └── common.py                    # registry、prompt、请求参数与停止条件
├── runtime/
│   └── gpu_lease.py                 # 授权时段、文件锁、监督进程与到期退出
└── upcycling_inference.py           # 历史导入兼容；实际实现为 inference/engine.py
```

## 实现边界

- models/ 的实际实现不导入 scripts/。脚本负责解析参数并实例化这里的类。
- 专家 LoRA 注入统一使用 components/lora.py 的 inject_expert_lora；训练、评估、诊断与推理不重复实现该循环。
- Alpaca 提示模板只在 inference/common.py 定义，数据准备、样例生成和 API 共用；保留历史字符串格式。
- 合并器和合并审计器分别计算预期张量，以保留独立校验；审计器不调用合并器重新生成被审计结果。
- 四个训练器均继承 BasicTrainer，共用模型/参数/优化器初始化；阶段损失、数据顺序、学习率、评估和断点格式保持兼容。
- Qwen3Shape 记录层数、hidden/FFN/head 维度及专家数。完整 Qwen3 attention/decoder 使用固定版本 Transformers 的实现；本项目修改过的路由和 LoRA forward 在 components/ 维护。
- install_qwen3_patches() 在加载前显式调用，并可重复调用；导入训练脚本不再隐式 monkeypatch Transformers。
- 此次是职责拆分，没有实现 Qwen3.8。新增架构应增加自己的 backbone/组件，再接入训练器和推理加载。

## 调用

当前命令从项目根目录执行，例如 python3 scripts/training/sft.py --help。参数名称和默认值不变。

```python
from models.trainers import SFTTrainer
trainer = SFTTrainer(args)  # 构造不加载 checkpoint、不访问 GPU
# 仅在本次 GPU 使用已获批准、容器由用户启动后，才能执行 trainer.run()
```

旧平铺 scripts/ 兼容模块已归档；新代码直接导入 models.components.lora、models.runtime.gpu_lease 等正式模块。CLI 分类导航见 ../scripts/README.md。
