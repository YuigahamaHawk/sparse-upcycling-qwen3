# GitHub 源码仓库说明

## 首次上传范围

- 上传日期：2026-11-06（UTC+8）。
- 来源：本机 sparse_upcycling_qwen3 工作副本；其中 models/ 的 30 个 Python 文件曾于 2026-11-02 与 icse 逐文件核对一致。
- 本次上传时 icse SSH 连接超时，未声称已经包含服务器在此之后的全部改动。恢复连接后，应先比较差异再同步，避免覆盖服务器上的人工修改。
- 上传模型结构、forward、训练器、转换/评估/部署入口、测试、配置及最终实验报告。
- 不上传模型权重、LoRA adapter、训练断点、优化器状态、数据集、运行日志、API key 和临时维护快照。

## 模型文件在仓库外

models/ 保存 Python 实现代码，保留在 Git 中。完整模型在 checkpoints/，训练状态在 runs/，语料在外部 datasets 目录；这些文件不会随 git clone 下载。

克隆后须自行准备对应权重，并根据自己的目录修改 configs/serving/models.json 中的 project_root、path 和 adapter。配置中的绝对路径和历史报告中的路径反映原实验环境，不是随仓库提供的模型。

## 环境

原 GPU 实验环境为天数 Corex 4.3.6、厂商 PyTorch 2.4.1、Transformers 4.53.2。硬件对应的 PyTorch/驱动由所用平台提供；不要用普通 CUDA wheel 直接覆盖 Corex 环境。

本地 CPU 回归使用 Python 3.10、CPU-only PyTorch 2.9.1、Transformers 4.53.2、Accelerate 1.9.0、Safetensors 0.5.3、FastAPI 0.116.1、Pydantic 2.11.7、HTTPX 0.28.1。CPU 检查不表示已验证其他显卡或新模型架构。

## 运行与授权

GPU 使用规则见 AGENTS.md。用户登记并确认具体物理 GPU、时间及实验范围后，由用户启动容器，再在容器中的 tmux 内运行实验。

真实 lease 和 API key 均不提交。configs/serving/lease.example.json 是故意过期、卡号为空的模板；复制为本地 lease-current.json 后按本次真实授权填写，不应直接运行模板。

docs/archive/ 与 docs/history/ 是历史资料；其中启动器、时段和模型路径不能作为当前启动命令。当前入口以 scripts/README.md 为准。

## CPU 回归

在安装了 CPU-only PyTorch 及上述依赖的环境，从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

Windows 下 Linux 专用进程组/文件锁测试会跳过。测试使用临时小张量和模拟服务，不需要实际 Qwen 权重。
