# 2026-10-20 SFT 合并修复与 DPO

本次人工授权：用户启动容器 `wrd-sparse-upcycling`，登记物理 GPU
8、9、10、11，时间为 2026-10-20 11:00–16:00（UTC+8）。
授权记录保存在 `runs/alignment-20261020/authorization.json`。

## 已完成的修复和验证

- 合并脚本兼容 SFT/DPO adapter 元数据，不再提前求值不存在的 CPT 字段。
- 保存独立的阶段清单和合并完成标记；所有 882 个 adapter/路由器张量均已消费。
- 评估从 adapter 读取 rank，不再将 rank-8 SFT 错按 rank 4 加载。
- DPO 标签移到输出 logits 所在设备，兼容跨 GPU 的 embedding 与 lm_head。
- reference 缓存通过文件句柄原子写入，包含模型和 tokenized pairs 的指纹，支持部分缓存恢复。
- 续训按全局优化步跳过已完成的数据组；验证指标使用 policy/reference 的差分。
- 六项 CPU 回归检查通过：合并、缓存写入、部分缓存恢复、续训数据位置、概率检查、到期终止子进程。

已有 SFT 不重新训练。新独立模型：
`checkpoints/qwen3-8b-8x2-norm-sft-alpaca-merged-v2`。

相同 512-token WikiText-2 验证设置下，adapter PPL 为 10.2541556，
合并 PPL 为 10.2659985；平均 loss 差为 0.0011543。
8 个抽查位置的平均 KL 为 0.0010123 nats，top-1 一致率 100%。
这仅验证小样本上的数值接近，不是正式 benchmark 或逐位一致性证明。
初始的原始 logits MAE 检查记录也已保留；概率检查对公共 logits 平移不敏感。

## DPO 设置与输出

- 从上述合并 SFT 模型开始，rank-8 专家 LoRA 加可训练路由器，基础权重冻结。
- HH 数据经过长度/空回答过滤后，793 个训练 pair、19 个验证 pair。
- 1 epoch，梯度累积 8，100 个优化步，beta=0.1，最大长度 640。
- 初始 B=0 的 policy 等于 SFT reference；最长组合样本反向检查不进行 optimizer 更新。
- reference 分数在首次 optimizer 更新前缓存；之后只训练 policy。
- 每 5 步保留 adapter-latest.pt 和 resume.pt，每 20 步评估。
- DPO 训练产物：`runs/dpo-hh-20261020/`。
- 完成后独立模型：`checkpoints/qwen3-8b-8x2-norm-dpo-hh-merged-20261020`。
- 状态与数值验证：`runs/alignment-20261020/`。
- 日志：`logs/alignment-20261020/`。

最长组合样本（chosen 608、rejected 553 tokens）的反向检查已通过，
四卡峰值 allocated 显存约为 12.1、17.3、15.1、22.3 GiB。
该检查包含 0 次 optimizer 更新，当前训练进度应以 metrics.jsonl 为准。

## tmux 和到期释放

容器内 tmux 会话 `upcycling-align`、窗口 `sft-dpo`：

```bash
docker exec -it wrd-sparse-upcycling tmux attach -t upcycling-align
```

退出查看时按 Ctrl-b、d，不中断流程。

DPO 在 15:45 主动保存并退出；总流程最迟 15:55 终止 GPU 子进程。
另有停止专用的到期守护，在 15:58 停止本次容器的固定 ID，不启动或重启容器。
若登记时间不足以完成全部阶段，保留断点等待用户另行登记和批准。
