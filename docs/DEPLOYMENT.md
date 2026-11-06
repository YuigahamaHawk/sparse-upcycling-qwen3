# 各阶段模型推理服务

## 历史验证资源与服务边界（2026-10-28，授权已过期）

- 服务器：icse；用户已启动容器 `wrd-icse-llm`，代理不得启动或重启容器。
- 本次授权：2026-10-28 原 16:00–21:00，加上用户明确续约的 21:00–22:00；连续有效至 22:00（UTC+8），物理 GPU 2、3、4、5。
- 服务实际选择物理 2、3；容器枚举 0、1 分别对应物理 2、3。可见设备顺序由 `/dev/iluvatar2..5` 和容器检查确认。4、5 不分配推理负载。
- 默认 BF16，单模型实例，2048 总上下文 token，最多输出 512 token，单请求执行；超长输入明确报错，不静默截断。
- 父进程在 21:55 停止自己管理的 GPU 子进程，15 秒内未退出则强制结束该进程组。续约时已重启服务进程以同时更新父、子进程的内存截止时间。不会停止容器或其他人的任务。
- 授权过期后脚本拒绝启动。未来必须由用户确认新卡号、时段和范围，更新 lease；不能沿用旧授权。
- 本次运行的是功能验证和推理服务，不包含训练或正式 benchmark。

## 文件

项目根目录：`/home/wurendi/projects/sparse_upcycling_qwen3`。

| 路径 | 用途 |
|---|---|
| `configs/serving/models.json` | 七个阶段/分支的固定模型登记 |
| `configs/serving/lease-20261028.json` | 本次时段、物理 GPU 与容器编号映射 |
| `models/upcycling_inference.py` | 共享模型加载、adapter、路由器、推理实现 |
| `scripts/deployment/serve_models.py` | HTTP 服务，单进程，无自动重载 |
| `scripts/deployment/infer.py` | 独立本地模型推理，支持 JSONL 批量输入 |
| `scripts/deployment/call_model.py` | 不使用 GPU 的 HTTP 调用客户端 |
| `models/runtime/gpu_lease.py` | CPU 父进程监督 GPU 子进程及到期退出 |
| `scripts/deployment/launch_service.sh` | 容器内服务入口，不启动容器 |
| `runs/serving-20261028/` | 状态、功能验证结果、服务密钥 |
| `logs/serving-20261028.log` | tmux 服务日志 |

`models/` 仅放代码；完整权重保留在 `checkpoints/`，原始 dense 使用 `/home/wurendi/models/Qwen-3-8B`。

2026-10-29 整理后，推理实现位于 `models/inference/engine.py`，HTTP API 位于 `models/inference/api.py`，资源监督位于 `models/runtime/gpu_lease.py`。CLI 统一放在 `scripts/deployment/`，旧平铺脚本路径已归档。以下运行状态和日期只记录历史验证，不代表当前服务在线。

## 模型名称与切换

| 名称 | 阶段 |
|---|---|
| `dense-base` | 原始 dense |
| `moe-init` | 未归一化 MoE 初始化 |
| `moe-init-normalized` | 归一化 MoE 初始化 |
| `moe-cpt` | 未归一化 CPT |
| `moe-cpt-normalized` | 归一化 CPT |
| `moe-sft` | 已修复合并的 SFT，默认服务模型 |
| `moe-dpo` | SFT merged-v2 + 原始 DPO adapter（含路由器） |

21:00–22:00 续约验证后当前加载 `moe-dpo`，以 `GET /readyz` 为准。启动脚本仍可通过 `--model` 指定启动版本。

一个时刻仅加载一个模型。`/admin/load` 先完整卸载当前模型，再加载新模型；在切换完成前不接收生成。加载/生成竞争返回 429。生成请求的 `model` 必须与已加载模型匹配，不隐式切换或额外常驻模型。

DPO 的 882 个 adapter/路由器 tensor 必须完整匹配，检查 base 路径和非有限参数。独立 BF16 DPO 合并模型有已有一致性告警，默认登记不使用它。

实测发现 Corex 在重复切换模型时，Transformers 4.53.2 的 allocator warmup 一次申请约 23 GiB 可出现 OOM，即使报告的空闲显存足够。加载器仅在 `from_pretrained` 调用期间关闭这项大块预分配，继续逐张量加载；不更改权重、精度、路由或推理计算，也不修改安装包。

初始化 checkpoint 的训练配置开启了 `output_router_logits`；在 KV cache 生成阶段计算训练用辅助损失会遇到 router 与完整 attention mask 的长度不匹配。服务加载时明确关闭返回 router logits/辅助损失，仍然执行原本的专家路由；不改写 checkpoint 配置文件。

## 容器内调用示例（历史时段和文件名须替换）

```bash
cd /home/wurendi/projects/sparse_upcycling_qwen3
# 查看正在运行的服务；需要在容器内连接此 tmux。
tmux attach -t llm-serving

# HTTP 客户端不额外加载模型、不占用 GPU。
python3 scripts/deployment/call_model.py --api-key-file runs/serving-20261028/api.key --status
python3 scripts/deployment/call_model.py --api-key-file runs/serving-20261028/api.key --models
python3 scripts/deployment/call_model.py --api-key-file runs/serving-20261028/api.key \
  --model moe-sft --instruction '用一句话解释什么是机器学习。' --max-new-tokens 96

# 显式切换到 DPO；加载需要等待。之后 model 参数也改为 moe-dpo。
python3 scripts/deployment/call_model.py --api-key-file runs/serving-20261028/api.key --load moe-dpo
```

服务在 `127.0.0.1:18082`，容器使用 host 网络，无需创建新容器或修改端口映射。服务密钥为本次随机生成，保存在 `runs/serving-20261028/api.key`，只读给服务使用者，不提交进报告或打印到日志。

如需重新部署，先取得本次 GPU/时段/实验批准，由用户启动容器，再在 tmux 中运行以下命令。CURRENT 是占位符，必须替换为新的有效文件，不能沿用 2026-10-28 授权；启动器不设置旧卡号或旧时段，也不启动容器。

```bash
bash scripts/deployment/launch_service.sh \
  --lease configs/serving/lease-CURRENT.json \
  --api-key-file runs/serving-CURRENT/api.key \
  --status runs/serving-CURRENT/status.json --model moe-sft
```

## Windows 本地调用

在本地终端保持 SSH 隧道打开：

本次已建立本地 `127.0.0.1:18082` 隧道，进程记录在本地 `runs/serving-20261028/ssh-tunnel.json`；API key 也已同步到本地同名目录。隧道仍在时可直接调用，无需再开一个。以下命令供断开后手动重连。

```powershell
ssh -N -L 18082:127.0.0.1:18082 icse
```

然后本机使用 `http://127.0.0.1:18082`。首次可通过 SSH 复制 API key 到仅自己可读的本地文件。示例（客户端脚本与密钥均已下载到本机时）：

```powershell
python scripts/deployment/call_model.py --api-key-file runs/serving-20261028/api.key `
  --model moe-sft --instruction '请概括这段文字。' --input '待处理文本'
```

也可以直接在本地 Python 中调用：

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

key = Path('runs/serving-20261028/api.key').read_text().strip()
req = Request('http://127.0.0.1:18082/generate',
    data=json.dumps({'model': 'moe-sft', 'instruction': '请用一句话解释过拟合。',
                     'max_new_tokens': 96}).encode(),
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
with urlopen(req, timeout=900) as response:
    print(json.load(response)['text'])
```

停止 HTTP 服务时，在容器内对 `llm-serving:serve` 窗口按 Ctrl+C；监督进程会停止它管理的 GPU 子进程。不要只从 tmux detach，那会让服务继续运行。无需停止或重启 Docker。

## 推理接口

所有模型接口均使用 `Authorization: Bearer <key>`。`GET /healthz` 不需要密钥；`GET /readyz` 返回当前模型、PID、时段及运行状态。

浏览器调用：打开 `http://127.0.0.1:18082/docs`，点击右上角 **Authorize**，在 `ModelAPIKey` 中只粘贴 `api.key` 文件里的密钥，不添加 `Bearer `。Swagger 会自动添加该前缀。授权后展开 `POST /generate` → **Try it out** → 填写 JSON → **Execute**；在返回结果的 `text` 中查看回答。若看到旧的 `authorization` 输入框，请强制刷新页面。

2026-10-28 修复了 Swagger 的 401 问题：旧版本把 OpenAPI 保留的 `Authorization` 当作普通 Header 参数，Swagger 生成的请求会遗漏它。现使用标准 HTTP bearer security scheme；Python/curl 客户端仍按原来的 `Authorization: Bearer <key>` 调用，密钥未更换。

### POST /generate

```json
{"model":"moe-sft","instruction":"提取产品名称和数量，输出 JSON。","input":"采购10台设备A。","max_new_tokens":128,"temperature":0,"seed":42}
```

也可以传 `prompt` 而非 `instruction/input`，原始 prompt 不包模板。返回 `model`、`text`、`finish_reason`、`usage`、`latency_seconds`、`completion_tokens_per_second`。输出 JSON 是任务指令，不承诺模型一定生成合法 JSON；调用方应校验。

### POST /v1/completions、POST /v1/chat/completions

提供常用的非流式兼容格式，支持 `model`、`max_tokens`、`temperature`、`top_p`、`seed`、`stop`，仅 `n=1`、`stream=false`。不支持工具调用、图像、logprobs、流式 SSE 等字段；显式返回校验错误，不静默忽略。

`/v1/completions` 的 `prompt` 原样输入。`/v1/chat/completions` 接收文本 `messages`，内部仍采用训练时 Alpaca 模板；可把历史对话序列化为输入，但本次 SFT 没有验证多轮能力，不宣称等价于官方 Qwen ChatML 或 thinking 模式。

`GET /v1/models` 同时列出已登记与当前加载状态。`POST /admin/load` 接收 `{"model":"dense-base"}`；`POST /admin/unload` 释放权重，但 CUDA 上下文要到服务进程退出后才彻底释放。

## 独立推理 / 后续 benchmark 入口

独立 CLI 与 HTTP 使用同一加载器。它会加载另一实例，**先退出 HTTP 服务并确认 GPU 子进程结束后再运行**，避免两个实例重叠。它也受同一个 CPU watchdog 和授权时段保护。

两个入口在 CPU 父进程中持有相同的物理 GPU 文件锁；已有服务时，第二个入口会在加载模型前拒绝启动，避免意外重复分配权重。锁文件保留并不表示 GPU 仍被占用，进程结束会释放系统锁。

```bash
python3 scripts/deployment/infer.py --lease configs/serving/lease-20261028.json \
  --model moe-sft --instruction '用一句话解释机器学习。' \
  --max-new-tokens 96 --output runs/serving-20261028/standalone-new.jsonl

# cases.jsonl 每行包含 id 和 prompt，或 id、instruction、input。
python3 scripts/deployment/infer.py --lease configs/serving/lease-20261028.json \
  --model moe-cpt-normalized --input-jsonl /home/wurendi/datasets/your_cases/cases.jsonl \
  --max-new-tokens 128 --output runs/your-evaluation/cpt.jsonl
```

输出逐条 flush，不覆盖已有结果。比较能力时固定评测集、prompt、生成参数和评分方式；功能验证样例不是正式 benchmark。dense/CPT 是 base 阶段，指令遵循与 SFT/DPO 的差异需结合训练阶段解读。

## CPU 测试与依赖

容器现有版本：Python 3.12，torch 2.4.1+Corex4.3.6，transformers 4.53.2，accelerate 1.9.0，safetensors 0.5.3，FastAPI 0.116.1，Uvicorn 0.35.0，Pydantic 2.11.7。优先复用已验证镜像，不直接升级推理栈。

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -p test_serving.py -v
```

测试使用假模型，只验证 API 合约、模板、并发拒绝、授权时段、GPU 映射和 watchdog 对加载中进程的退出能力，不加载权重。真实功能验证记录另见 `runs/serving-20261028/`。

## 2026-10-28 功能验收结果

- 15 项 CPU 单元测试通过（含 Swagger bearer security scheme 回归检查）：`runs/serving-20261028/cpu-tests.log`。
- 七个登记模型均通过顺序加载、原始 completion 和 instruction 生成：`runs/serving-20261028/http-smoke-v3.json`。
- 非流式 chat、密钥验证、输出上限、超长输入拒绝和未登记模型拒绝通过。
- DPO adapter 与 18 个路由器共 882 个 tensor 完整匹配，使用原始 adapter，不使用有一致性告警的独立 BF16 DPO 合并模型。
- HTTP 服务存活时，第二个独立推理入口在加载前被 GPU 文件锁拒绝：`runs/serving-20261028/overlap-check.json`。
- 停止 HTTP GPU 进程后，独立 CLI 生成成功并以状态码 0 退出：`runs/serving-20261028/standalone-smoke.jsonl`。
- 验证后恢复默认 `moe-sft` 服务。早期失败的 `http-smoke.json`、`http-smoke-v2.json` 与日志保留供排错；最终通过记录是 `http-smoke-v3.json`。

这些是部署功能检查。样例极少、输出长度很短，不代表正式 benchmark 得分，也不能据此判断模型能力高低。

## 21:00–22:00 续约

同一容器、同一 API 地址和密钥继续使用；实际仍仅用物理 GPU 2、3。续约检查记录在 `runs/serving-20261028/renewal-base-dpo-smoke.json`，通过后当前服务保留 `moe-dpo`。

在 Swagger 页面完成 Authorize 后，先执行 `POST /admin/load`：

```json
{"model": "dense-base"}
```

加载完成后，`POST /generate` 的 `model` 同样填写 `dense-base`。切回 DPO 时，先向 `/admin/load` 提交 `{"model":"moe-dpo"}`，再把生成请求的 `model` 改为 `moe-dpo`。模型切换需要等待，生成请求不会隐式切换模型。Base 没有经过本项目的指令 SFT；比较续写能力时可使用 `/v1/completions` 的原始 prompt。
