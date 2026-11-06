"""CPU file maintenance: checkpoints store weights; models stores model code."""
import argparse
import ast
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


LAYOUT = """# 项目目录

```text
sparse_upcycling_qwen3/
  AGENTS.md                 # 资源授权与项目约束
  README.md
  scripts/                  # 转换、训练、合并、数据和评估入口
    docker/                 # 仅供用户手动运行的容器脚本
  models/                   # 模型结构、模型组件等 Python 定义
  checkpoints/              # upcycling、CPT、SFT、DPO 的完整模型权重
  runs/                     # 各实验的 adapter、可恢复断点、配置和指标
  logs/                     # 运行日志
  docs/                     # 实验报告、目录说明、维护记录
```

以项目根目录为工作目录调用 `python3 scripts/<脚本>.py` 或
`bash scripts/<脚本>.sh`。GPU 实验须先报告具体卡号及最低卡数，
经用户批准并由用户启动容器后才能执行。

`models/` 只放模型定义代码，不放模型权重或训练产物。当前完整模型
直接位于 `checkpoints/<模型名>/`，不再使用 checkpoints 符号链接。
基础模型继续使用 `/home/wurendi/models/Qwen-3-8B`，数据继续使用
`/home/wurendi/datasets/`。`runs/` 中的历史 adapter、断点和指标原位保留。

目录维护只使用 CPU 文件操作，不启动容器、模型或实验。
`docs/maintenance/` 保存历史迁移记录，旧记录中的路径描述仅反映当时状态。
"""


def internal(root, path):
    resolved = path.resolve()
    resolved.relative_to(root)
    if resolved == root:
        raise ValueError("Cannot move the project root")
    return resolved


def inventory(folder):
    return {str(path.relative_to(folder)): path.stat().st_size
            for path in folder.rglob("*") if path.is_file()}


def organize(root):
    root = root.resolve(strict=True)
    if root.name != "sparse_upcycling_qwen3":
        raise ValueError("Unexpected project directory: " + str(root))
    models = root / "models"
    source = models / "checkpoints"
    target = root / "checkpoints"
    internal(root, source)
    internal(root, target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "docs" / "maintenance" / ("checkpoint-location-" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    modified = [root / "AGENTS.md", root / "README.md",
                root / "docs/PROJECT_LAYOUT.md", root / "scripts/summarize_results.py"]
    for path in modified:
        if path.is_file():
            dest = backup / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    before = inventory(source) if source.is_dir() else inventory(target) if target.is_dir() else {}
    moved = False
    if source.is_dir():
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError("Unexpected checkpoint symlink")
            target.unlink()  # Only remove the confirmed compatibility link.
        elif target.exists():
            raise FileExistsError(target)
        source.rename(target)  # Metadata-only move on the same filesystem.
        moved = True
    else:
        target.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    # Do not replace maintained architecture/trainer documentation on reruns.
    if not (models / "README.md").exists():
        (models / "README.md").write_text(
            "# 模型定义代码\n\n此目录用于模型结构、路由器、专家、训练器和推理组件。\n"
            "完整模型权重保存于 checkpoints/；adapter 和训练断点保存于 runs/。\n",
            encoding="utf-8")
    policy = root / "AGENTS.md"
    text = policy.read_text(encoding="utf-8-sig")
    text = text.replace(
        "- models/checkpoints/：upcycling、CPT、SFT、DPO 的完整模型。基础模型继续使用 /home/wurendi/models/Qwen-3-8B，不重复复制。",
        "- models/：只放模型结构及组件定义的 Python 代码，不存放 checkpoint。\n"
        "- checkpoints/：upcycling、CPT、SFT、DPO 的完整模型权重。基础模型继续使用 /home/wurendi/models/Qwen-3-8B，不重复复制。")
    text = text.replace(
        "- 历史 checkpoints -> models/checkpoints 符号链接仅用于兼容已有日志和断点路径，不能重复存放模型。\n", "")
    policy.write_text(text, encoding="utf-8")
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8-sig")
    text = text.replace("完整模型位于 `models/checkpoints/`", "完整模型位于 `checkpoints/`，`models/` 用于模型定义代码")
    readme.write_text(text, encoding="utf-8")
    summary = root / "scripts/summarize_results.py"
    text = summary.read_text(encoding="utf-8-sig")
    text = text.replace('CHECKPOINTS = ROOT / "models" / "checkpoints"', 'CHECKPOINTS = ROOT / "checkpoints"')
    summary.write_text(text, encoding="utf-8")
    if not (root / "docs/PROJECT_LAYOUT.md").exists():
        (root / "docs/PROJECT_LAYOUT.md").write_text(LAYOUT, encoding="utf-8")
    after = inventory(target)
    assert before == after, "Checkpoint file inventory changed"
    assert not target.is_symlink() and not source.exists()
    for path in (root / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    result = {"root": str(root), "checkpoint_directory_moved": moved,
              "checkpoint_files_preserved": len(after), "checkpoint_bytes": sum(after.values()),
              "models_directory": "Python model definitions only", "checkpoints_directory": "checkpoints/",
              "gpu_operations": False, "container_operations": False, "file_inventory": after}
    (backup / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "file_inventory"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    organize(parser.parse_args().root)
