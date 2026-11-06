"""Machine-readable status for the SimpleWiki CPT run."""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="runs/simplewiki-cpt-1m")
parser.add_argument("--final", default="checkpoints/qwen3-8b-8x2-half-cpt-simplewiki-1m")
args = parser.parse_args()
root = Path("/home/wurendi/projects/sparse_upcycling_qwen3")
run = root / args.run
metrics = run / "metrics.jsonl"
last = None
if metrics.exists():
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
processes = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True).stdout
running = any("train_long.py" in line and run.name in line
              for line in processes.splitlines())
check = run / "resume.pt"
status_path = run / "pipeline.status"
status = status_path.read_text().strip() if status_path.exists() else "not_started"
result = {
    "pipeline_status": status,
    "training_process_running": running,
    "latest_metric": last,
    "last_metric_age_seconds": round(time.time() - metrics.stat().st_mtime) if metrics.exists() else None,
    "checkpoint_exists": check.exists(),
    "last_checkpoint_age_seconds": round(time.time() - check.stat().st_mtime) if check.exists() else None,
    "final_model_exists": (root / args.final / "model.safetensors.index.json").exists(),
}
print(json.dumps(result, ensure_ascii=False))
