"""Download a small public text corpus for a pipeline smoke run."""
import json
import subprocess
from pathlib import Path

root = Path("/home/wurendi/datasets/sparse_upcycling_wikitext2")
root.mkdir(parents=True, exist_ok=True)
for name, source_name in (("train", "train"), ("validation", "valid")):
    filename = f"wiki.{source_name}.raw"
    raw = root / filename
    url = f"https://cosmo.zip/pub/datasets/wikitext-2-raw/{filename}"
    if not raw.exists():
        subprocess.run(["curl", "-4", "-fsSL", "--retry", "3", "--max-time", "120",
                        "-o", str(raw), url], check=True)
    rows = raw.read_text(encoding="utf-8").splitlines()
    path = root / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = row.strip()
            if text:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(path, path.stat().st_size, flush=True)
