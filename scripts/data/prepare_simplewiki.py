"""Fetch and clean a Simple English Wikipedia article dump for CPT."""
import bz2
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell

root = Path("/home/wurendi/datasets/simplewiki_cpt_20260901")
root.mkdir(parents=True, exist_ok=True)
name = "simplewiki-latest-pages-articles.xml.bz2"
archive = root / name
url = f"https://dumps.wikimedia.org/simplewiki/latest/{name}"
subprocess.run(["curl", "-4", "-fL", "--retry", "5", "-C", "-",
                "-o", str(archive), url], check=True)
sha256 = hashlib.sha256()
with archive.open("rb") as f:
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
        sha256.update(chunk)
print("archive_sha256", sha256.hexdigest(), flush=True)

counts = {"train": 0, "validation": 0, "skipped": 0}
chars = {"train": 0, "validation": 0}
train_file = (root / "train.jsonl").open("w", encoding="utf-8")
valid_file = (root / "validation.jsonl").open("w", encoding="utf-8")
try:
    with bz2.open(archive, "rb") as stream:
        for _, page in ET.iterparse(stream, events=("end",)):
            if not page.tag.endswith("}page"):
                continue
            namespace = page.tag.split("}")[0] + "}"
            title = page.findtext(namespace + "title") or ""
            ns = page.findtext(namespace + "ns")
            raw = page.findtext(f"{namespace}revision/{namespace}text") or ""
            if ns != "0" or not raw or raw.lstrip().lower().startswith("#redirect"):
                counts["skipped"] += 1
                page.clear()
                continue
            try:
                clean = mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)
            except Exception:
                counts["skipped"] += 1
                page.clear()
                continue
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) < 200:
                counts["skipped"] += 1
                page.clear()
                continue
            split = "validation" if int(hashlib.sha256(title.encode()).hexdigest(), 16) % 100 == 0 else "train"
            row = json.dumps({"title": title, "text": title + "\n" + clean}, ensure_ascii=False) + "\n"
            (valid_file if split == "validation" else train_file).write(row)
            counts[split] += 1
            chars[split] += len(clean)
            if (counts["train"] + counts["validation"]) % 20000 == 0:
                print("processed", counts, flush=True)
            page.clear()
finally:
    train_file.close()
    valid_file.close()

manifest = {"source": url, "sha256": sha256.hexdigest(),
            "split": "sha256(title) mod 100 == 0 for validation",
            "articles": counts, "clean_characters": chars}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
