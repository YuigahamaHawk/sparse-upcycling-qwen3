"""Tokenize cleaned SimpleWiki JSONL into compact uint32 token streams."""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("/home/wurendi/datasets/simplewiki_cpt_20260901"))
args = p.parse_args()
root = args.root
tokenizer = AutoTokenizer.from_pretrained("/home/wurendi/models/Qwen-3-8B", use_fast=True)
manifest = {"tokenizer": "/home/wurendi/models/Qwen-3-8B", "dtype": "uint32_le"}
for split in ("train", "validation"):
    source = root / f"{split}.jsonl"
    target = root / f"{split}.uint32.bin"
    count = 0
    documents = 0
    buffer = []
    with source.open(encoding="utf-8") as incoming, target.open("wb") as outgoing:
        for line in incoming:
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            buffer.extend(ids)
            documents += 1
            if len(buffer) >= 200_000:
                np.asarray(buffer, dtype="<u4").tofile(outgoing)
                count += len(buffer)
                buffer.clear()
            if documents % 20_000 == 0:
                print(split, "documents", documents, "tokens", count + len(buffer), flush=True)
        if buffer:
            np.asarray(buffer, dtype="<u4").tofile(outgoing)
            count += len(buffer)
    digest = hashlib.sha256()
    with target.open("rb") as incoming:
        for chunk in iter(lambda: incoming.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest[split] = {"path": str(target), "documents": documents,
                       "tokens": count, "sha256": digest.hexdigest()}
    print(split, manifest[split], flush=True)
(root / "tokens_manifest.json").write_text(json.dumps(manifest, indent=2))
