"""Download and subset public SFT/DPO corpora for the domain-model pipeline.

Produces, under --output:
  alpaca_train.jsonl / alpaca_val.jsonl  {"prompt", "completion"} rows
  hh_pairs.jsonl / hh_val.jsonl          {"prompt", "chosen", "rejected"} rows
  manifest.json                          download URLs, SHA-256, row counts

The downloads happen inside the Docker container; no model work is done here.
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.inference.common import TEMPLATE

ALPACA_URLS = [
    "https://hf-mirror.com/datasets/tatsu-lab/alpaca/resolve/main/alpaca_data.json",
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
]
HH_URLS = [
    "https://hf-mirror.com/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl.gz",
    "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl.gz",
]

ASSISTANT_TAG = "\n\nAssistant: "


def fetch(urls, dest):
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"cached: {dest}", flush=True)
        return
    last = None
    for url in urls:
        try:
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            print(f"saved {dest} ({dest.stat().st_size} bytes)", flush=True)
            return
        except Exception as exc:  # try the next mirror
            last = exc
            print(f"failed: {exc}", flush=True)
    raise RuntimeError(f"All mirrors failed for {dest}: {last}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpaca_rows(raw):
    for item in raw:
        output = item.get("output", "").strip()
        if not output:
            continue
        inp = item.get("input", "").strip()
        prompt = TEMPLATE.format(instruction=item["instruction"].strip(),
                                 input_part=f"### Input:\n{inp}\n\n" if inp else "")
        yield {"prompt": prompt, "completion": output}


def hh_rows(raw_lines):
    for line in raw_lines:
        if not line.strip():
            continue
        row = json.loads(line)
        chosen, rejected = row["chosen"], row["rejected"]
        idx = chosen.rfind(ASSISTANT_TAG)
        idx_r = rejected.rfind(ASSISTANT_TAG)
        if idx < 0 or idx_r < 0:
            continue
        # Prompt must be shared by both sides; HH pairs share the dialogue prefix.
        prompt = chosen[: idx + len(ASSISTANT_TAG)]
        if rejected[: idx_r + len(ASSISTANT_TAG)] != prompt:
            continue
        yield {"prompt": prompt, "chosen": chosen[idx + len(ASSISTANT_TAG):],
               "rejected": rejected[idx_r + len(ASSISTANT_TAG):]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sft-train", type=int, default=2000)
    p.add_argument("--sft-val", type=int, default=100)
    p.add_argument("--dpo-pairs", type=int, default=800)
    p.add_argument("--dpo-val", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir(exist_ok=True)

    alpaca_file = raw_dir / "alpaca_data.json"
    fetch(ALPACA_URLS, alpaca_file)
    hh_file = raw_dir / "harmless-base-train.jsonl.gz"
    fetch(HH_URLS, hh_file)

    alpaca_raw = json.loads(alpaca_file.read_text(encoding="utf-8"))
    rows = list(alpaca_rows(alpaca_raw))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    take = args.sft_train + args.sft_val
    if len(rows) < take:
        raise RuntimeError(f"Alpaca usable rows {len(rows)} < requested {take}")
    subsets = {"alpaca_train.jsonl": rows[:args.sft_train],
               "alpaca_val.jsonl": rows[args.sft_train:take]}

    with gzip.open(hh_file, "rt", encoding="utf-8") as handle:
        hh_raw = handle.readlines()
    pairs = list(hh_rows(hh_raw))
    rng.shuffle(pairs)
    take = args.dpo_pairs + args.dpo_val
    if len(pairs) < take:
        raise RuntimeError(f"HH usable pairs {len(pairs)} < requested {take}")
    subsets["hh_pairs.jsonl"] = pairs[:args.dpo_pairs]
    subsets["hh_val.jsonl"] = pairs[args.dpo_pairs:take]

    manifest = {"alpaca_sha256": sha256(alpaca_file), "hh_sha256": sha256(hh_file),
                "alpaca_usable": len(list(alpaca_rows(alpaca_raw))),
                "hh_usable": len(list(hh_rows(hh_raw))), "seed": args.seed, "files": {}}
    for name, subset in subsets.items():
        path = args.output / name
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][name] = len(subset)
        print(f"wrote {path}: {len(subset)} rows", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
