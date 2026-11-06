"""Summarize completed sparse-upcycling runs without loading model weights."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CHECKPOINTS = ROOT / "checkpoints"


def read_json(path):
    return json.loads(path.read_text())


def read_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_info(name):
    path = CHECKPOINTS / name
    index = read_json(path / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty model shards in {path}: {missing}")
    config = read_json(path / "config.json")
    return {"path": str(path), "norm_topk_prob": config["norm_topk_prob"],
            "shards": len(shards),
            "weight_bytes": sum((path / name).stat().st_size for name in shards),
            "manifest": read_json(path / "cpt_manifest.json")}


dense = read_json(RUNS / "eval/dense.json")
variants = {
    "unnormalized": {
        "run": "simplewiki-cpt-1m", "wiki_run": "wikitext-20steps",
        "initial": RUNS / "eval/moe-initial.json",
        "cpt20": RUNS / "simplewiki-cpt-1m/baseline_eval.json",
        "final_checkpoint": "qwen3-8b-8x2-half-cpt-simplewiki-1m",
    },
    "normalized": {
        "run": "simplewiki-normalized-cpt-1m",
        "wiki_run": "wikitext-normalized-20steps",
        "initial": RUNS / "simplewiki-normalized-cpt-1m/initial_eval.json",
        "cpt20": RUNS / "simplewiki-normalized-cpt-1m/cpt20_eval.json",
        "final_checkpoint": "qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m",
    },
}

summary = {"evaluation": {"corpus": "WikiText-2 validation", "tokens": dense["tokens"],
                          "blocks": dense["blocks"], "dense": dense},
           "variants": {}}
curve = {}
for label, spec in variants.items():
    run_dir = RUNS / spec["run"]
    if (run_dir / "pipeline.status").read_text().strip() != "complete":
        raise RuntimeError(f"Run is not complete: {run_dir}")
    metrics = read_metrics(run_dir / "metrics.jsonl")
    if len(metrics) != 1000 or metrics[-1]["step"] != 1000:
        raise RuntimeError(f"Expected exactly 1000 steps: {run_dir}")
    wiki_metrics = read_metrics(RUNS / spec["wiki_run"] / "metrics.jsonl")
    result = {"initial": read_json(spec["initial"]),
              "after_wikitext20": read_json(spec["cpt20"]),
              "after_simplewiki_adapter": {k: metrics[-1][k] for k in
                                           ("step", "tokens", "loss", "eval_loss", "eval_perplexity")},
              "after_simplewiki_merged": read_json(run_dir / "merged_eval.json"),
              "wiki_training_tokens": wiki_metrics[-1]["tokens"],
              "simplewiki_training_tokens": read_json(run_dir / "complete.json")["trained_tokens"],
              "run_config": read_json(run_dir / "run.json"),
              "checkpoint": checkpoint_info(spec["final_checkpoint"])}
    result["total_training_tokens"] = result["wiki_training_tokens"] + result["simplewiki_training_tokens"]
    summary["variants"][label] = result
    curve[label] = {0: result["after_wikitext20"]["perplexity"]}
    curve[label].update({entry["step"]: entry["eval_perplexity"] for entry in metrics
                         if "eval_perplexity" in entry})

summary["comparison"] = {
    "final_merged_ppl_difference_unnormalized_minus_normalized":
        summary["variants"]["unnormalized"]["after_simplewiki_merged"]["perplexity"] -
        summary["variants"]["normalized"]["after_simplewiki_merged"]["perplexity"],
    "final_merged_ppl_normalized_relative_to_unnormalized_percent":
        100 * summary["variants"]["normalized"]["after_simplewiki_merged"]["perplexity"] /
        summary["variants"]["unnormalized"]["after_simplewiki_merged"]["perplexity"],
}

(RUNS / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
with (RUNS / "validation_curve.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("simplewiki_step", "unnormalized_adapter_ppl", "normalized_adapter_ppl"))
    for step in sorted(set(curve["unnormalized"]) | set(curve["normalized"])):
        writer.writerow((step, curve["unnormalized"].get(step), curve["normalized"].get(step)))
print(json.dumps({"dense_ppl": dense["perplexity"],
                  "final_merged_ppl": {key: value["after_simplewiki_merged"]["perplexity"]
                                       for key, value in summary["variants"].items()},
                  "total_training_tokens_each": {key: value["total_training_tokens"]
                                                 for key, value in summary["variants"].items()}},
                 ensure_ascii=False))
