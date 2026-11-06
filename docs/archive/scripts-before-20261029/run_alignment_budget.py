"""Run the authorized alignment continuation with a strict GPU reservation deadline."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/alignment-20261020"
LOGS = ROOT / "logs/alignment-20261020"
SFT = "checkpoints/qwen3-8b-8x2-norm-sft-alpaca-merged-v2"
DPO = "checkpoints/qwen3-8b-8x2-norm-dpo-hh-merged-20261020"
BASE = "checkpoints/qwen3-8b-8x2-half-normalized-cpt-simplewiki-1m"
DPO_RUN = "runs/dpo-hh-20261020"
DATA = "/home/wurendi/datasets/domain_sft_dpo_20260930"
WIKI = "/home/wurendi/datasets/sparse_upcycling_wikitext2/validation.jsonl"
PYTHON = sys.executable
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def status(phase, **extra):
    content = {"phase": phase, "updated_utc": datetime.now().astimezone().isoformat(), **extra}
    tmp = RUN / "pipeline_status.json.tmp"
    tmp.write_text(json.dumps(content, indent=2))
    tmp.replace(RUN / "pipeline_status.json")
    print(json.dumps(content), flush=True)


def command(stage, argv, gpus, deadline):
    if STOP or time.time() >= deadline - 60:
        raise TimeoutError("Reservation deadline reached")
    status(stage, command=argv, logical_gpus=gpus)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, APPROVED_PHYSICAL_GPUS="8,9,10,11",
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    with (LOGS / (stage + ".log")).open("a", buffering=1) as logfile:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def forward():
            for line in process.stdout:
                logfile.write(line)
                print(line, end="", flush=True)
        thread = threading.Thread(target=forward, daemon=True)
        thread.start()
        try:
            while process.poll() is None:
                if STOP or time.time() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError("Worker stopped before reservation expiry")
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            thread.join(timeout=10)
            process.stdout.close()
        if process.returncode:
            raise RuntimeError(f"{stage} exited with code {process.returncode}; see {logfile.name}")


def merge(stage, base, adapter, output, deadline):
    marker = ROOT / output / "merge_manifest.json"
    if marker.is_file():
        manifest = json.loads(marker.read_text())
        if manifest.get("status") != "complete" or manifest.get("adapter_sha256") != hashlib.sha256((ROOT / adapter).read_bytes()).hexdigest():
            raise RuntimeError(f"Completed merge has unexpected provenance: {output}")
        print(f"Reusing verified merge marker: {output}", flush=True)
        return
    command(stage + "-merge", [PYTHON, "scripts/merge_adapters.py", "--base", base, "--adapter", adapter,
                               "--output", output, "--stage", stage], "", deadline)


def evaluate(stage, model, adapter, probe, deadline):
    output = RUN / (stage + ".json")
    if output.is_file() and probe.is_file():
        return
    args = [PYTHON, "scripts/evaluate.py", "--model", model, "--data", WIKI,
            "--output", str(output), "--blocks", "8", "--block", "64", "--max-memory", "28GiB",
            "--save-probe", str(probe)]
    if adapter:
        args += ["--adapter", adapter]
    command(stage, args, "0,1", deadline)


def compare(stage, before, after):
    with np.load(before) as left, np.load(after) as right:
        x, y = left["logits"].astype(np.float64), right["logits"].astype(np.float64)
        delta = np.abs(x - y)
        loss_delta = float(abs(left["losses"].mean() - right["losses"].mean()))
        def logsoftmax(values):
            values = values - values.max(axis=-1, keepdims=True)
            return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))
        logp, logq = logsoftmax(x), logsoftmax(y)
        kl = (np.exp(logp) * (logp - logq)).sum(axis=-1)
    report = {"mean_absolute_logits_difference": float(delta.mean()),
              "maximum_absolute_logits_difference": float(delta.max()),
              "absolute_mean_loss_difference": loss_delta,
              "mean_kl_nats": float(kl.mean()), "maximum_kl_nats": float(kl.max()),
              "top1_agreement": float((x.argmax(axis=-1) == y.argmax(axis=-1)).mean()),
              "mean_kl_tolerance": 0.005, "maximum_kl_tolerance": 0.01, "mean_loss_tolerance": 0.01,
              "note": "BF16 merging rounds expert weights. Gate on probability KL and target loss; a constant logit shift changes neither probability nor loss."}
    report["passed"] = report["mean_kl_nats"] <= 0.005 and report["maximum_kl_nats"] <= 0.01 and loss_delta <= 0.01
    (RUN / (stage + "-merge-verification.json")).write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"{stage} merge numerical agreement failed: {report}")
    print(json.dumps(report), flush=True)


def main():
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    approval = json.loads((RUN / "authorization.json").read_text())
    if approval["container"] != "wrd-sparse-upcycling" or approval["physical_gpus"] != [8, 9, 10, 11]:
        raise RuntimeError("Unexpected authorized container or GPUs")
    now = time.time()
    starts = datetime.fromisoformat(approval["starts_at"]).timestamp()
    deadline = datetime.fromisoformat(approval["hard_stop_at"]).timestamp()
    if not starts <= now < deadline:
        raise RuntimeError("Outside the user-authorized reservation window")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    snapshots = RUN / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for name in ("train_dpo.py", "merge_adapters.py", "evaluate.py", "generate_samples.py", "run_alignment_budget.py"):
        file = ROOT / "scripts" / name
        (snapshots / name).write_bytes(file.read_bytes())
    try:
        merge("sft", BASE, "runs/sft-alpaca/adapter.pt", SFT, deadline)
        evaluate("sft-adapter-eval", BASE, "runs/sft-alpaca/adapter.pt", RUN / "sft-adapter-probe.npz", deadline)
        evaluate("sft-merged-eval", SFT, None, RUN / "sft-merged-probe.npz", deadline)
        compare("sft", RUN / "sft-adapter-probe.npz", RUN / "sft-merged-probe.npz")
        if not (ROOT / DPO_RUN / "complete.json").is_file():
            args = [PYTHON, "scripts/train_dpo.py", "--model", SFT, "--data", DATA + "/hh_pairs.jsonl",
                    "--eval-data", DATA + "/hh_val.jsonl", "--output", DPO_RUN, "--epochs", "1",
                    "--beta", "0.1", "--max-len", "640", "--grad-accum", "8", "--rank", "8",
                    "--lr", "5e-5", "--min-lr", "5e-6", "--warmup", "10", "--save-every", "5",
                    "--eval-every", "20", "--max-memory", "18GiB", "--stop-at", approval["training_stop_at"], "--smoke-test"]
            if (ROOT / DPO_RUN / "resume.pt").is_file():
                args.append("--resume")
            command("dpo-training", args, "0,1,2,3", deadline)
        if not (ROOT / DPO_RUN / "complete.json").is_file():
            status("stopped-with-resumable-progress", run=DPO_RUN)
            return
        merge("dpo", SFT, DPO_RUN + "/adapter.pt", DPO, deadline)
        evaluate("dpo-adapter-eval", SFT, DPO_RUN + "/adapter.pt", RUN / "dpo-adapter-probe.npz", deadline)
        evaluate("dpo-merged-eval", DPO, None, RUN / "dpo-merged-probe.npz", deadline)
        merge_warning = None
        try:
            compare("dpo", RUN / "dpo-adapter-probe.npz", RUN / "dpo-merged-probe.npz")
        except RuntimeError:
            audit = RUN / "dpo-weight-audit.json"
            if not audit.is_file() or not json.loads(audit.read_text()).get("passed"):
                raise
            merge_warning = "Exact merged tensor audit passed, but BF16 inference difference exceeded the probability gate. Use the SFT base plus DPO adapter as the reference policy."
            status("dpo-complete-merged-inference-warning", warning=merge_warning)
        for label, model in (("sft", SFT), ("dpo", DPO)):
            argv = [PYTHON, "scripts/generate_samples.py", "--model", SFT if label == "dpo" and merge_warning else model,
                    "--output", str(RUN / (label + "-generations.jsonl")), "--max-memory", "28GiB"]
            if label == "dpo" and merge_warning:
                argv += ["--adapter", DPO_RUN + "/adapter.pt"]
            command(label + "-generation", argv, "0,1", deadline)
        status("complete-with-merged-inference-warning" if merge_warning else "complete",
               sft_checkpoint=SFT, dpo_checkpoint=DPO, dpo_adapter=DPO_RUN + "/adapter.pt", warning=merge_warning)
    except TimeoutError as exc:
        status("stopped-at-reservation-deadline", reason=str(exc), run=DPO_RUN)
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
