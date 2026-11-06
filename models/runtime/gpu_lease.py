"""Require an explicit, current GPU lease and supervise only our child process."""
import json
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
from models.paths import PROJECT_ROOT
import signal
import subprocess
import sys
import time
from datetime import datetime


def lease_details(path, now=None, verify_devices=True):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    start = datetime.fromisoformat(data["starts_at"])
    end = datetime.fromisoformat(data["expires_at"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("lease timestamps require an explicit timezone")
    start, end = start.timestamp(), end.timestamp()
    margin = float(data.get("shutdown_margin_seconds", 300))
    if margin < 30 or end <= start or end - margin <= start:
        raise ValueError("invalid lease interval or shutdown margin")
    now = time.time() if now is None else now
    if not start <= now < end - margin:
        raise ValueError("outside GPU lease (including shutdown margin); refusing GPU startup")
    allowed = data["authorized_physical_gpus"]
    active = data["active_physical_gpus"]
    order = data["container_device_order"]
    if not active or len(set(active)) != len(active) or not set(active) <= set(allowed):
        raise ValueError("active GPUs must be a unique subset of the authorized physical GPUs")
    if len(set(order)) != len(order) or not set(order) <= set(allowed):
        raise ValueError("container exposes unauthorized or duplicate GPUs")
    try:
        mask = [int(x) for x in data["cuda_visible_devices"].split(",")]
        mapped = [order[i] for i in mask if i >= 0]
    except (ValueError, IndexError):
        raise ValueError("invalid CUDA device mask") from None
    if len(mapped) != len(mask) or mapped != active:
        raise ValueError("CUDA mask does not map to the approved active physical GPUs")
    if verify_devices:
        devices = sorted(int(p.name.removeprefix("iluvatar")) for p in Path('/dev').glob('iluvatar[0-9]*')
                         if p.name.removeprefix("iluvatar").isdigit())
        if devices != sorted(order):
            raise ValueError(f"container device files {devices} differ from approved mapping {order}")
    return data, end - margin


def require_worker_lease():
    path = os.environ.get("UPCYCLING_GPU_LEASE")
    if not path:
        raise ValueError("GPU worker must be started through the supervised CLI entry point")
    data, deadline = lease_details(path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != data["cuda_visible_devices"]:
        raise ValueError("worker GPU visibility differs from lease")
    # A dead tmux/supervisor must not leave an orphan GPU worker behind.
    import ctypes
    expected_parent = int(os.environ.get("UPCYCLING_LEASE_SUPERVISOR_PID", "-1"))
    if os.getppid() != expected_parent:
        raise ValueError("GPU supervisor is absent")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # Linux PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot arm worker parent-death protection")
    if os.getppid() != expected_parent:
        raise ValueError("GPU supervisor exited during startup")
    return data, deadline


@contextmanager
def lease_locks(data, directory=None):
    """Prevent our HTTP and standalone entry points from overlapping on GPUs."""
    import fcntl
    directory = Path(directory) if directory else PROJECT_ROOT / "runs/gpu-serving-locks"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(data["active_physical_gpus"]):
            handle = stack.enter_context((directory / f"physical-{gpu}.lock").open("a+"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"physical GPU {gpu} already has a project inference worker; stop it before starting another") from None
        yield


def supervise(script, lease_path, arguments, lock_directory=None):
    data, _ = lease_details(lease_path)
    with lease_locks(data, lock_directory):
        return _supervise(script, lease_path, arguments)


def _supervise(script, lease_path, arguments):
    """Watchdog remains on CPU even while its child is loading model weights."""
    data, deadline = lease_details(lease_path)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=data["cuda_visible_devices"],
               UPCYCLING_GPU_LEASE=str(Path(lease_path).resolve()),
               UPCYCLING_LEASE_SUPERVISOR_PID=str(os.getpid()),
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", str(script), *arguments, "--lease-worker"]
    print(json.dumps({"event": "lease_supervisor", "physical_gpus": data["active_physical_gpus"],
                      "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                      "stop_at": datetime.fromtimestamp(deadline, tz=datetime.fromisoformat(data["expires_at"]).tzinfo).isoformat(),
                      "expires_at": data["expires_at"]}), flush=True)
    child = subprocess.Popen(command, env=env, start_new_session=True)
    interrupted = False

    def stop_requested(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGHUP, stop_requested)
    try:
        while child.poll() is None and time.time() < deadline and not interrupted:
            time.sleep(min(1, max(.01, deadline - time.time())))
        if child.poll() is None:
            print(json.dumps({"event": "stopping_gpu_worker", "pid": child.pid,
                              "reason": "signal" if interrupted else "lease_deadline"}), flush=True)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
