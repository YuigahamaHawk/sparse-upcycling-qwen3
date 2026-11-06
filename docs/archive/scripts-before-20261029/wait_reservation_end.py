"""CPU-only expiry signal for the host's stop-only container lifecycle wrapper."""
import json
import time
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
approval = json.loads((root / "runs/alignment-20261020/authorization.json").read_text())
expires = datetime.fromisoformat(approval["expires_at"]).timestamp()
release = expires - 120  # 15:58, two minutes before the reservation expires.
if not datetime.fromisoformat(approval["starts_at"]).timestamp() <= time.time() < expires:
    raise SystemExit("Outside the recorded reservation; refusing to arm a stale stop guard")
print("Stop-only guard armed for " + datetime.fromtimestamp(release).astimezone().isoformat(), flush=True)
while time.time() < release:
    time.sleep(min(30, max(0.1, release - time.time())))
status = {"action": "stop_original_container", "container": approval["container"],
          "time_utc": datetime.now().astimezone().isoformat(), "registered_end": approval["expires_at"]}
(root / "runs/alignment-20261020/reservation_release.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
