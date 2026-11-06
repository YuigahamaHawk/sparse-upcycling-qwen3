"""CPU-only HTTP client: generate, load a registered model, or inspect status."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:18082")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--timeout", type=float, default=900)
    command = p.add_mutually_exclusive_group(required=True)
    command.add_argument("--instruction")
    command.add_argument("--prompt")
    command.add_argument("--load", metavar="MODEL_ID")
    command.add_argument("--status", action="store_true")
    command.add_argument("--models", action="store_true")
    command.add_argument("--unload", action="store_true")
    p.add_argument("--input", default="")
    p.add_argument("--model")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    body = None
    if args.status:
        endpoint = "/readyz"
    elif args.models:
        endpoint = "/v1/models"
    elif args.load:
        endpoint, body = "/admin/load", {"model": args.load}
    elif args.unload:
        endpoint, body = "/admin/unload", {}
    else:
        endpoint = "/generate"
        body = {"model": args.model, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "seed": args.seed}
        body.update({"prompt": args.prompt} if args.prompt is not None else
                    {"instruction": args.instruction, "input": args.input})
    headers = {"Content-Type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = "Bearer " + args.api_key_file.read_text().strip()
    request = urllib.request.Request(args.url.rstrip("/") + endpoint,
              data=json.dumps(body).encode() if body is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from None


if __name__ == "__main__":
    main()
