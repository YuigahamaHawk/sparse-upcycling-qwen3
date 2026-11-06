#!/usr/bin/env bash
# INSIDE a user-started container, with an explicitly approved CURRENT lease.
# All settings, including --lease and --api-key-file, are supplied by the caller.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
exec python3 -u scripts/deployment/serve_models.py "$@"
