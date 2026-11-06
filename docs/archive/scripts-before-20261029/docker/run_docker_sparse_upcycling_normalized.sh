#!/usr/bin/env bash
# Physical GPUs 6 and 7; authorized for the normalized-routing comparison.
set -euo pipefail
name=wrd-sparse-upcycling-normalized
if docker container inspect "$name" >/dev/null 2>&1; then
  mapped=$(docker inspect "$name" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}')
  if [[ "$mapped" != "/dev/iluvatar6 /dev/iluvatar7 " ]]; then
    echo "Existing $name has unexpected GPU mapping: $mapped" >&2
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$name"; then
    docker start "$name" >/dev/null
  fi
else
  docker run -d --shm-size=32g -h workspace --name "$name" \
    -w /home/wurendi/projects/sparse_upcycling_qwen3 \
    -v /home/wurendi:/home/wurendi \
    -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    --device=/dev/iluvatar6 --device=/dev/iluvatar7 \
    --cap-add=ALL --pid=host corex:4.3.6 sleep infinity >/dev/null
fi
echo "$name: physical GPUs 6,7"
