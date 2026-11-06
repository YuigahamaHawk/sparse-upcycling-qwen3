#!/usr/bin/env bash
# Invoke only after the user approves physical GPUs 4 and 5 for this run.
set -euo pipefail
name=wrd-sparse-upcycling-2gpu
if docker container inspect "$name" >/dev/null 2>&1; then
  mapped=$(docker inspect "$name" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}')
  if [[ "$mapped" != "/dev/iluvatar4 /dev/iluvatar5 " ]]; then
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
    --device=/dev/iluvatar4 --device=/dev/iluvatar5 \
    --cap-add=ALL --pid=host corex:4.3.6 sleep infinity >/dev/null
fi
echo "$name: physical GPUs 4,5"
