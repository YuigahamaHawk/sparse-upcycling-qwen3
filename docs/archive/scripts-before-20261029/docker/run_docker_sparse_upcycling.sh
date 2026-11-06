#!/usr/bin/env bash
set -euo pipefail
name=wrd-sparse-upcycling
if docker container inspect "$name" >/dev/null 2>&1; then
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$name"; then
    docker start "$name" >/dev/null
  fi
else
  docker run -d --shm-size=32g -h workspace --name "$name" \
    -w /home/wurendi/projects/sparse_upcycling_qwen3 \
    -v /home/wurendi:/home/wurendi \
    -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    --device=/dev/iluvatar4 --device=/dev/iluvatar5 \
    --device=/dev/iluvatar6 --device=/dev/iluvatar7 \
    --device=/dev/iluvatar8 --device=/dev/iluvatar9 \
    --device=/dev/iluvatar10 --device=/dev/iluvatar11 \
    --device=/dev/iluvatar12 --device=/dev/iluvatar13 \
    --device=/dev/iluvatar14 --device=/dev/iluvatar15 \
    --cap-add=ALL --pid=host corex:4.3.6 sleep infinity >/dev/null
fi
docker exec -it -w /home/wurendi/projects/sparse_upcycling_qwen3 "$name" bash
