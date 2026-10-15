#!/usr/bin/env bash
# Start the Joader server, register the ImageNet LMDB as dataset "ImageNet"
# and keep running until the harness terminates the process group.
#   serve.sh /data/imagenet_joader/ILSVRC-train.lmdb [cpus]
# The server is pinned to <cpus> CPUs (default 8): the data-loading budget that
# the baseline and TensorSocket get as worker processes in Figure 15.
set -euo pipefail

LMDB="${1:?usage: serve.sh <ILSVRC-train.lmdb> [cpus]}"
CPUS="${2:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN="$(python3 -c "import os, sys; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:int(sys.argv[1])])))" "$CPUS")"

# Joader reads its log config from the absolute path /server/log4rs.yaml and
# writes logs relative to the working directory.
cd /server
echo "joader: pinned to CPUs $PIN"
LD_LIBRARY_PATH=/opt/libtorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} taskset -c "$PIN" ./target/release/joader &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT TERM INT

for _ in $(seq 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/4321) 2>/dev/null; then break; fi
    sleep 1
done

python3 "$HERE/register.py" "$LMDB"
echo "joader: serving $LMDB on 127.0.0.1:4321"
wait $SERVER
