#!/usr/bin/env bash
# Start a DCGM host engine inside the container so that `dcgmi dmon` can sample
# SM activity and PCIe/NVLink traffic. Profiling fields need CAP_SYS_ADMIN
# (reproduce.sh passes --cap-add SYS_ADMIN). If the host already runs
# nv-hostengine, or DCGM is unavailable, the harness falls back to nvidia-smi.
if command -v nv-hostengine >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
    if ! dcgmi discovery -l >/dev/null 2>&1; then
        nv-hostengine --log-filename /tmp/nv-hostengine.log >/dev/null 2>&1 || true
        sleep 1
    fi
fi
exec "$@"
