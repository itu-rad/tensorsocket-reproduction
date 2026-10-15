#!/usr/bin/env bash
# Figure 14 (CoorDL): 1-4 ResNet18 jobs with CoorDL's coordinated data
# preparation, using the hyper-parameter-search script from DS-Analyzer
# (CoorDL's evaluation script). Settings match fig14_a100_coordl_ts.csv:
# batch size 512, 4 DALI CPU threads, no AMP, 3 epochs.
#
# COORDL_VARIANT=multi (default): one job per GPU, as on the paper's A100 server.
# COORDL_VARIANT=1gpu: all jobs share GPU 0 (stand-in on single-GPU machines,
#   matching fig14_h100_1gpu.csv).
# Runs inside tsrepro:coordl (reproduce.sh run fig14-coordl / fig14-coordl-1gpu);
# writes results/<experiment>/<tag>/n<N>/rep<r>/{log.txt,top.txt,meta.json}.
# Throughput is CoorDL's "##Perf" line (images/s of the first job, last epoch);
# CPU utilisation is the summed %CPU of top snapshots.
# FAST_SECONDS=N (reproduce.sh --fast): stop each run after N seconds; the
# analysis then uses the job's running average speed instead of ##Perf.
set -uo pipefail
# Unbuffered, so progress (and the fast-mode clock) sees log lines as they happen.
export PYTHONUNBUFFERED=1

IMAGENET="${IMAGENET:-/data/imagenet}"
TAG="${TAG:-default}"
REPEATS="${REPEATS:-1}"
EPOCHS="${EPOCHS:-3}"
JOBS="${JOBS:-1 2 3 4}"
VARIANT="${COORDL_VARIANT:-multi}"
case "$VARIANT" in
    multi) SCRIPT=hyperparam_imagenet_dali_mp.py; EXP=fig14_a100_coordl ;;
    1gpu)  SCRIPT=hyperparam_imagenet_dali_mp_1gpu.py; EXP=fig14_h100_1gpu_coordl ;;
    *) echo "unknown COORDL_VARIANT=$VARIANT" >&2; exit 2 ;;
esac
OUT_ROOT="${TSREPRO_RESULTS:-/workspace/results}/$EXP/$TAG"

for rep in $(seq 0 $((REPEATS - 1))); do
    for n in $JOBS; do
        out="$OUT_ROOT/n$n/rep$rep"
        if [ -f "$out/done" ]; then echo "skip $out"; continue; fi
        mkdir -p "$out"
        echo "=== CoorDL ($VARIANT): $n job(s) -> $out"
        top -b -d 1 > "$out/top.txt" &
        top_pid=$!
        start=$(date +%s)
        # A single job has nothing to coordinate, and the coordinated (--unified)
        # path stalls in its set-up with one job; it then runs CoorDL's DALI pipeline alone.
        coordination=(--unified --pin)
        [ "$n" -eq 1 ] && coordination=()
        python -O "$SCRIPT" -a resnet18 -j 4 -b 512 --epochs "$EPOCHS" \
            --dist --num_jobs "$n" --dali_cpu "${coordination[@]}" --noeval "$IMAGENET" > "$out/log.txt" 2>&1 &
        job=$!
        if [ -n "${FAST_SECONDS:-}" ]; then
            # Count FAST_SECONDS from the first logged iteration: start-up (file
            # indexing, CUDA JIT on new GPUs) varies a lot between runs.
            waited=0
            until grep -q "Speed" "$out/log.txt" || ! kill -0 "$job" 2>/dev/null || [ "$waited" -ge 1800 ]; do
                sleep 5; waited=$((waited + 5))
            done
            elapsed=0
            while kill -0 "$job" 2>/dev/null && [ "$elapsed" -lt "$FAST_SECONDS" ]; do
                sleep 5; elapsed=$((elapsed + 5))
            done
            kill "$job" 2>/dev/null
        fi
        wait "$job"
        status=$?
        kill "$top_pid" 2>/dev/null
        echo "{\"jobs\": $n, \"variant\": \"$VARIANT\", \"status\": $status, \"seconds\": $(( $(date +%s) - start )), \"ncpu\": $(nproc)}" > "$out/meta.json"
        grep "##Perf" "$out/log.txt" || tail -20 "$out/log.txt"
        # In fast mode the run is stopped on purpose; it counts if it logged iterations.
        if [ "$status" -eq 0 ] || { [ -n "${FAST_SECONDS:-}" ] && grep -q "Speed" "$out/log.txt"; }; then
            touch "$out/done"
        fi
        # Jobs of a stopped run can outlive the parent; clear them before the next run.
        pkill -f "$SCRIPT" 2>/dev/null; sleep 5
        [ -n "${HOST_UID:-}" ] && chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$OUT_ROOT"
    done
done
