#!/usr/bin/env bash
# Driver for the TensorSocket reproducibility package. See README.md.
#
#   ./reproduce.sh list                         experiments and what they need
#   ./reproduce.sh build <image|all>            build Docker images
#   ./reproduce.sh prepare <dataset>            download / convert datasets into $DATA_DIR
#   ./reproduce.sh smoke                        5-minute end-to-end check on synthetic data
#   ./reproduce.sh run <experiment> [args...]   run one experiment (args go to tsrepro.run;
#                                               --fast: ~10-20 min version, see README)
#   ./reproduce.sh plot [--fast] [<figure>|all] parse results and draw figures/tables
#   ./reproduce.sh run-all-1gpu [--fast]        every experiment that fits one 80 GB GPU:
#                                               the H100 figures, the single-GPU stand-ins for
#                                               the A100 items, and the g5 figures emulated
#   ./reproduce.sh shell [image]                interactive shell in an image
#
# Environment:
#   DATA_DIR   host directory with the datasets (default: ./data), mounted at /data
#   GPUS       physical GPU ids used for logical devices 0,1,2,... (e.g. 0,1,2,4)
#   CPUSET     restrict the container to these host CPUs (docker --cpuset-cpus)
#   INSTANCE   g5.2xlarge | g5.4xlarge | g5.8xlarge for Figures 11 and 13 (results tag)
#   EMULATE    1: emulate INSTANCE's vCPU count on a larger machine via CPUSET
#   TAG        results sub-directory (default: INSTANCE, else "default")
#   DOCKER_GPU_ARGS  how containers get GPUs (default: --gpus all; some setups need
#              "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all")
#   DOCKER_EXTRA_ARGS  additional docker run arguments (e.g. "--memory 120g")
#   DATASET_MOUNTS  extra "host_dir:container_dir" pairs (space-separated) to mount
#              existing datasets read-only, e.g. "/raid/datasets/imagenet:/data/imagenet"
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "${DATA_DIR:-$REPO/data}" 2>/dev/null && pwd || echo "${DATA_DIR:-$REPO/data}")"
mkdir -p "$REPO/results" "$REPO/figures" "$DATA_DIR/cache"

die() { echo "error: $*" >&2; exit 1; }

# experiment -> image | experiment file | needs
experiment_info() {
    case "$1" in
        fig08)        echo "timm|experiments/fig08_a100_4way.csv|A100 server: 4 GPUs with NVLink, 48 cores; ImageNet" ;;
        fig09)        echo "timm|experiments/fig09_a100_collocation.csv|A100 server; ImageNet; 4x points come from fig08" ;;
        tab03)        echo "timm|experiments/tab03_a100_data_movement.csv|A100 server; ImageNet on local disk" ;;
        fig10)        echo "timm|experiments/fig10_h100_flexible.csv|H100 server; ImageNet" ;;
        fig11)        echo "clmr|experiments/fig11_g5_clmr.csv|AWS g5.{2,4,8}xlarge (set INSTANCE); LibriSpeech" ;;
        fig12)        echo "dalle2|experiments/fig12_h100_dalle2.csv|H100 server; CC3M (100k samples)" ;;
        fig13)        echo "timm|experiments/fig13_g5_model_selection.csv|AWS g5.{2,4,8}xlarge (set INSTANCE); ImageNet 10%" ;;
        tab04)        echo "llm|experiments/tab04_a100_llm.csv|A100 server (3 GPUs); Qwen2.5-0.5B + Alpaca" ;;
        fig14)        echo "timm|experiments/fig14_a100_coordl_ts.csv|A100 server; ImageNet (baseline + TensorSocket)" ;;
        fig14-coordl) echo "coordl|experiments/fig14_a100_coordl.sh|A100 server (one job per GPU); ImageNet (train/ and val/)" ;;
        fig14-coordl-1gpu) echo "coordl|experiments/fig14_a100_coordl.sh|single-GPU stand-in for the CoorDL side of fig14; ImageNet (train/ and val/)" ;;
        fig15)        echo "timm|experiments/fig15_h100_joader_ts.csv|H100 server; ImageNet (baseline + TensorSocket)" ;;
        fig15-joader) echo "joader|experiments/fig15_h100_joader.csv|H100 server; ImageNet + Joader LMDB" ;;
        fig08-1gpu)   echo "timm|experiments/fig08_h100_1gpu.csv|single-GPU stand-in for fig08 (one GPU, MPS); ImageNet" ;;
        fig09-1gpu)   echo "timm|experiments/fig09_h100_1gpu.csv|single-GPU stand-in for fig09; ImageNet; 4x points from fig08-1gpu" ;;
        tab03-1gpu)   echo "timm|experiments/tab03_h100_1gpu.csv|single-GPU stand-in for tab03 (no NVLink column); ImageNet" ;;
        tab04-1gpu)   echo "llm|experiments/tab04_h100_1gpu.csv|single-GPU stand-in for tab04; Qwen2.5-0.5B + Alpaca" ;;
        fig14-1gpu)   echo "timm|experiments/fig14_h100_1gpu.csv|single-GPU stand-in for fig14 (baseline + TensorSocket); ImageNet" ;;
        smoke)        echo "timm|experiments/smoke.csv|any NVIDIA GPU; synthetic data" ;;
        *) return 1 ;;
    esac
}
EXPERIMENTS="fig08 fig09 tab03 fig10 fig11 fig12 fig13 tab04 fig14 fig14-coordl fig15 fig15-joader fig08-1gpu fig09-1gpu tab03-1gpu tab04-1gpu fig14-1gpu fig14-coordl-1gpu"

instance_cpus() {
    case "$1" in
        g5.2xlarge) echo 8 ;; g5.4xlarge) echo 16 ;; g5.8xlarge) echo 32 ;;
        *) die "unknown INSTANCE '$1' (g5.2xlarge, g5.4xlarge or g5.8xlarge)" ;;
    esac
}

docker_run() {
    # docker_run <image> [docker args...] -- <command...>
    local image="$1"; shift
    local extra=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
    shift
    local tty=()
    [ -t 0 ] && [ -t 1 ] && tty=(-it)
    [ -n "${CPUSET:-}" ] && extra+=(--cpuset-cpus "$CPUSET")
    local m
    for m in ${DATASET_MOUNTS:-}; do extra+=(-v "$m:ro"); done
    # shellcheck disable=SC2206
    [ -n "${DOCKER_EXTRA_ARGS:-}" ] && extra+=(${DOCKER_EXTRA_ARGS})
    # shellcheck disable=SC2086
    docker run --rm ${tty[@]+"${tty[@]}"} \
        ${DOCKER_GPU_ARGS:---gpus all} --ipc=host --cap-add SYS_ADMIN \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$REPO":/workspace -v "$DATA_DIR":/data -v "$DATA_DIR/cache":/root/.cache \
        -e IMAGENET=/data/imagenet \
        -e IMAGENET_10PCT=/data/imagenet_10pct \
        -e IMAGENET_JOADER=/data/imagenet_joader/ILSVRC-train.lmdb \
        -e LIBRISPEECH=/data/librispeech \
        -e CC3M=/data/cc3m \
        -e SMOKE_DATA=/data/synthetic_imagenet \
        -e HF_HOME=/data/hf \
        -e TAG="${TAG:-}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        ${extra[@]+"${extra[@]}"} "tsrepro:$image" "$@"
}

cmd_list() {
    printf "%-18s %-7s %s\n" EXPERIMENT IMAGE "NEEDS"
    for e in $EXPERIMENTS; do
        IFS='|' read -r image file needs <<<"$(experiment_info "$e")"
        printf "%-18s %-7s %s\n" "$e" "$image" "$needs"
    done
}

cmd_build() {
    local target="${1:-all}"
    if [ "$target" = all ]; then
        for t in timm clmr dalle2 llm joader; do cmd_build "$t"; done
        echo "note: the CoorDL image is built separately: ./reproduce.sh build coordl"
        return
    fi
    if [ "$target" = coordl ]; then
        "$REPO/docker/coordl/build.sh"
        return
    fi
    docker build -f "$REPO/docker/Dockerfile" --target "$target" -t "tsrepro:$target" "$REPO"
}

cmd_prepare() {
    local what="${1:-}"
    case "$what" in
        synthetic)
            docker_run timm -- python datasets/make_synthetic_imagenet.py /data/synthetic_imagenet ;;
        imagenet)
            [ -d "$DATA_DIR/imagenet/train" ] || die "place the ImageNet-1k train set in $DATA_DIR/imagenet/train (see README, Datasets)"
            echo "found $(find "$DATA_DIR/imagenet/train" -maxdepth 1 -mindepth 1 -type d | wc -l) classes in $DATA_DIR/imagenet/train" ;;
        imagenet-10pct)
            docker_run timm -- python datasets/make_imagenet_subset.py /data/imagenet /data/imagenet_10pct ;;
        imagenet-joader)
            docker_run timm -- python workloads/joader/convert_imagenet.py /data/imagenet/train /data/imagenet_joader/ILSVRC-train.lmdb ;;
        librispeech)
            docker_run clmr -- python datasets/prepare_librispeech.py /data/librispeech ;;
        cc3m)
            docker_run dalle2 -- bash datasets/prepare_cc3m.sh /data/cc3m_raw/Train_GCC-training.tsv /data/cc3m ;;
        qwen)
            docker_run llm -- bash datasets/prepare_qwen.sh ;;
        *) die "usage: reproduce.sh prepare <synthetic|imagenet|imagenet-10pct|imagenet-joader|librispeech|cc3m|qwen>" ;;
    esac
}

cmd_run() {
    local exp="${1:-}"; shift || true
    local info
    info="$(experiment_info "$exp")" || die "unknown experiment '$exp' (see ./reproduce.sh list)"
    IFS='|' read -r image file needs <<<"$info"
    local extra=()

    if [ -n "${INSTANCE:-}" ]; then
        TAG="${TAG:-$INSTANCE}"
        if [ "${EMULATE:-0}" = 1 ]; then
            CPUSET="0-$(( $(instance_cpus "$INSTANCE") - 1 ))"
        fi
    elif [ "$exp" = fig11 ] || [ "$exp" = fig13 ]; then
        die "$exp runs per AWS instance size: set INSTANCE=g5.2xlarge|g5.4xlarge|g5.8xlarge (add EMULATE=1 to emulate the vCPU count)"
    fi
    TAG="${TAG:-default}"

    case "$exp" in
        fig08|fig09|tab03|fig14|fig14-coordl|tab04)
            # The paper restricts the A100 server to 48 cores (Table 2).
            if [ -z "${CPUSET:-}" ] && [ "$(nproc)" -gt 48 ]; then CPUSET="0-47"; fi ;;
    esac
    if [ "$exp" = tab03 ] || [ "$exp" = tab03-1gpu ]; then
        # Cap the container's page cache so ImageNet cannot stay cached.
        extra+=(--memory "${TAB03_MEMORY:-96g}")
    fi

    local gpus=()
    [ -n "${GPUS:-}" ] && gpus=(--gpus "$GPUS")
    echo "experiment $exp: $needs"
    echo "image tsrepro:$image, tag $TAG${CPUSET:+, cpuset $CPUSET}, data $DATA_DIR"

    case "$exp" in
        fig14-coordl|fig14-coordl-1gpu)
            local variant=multi results=/workspace/results fast=()
            [ "$exp" = fig14-coordl-1gpu ] && variant=1gpu
            if [ "${1:-}" = --fast ]; then
                results=/workspace/results_fast
                fast=(-e FAST_SECONDS="${FAST_SECONDS:-300}")
            fi
            docker_run "$image" ${extra[@]+"${extra[@]}"} -e TAG="$TAG" -e TSREPRO_RESULTS="$results" \
                -e COORDL_VARIANT="$variant" -e EPOCHS="${EPOCHS:-3}" ${fast[@]+"${fast[@]}"} -- \
                bash /workspace/experiments/fig14_a100_coordl.sh ;;
        tab03|tab03-1gpu)
            docker_run "$image" ${extra[@]+"${extra[@]}"} -- \
                python -m tsrepro.run "$file" --tag "$TAG" --evict "{IMAGENET}" ${gpus[@]+"${gpus[@]}"} "$@" ;;
        *)
            docker_run "$image" ${extra[@]+"${extra[@]}"} -- \
                python -m tsrepro.run "$file" --tag "$TAG" ${gpus[@]+"${gpus[@]}"} "$@" ;;
    esac
}

cmd_smoke() {
    [ -d "$DATA_DIR/synthetic_imagenet/train" ] || cmd_prepare synthetic
    TAG="${TAG:-smoke}" cmd_run smoke --force "$@"
    docker_run timm -- python -m tsrepro.plot smoke
}

cmd_plot() {
    if docker image inspect tsrepro:timm >/dev/null 2>&1; then
        docker run --rm -v "$REPO":/workspace -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
            --entrypoint python tsrepro:timm -m tsrepro.plot "$@"
    else
        PYTHONPATH="$REPO/harness" python3 -m tsrepro.plot "$@"
    fi
}

cmd_run_all_1gpu() {
    local failed=()
    local e i
    for e in fig10 fig12 fig15 fig15-joader fig08-1gpu fig09-1gpu fig14-1gpu fig14-coordl-1gpu tab03-1gpu tab04-1gpu; do
        ( cmd_run "$e" "$@" ) || failed+=("$e")
    done
    for i in g5.2xlarge g5.4xlarge g5.8xlarge; do
        ( INSTANCE=$i EMULATE=1 cmd_run fig13 "$@" ) || failed+=("fig13/$i")
        ( INSTANCE=$i EMULATE=1 cmd_run fig11 "$@" --workloads 4,104,204,304 ) || failed+=("fig11/$i")
    done
    if [ "${1:-}" = --fast ]; then cmd_plot --fast; else cmd_plot; fi
    if [ ${#failed[@]} -gt 0 ]; then
        echo "experiments with failed workloads: ${failed[*]}" >&2
        return 1
    fi
}

cmd_shell() {
    docker_run "${1:-timm}" -- bash
}

main() {
    local cmd="${1:-}"; shift || true
    case "$cmd" in
        list) cmd_list ;;
        build) cmd_build "$@" ;;
        prepare) cmd_prepare "$@" ;;
        run) cmd_run "$@" ;;
        smoke) cmd_smoke "$@" ;;
        run-all-1gpu) cmd_run_all_1gpu "$@" ;;
        plot) cmd_plot "$@" ;;
        shell) cmd_shell "$@" ;;
        *) sed -n '2,20p' "$0"; exit 1 ;;
    esac
}

main "$@"
