# syntax=docker/dockerfile:1
#
# Images for reproducing the TensorSocket evaluation. One target per software
# stack (the workloads need incompatible framework versions):
#
#   timm    image classification (Figures 8, 9, 10, 13, 14, 15, Table 3), analysis/plots
#   clmr    CLMR audio classification (Figure 11)
#   dalle2  DALL-E 2 diffusion prior (Figure 12)
#   llm     Qwen2.5 fine-tuning with torchtune (Table 4)
#   joader  timm + the Joader server (Figure 15)
#
# Build with ./reproduce.sh build <target>, or:
#   docker build -f docker/Dockerfile --target timm -t tsrepro:timm .

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-base-ubuntu22.04

########################################################################
FROM ${CUDA_IMAGE} AS common

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python-is-python3 \
        git ca-certificates curl procps sysstat \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# DCGM (dcgmi + nv-hostengine) for SM activity, PCIe and NVLink counters. The
# CUDA base image already has NVIDIA's apt repository configured.
RUN apt-get update \
    && (apt-get install -y --no-install-recommends datacenter-gpu-manager \
        || apt-get install -y --no-install-recommends datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12) \
    && rm -rf /var/lib/apt/lists/*

# Harness and plotting dependencies.
RUN pip install \
        "numpy<2" pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.1 \
        psutil==5.9.8 pyyaml==6.0.1 pyzmq==25.1.2 tornado==6.4

# --no-deps: pyzmq and tornado are pinned above, and each image pins its own torch.
RUN pip install --no-deps tensorsocket==0.0.5

ENV PYTHONPATH=/workspace/harness \
    TSREPRO_RESULTS=/workspace/results \
    HF_HOME=/data/hf
WORKDIR /workspace
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

########################################################################
FROM common AS timm

# PIP_CONSTRAINT makes pip fail instead of silently replacing the pinned torch.
RUN printf 'torch==2.1.2\ntorchvision==0.16.2\nnumpy<2\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.2 torchvision==0.16.2 \
    && pip install timm==0.9.12 \
        grpcio==1.64.1 protobuf==5.27.2 lmdb==1.4.1 msgpack==1.0.8 pillow==10.2.0
COPY . /workspace

########################################################################
FROM common AS clmr

RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN printf 'torch==2.0.1\ntorchaudio==2.0.2\nnumpy<2\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 \
    && pip install pytorch-lightning==1.9.5 torchaudio-augmentations==0.2.4 \
        wavaugment==0.2 julius==0.2.7 torch-pitch-shift==1.2.4 simclr==1.0.2 \
        soundfile==0.12.1 scikit-learn==1.3.2 tensorboardX==2.6.2.2
# CLMR's setup.py installs no modules, so the pinned checkout goes on PYTHONPATH.
# Its MagnaTagATune module imports download helpers that torchaudio 2.0 removed;
# that dataset is not used here, so the import is made optional.
RUN git clone https://github.com/Spijkervet/CLMR.git /opt/clmr \
    && git -C /opt/clmr checkout 423414d \
    && python - <<'PY'
from pathlib import Path
p = Path("/opt/clmr/clmr/datasets/magnatagatune.py")
s = p.read_text()
old = "from torchaudio.datasets.utils import (\n    download_url,\n    extract_archive,\n)\n"
assert old in s
p.write_text(s.replace(old, "try:\n    from torchaudio.datasets.utils import download_url, extract_archive\nexcept ImportError:\n    download_url = extract_archive = None\n"))
PY
ENV PYTHONPATH=/opt/clmr:${PYTHONPATH}
COPY . /workspace

########################################################################
FROM common AS dalle2

RUN printf 'torch==2.1.2\ntorchvision==0.16.2\nnumpy<2\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# dalle2-pytorch 1.15.6 (Oct 2023) leaves its dependencies unpinned; they are
# pinned here to releases from the same period.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.2 torchvision==0.16.2 \
    && pip install dalle2-pytorch==1.15.6 webdataset==0.2.86 \
        accelerate==0.25.0 clip-anytorch==2.5.2 coca-pytorch==0.0.12 einops==0.7.0 \
        ema-pytorch==0.3.1 embedding-reader==1.5.1 kornia==0.7.1 open-clip-torch==2.23.0 \
        pydantic==2.5.3 pytorch-warmup==0.1.1 resize-right==0.0.2 rotary-embedding-torch==0.5.3 \
        torchmetrics==1.2.1 vector-quantize-pytorch==1.12.0 x-clip==0.14.4
# img2dataset (CC3M download only) lives in its own environment.
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/img2dataset \
    && PIP_CONSTRAINT= /opt/img2dataset/bin/pip install img2dataset==1.45.0
COPY . /workspace

########################################################################
FROM common AS llm

RUN printf 'torch==2.5.1\nnumpy<2\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 \
    && pip install torchtune==0.4.0 torchao==0.6.1 omegaconf==2.3.0 \
        datasets==3.1.0 huggingface_hub==0.26.2 safetensors==0.4.5 sentencepiece==0.2.0 tiktoken==0.8.0
COPY . /workspace

########################################################################
FROM ubuntu:22.04 AS joader-build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang libclang-dev llvm-dev pkg-config cmake \
        libopencv-dev liblmdb-dev libssl-dev protobuf-compiler \
        git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# Joader pins a nightly toolchain (server/rust-toolchain) and links libtorch 1.11 through tch 0.7.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2021-12-02 --profile minimal --component rustfmt
ENV PATH=/root/.cargo/bin:${PATH}
RUN curl -sSL -o /tmp/libtorch.zip \
        https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip \
    && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV LIBTORCH=/opt/libtorch \
    LD_LIBRARY_PATH=/opt/libtorch/lib

# Joader hard-codes 48 tokio worker threads (server/src/main.rs). Figure 15 gives
# every technique 8 data-loading workers, so the server is built with 8 threads
# (serve.sh additionally pins it to 8 CPUs, which also caps OpenCV/libtorch threads).
ARG JOADER_COMMIT=36beb2d
ARG JOADER_THREADS=8
RUN git clone https://github.com/XieJiann/Joader.git /src/joader \
    && git -C /src/joader checkout ${JOADER_COMMIT} \
    && grep -q "worker_threads = 48" /src/joader/server/src/main.rs \
    && sed -i "s/worker_threads = 48/worker_threads = ${JOADER_THREADS}/" /src/joader/server/src/main.rs \
    && cd /src/joader/server && cargo build --release

########################################################################
FROM timm AS joader

RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev liblmdb0 \
    && rm -rf /var/lib/apt/lists/*
# libtorch 1.11 is only for the Joader server (serve.sh puts it on its library
# path); exporting it image-wide would break the PyTorch 2.1 training processes.
COPY --from=joader-build /src/joader/server /server
COPY --from=joader-build /opt/libtorch/lib /opt/libtorch/lib
