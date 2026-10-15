# CoorDL (a DALI 0.20 fork with coordinated data loading) plus DS-Analyzer's
# hyper-parameter-search script: the CoorDL side of Figure 14.
#
# The environment mirrors the one CoorDL was run with for the paper:
# Python 3.6, PyTorch 1.8.1 (CUDA 11.1, cuDNN 8.0.5), torchvision 0.9.1, apex
# (ptrblck/apex apex_no_distributed, Python only) and the CoorDL DALI wheel, which docker/coordl/build.sh compiles
# with CoorDL's own (CUDA 10) build scripts and places next to this file.
#
# Two variants of the evaluation script are installed:
#   hyperparam_imagenet_dali_mp.py       one job per GPU (the A100 setup)
#   hyperparam_imagenet_dali_mp_1gpu.py  all jobs share GPU 0 (single-GPU setup)
# devel: apex's setup.py queries nvcc for the CUDA version even for its Python-only install.
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git procps bzip2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# coordl-env.lock is `conda list --explicit` of the environment CoorDL ran in on
# the paper's servers (exact package builds: Python 3.6.13, PyTorch 1.8.1 CUDA 11.1
# cuDNN 8.0.5, torchvision 0.9.1, CUDA toolkit 11.1.74). micromamba installs it
# without a solver run.
ARG MICROMAMBA_VERSION=1.5.8
COPY coordl-env.lock /tmp/coordl-env.lock
RUN curl -sSL "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
        | tar -xj -C /usr/local bin/micromamba \
    && micromamba create -y -p /opt/conda/envs/coordl -f /tmp/coordl-env.lock \
    && micromamba clean -afy
ENV PATH=/opt/conda/envs/coordl/bin:${PATH}

# The apex used with CoorDL: ptrblck's apex_no_distributed branch (2019), Python only.
ARG APEX_COMMIT=73f3a18
RUN git clone --branch apex_no_distributed https://github.com/ptrblck/apex.git /opt/apex \
    && git -C /opt/apex checkout ${APEX_COMMIT} \
    && cd /opt/apex && pip install --no-deps --no-build-isolation .

COPY nvidia_dali-*.whl /opt/dali/
RUN pip install --no-deps /opt/dali/nvidia_dali-*.whl future==1.0.0
# CoorDL's PyTorch plugin imports its helper module as a top-level `batch_util`,
# which does not resolve under Python 3; import it from the plugin package.
RUN f=$(python -c "import os, nvidia.dali.plugin as p; print(os.path.join(os.path.dirname(p.__file__), 'pytorch_coordl.py'))") \
    && sed -i -E 's/^import (nvidia\.dali\.plugin\.)?batch_util$/from nvidia.dali.plugin import batch_util/' "$f" \
    && grep -q "^from nvidia.dali.plugin import batch_util$" "$f"

ARG DSANALYZER_COMMIT=dbdff25
COPY patches/ds-analyzer-coordl.patch patches/ds-analyzer-single-gpu.patch /tmp/
RUN git clone https://github.com/msr-fiddle/DS-Analyzer.git /opt/DS-Analyzer \
    && cd /opt/DS-Analyzer && git checkout ${DSANALYZER_COMMIT} \
    && git apply /tmp/ds-analyzer-coordl.patch \
    && cp tasks/image-classification/hyperparam_imagenet_dali_mp.py /tmp/multi.py \
    && git apply /tmp/ds-analyzer-single-gpu.patch \
    && mv tasks/image-classification/hyperparam_imagenet_dali_mp.py tasks/image-classification/hyperparam_imagenet_dali_mp_1gpu.py \
    && mv /tmp/multi.py tasks/image-classification/hyperparam_imagenet_dali_mp.py

# Neither PyTorch 1.8.1 nor the CUDA 10 DALI build ships kernels for recent GPUs
# (e.g. Hopper); the driver JIT-compiles them from PTX on first use. Keep the
# compiled kernels (several GB) in the mounted cache so this happens only once.
ENV CUDA_CACHE_MAXSIZE=4294967296 \
    CUDA_CACHE_PATH=/root/.cache/nv

WORKDIR /opt/DS-Analyzer/tasks/image-classification
