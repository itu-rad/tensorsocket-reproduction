# TensorSocket: reproducibility package

This repository reproduces every experiment in

> Ties Robroek, Neil Kim Nielsen, Pınar Tözün. **TensorSocket: Shared Data Loading for Deep Learning Training.** SIGMOD 2026. [doi:10.1145/3749185](https://doi.org/10.1145/3749185) ([arXiv:2409.18749](https://arxiv.org/abs/2409.18749))

TensorSocket itself lives at <https://github.com/itu-rad/tensorsocket>. This package pins it, includes the training scripts used in the paper (`workloads/`), wraps every experiment in Docker, and provides one command per figure or table to run it, collect and parse its data, and plot it.

The paper's experiments were scheduled and tracked with [radT](https://github.com/itu-rad/radt), our MLflow-based tracking platform. This package does not use radT. Reproducing the paper also means running older codebases with their own pinned environments, such as CoorDL (Python 3.6, CUDA 10) and Joader, and setting up radT and an MLflow server around those would add work for every reviewer. Instead, a small custom wrapper based on radT (`harness/tsrepro`) runs each experiment in Docker and records the metrics the paper uses. It writes them in the MLflow export format with radT's metric names (`ML - Iterations/s`, `ML - Epoch`, ...), so the paper's analysis logic applies unchanged. The workload scripts still log through radT when they are run under it.

The original paper and its experiments were done without AI agents. This reproducibility package was extracted from the original training scripts and built with the help of AI coding agents.

Contents:

- [TensorSocket: reproducibility package](#tensorsocket-reproducibility-package)
  - [1. What is reproduced](#1-what-is-reproduced)
    - [1.1 Validation of this package](#11-validation-of-this-package)
  - [2. Hardware](#2-hardware)
    - [2.1 Machines used in the paper (Table 2)](#21-machines-used-in-the-paper-table-2)
    - [2.2 Minimum hardware to reproduce](#22-minimum-hardware-to-reproduce)
  - [3. Software and setup](#3-software-and-setup)
    - [3.1 Host prerequisites](#31-host-prerequisites)
    - [3.2 Images](#32-images)
    - [3.3 Build and smoke test](#33-build-and-smoke-test)
  - [4. Datasets](#4-datasets)
  - [5. Running the experiments](#5-running-the-experiments)
    - [Fast mode](#fast-mode)
    - [5.1 A100 server (Figures 8, 9, 14; Tables 3, 4)](#51-a100-server-figures-8-9-14-tables-3-4)
    - [5.2 H100 server (Figures 10, 12, 15)](#52-h100-server-figures-10-12-15)
    - [5.3 AWS g5 instances (Figures 11, 13)](#53-aws-g5-instances-figures-11-13)
    - [5.4 Emulating the g5 instances](#54-emulating-the-g5-instances)
  - [6. Collecting, parsing and plotting results](#6-collecting-parsing-and-plotting-results)
  - [7. Metrics](#7-metrics)
  - [8. Reproducibility of the paper's results and changes from the original runs](#8-reproducibility-of-the-papers-results-and-changes-from-the-original-runs)
    - [8.1 What can be reproduced (guideline C)](#81-what-can-be-reproduced-guideline-c)
    - [8.2 CoorDL (Figure 14)](#82-coordl-figure-14)
    - [8.3 Joader (Figure 15)](#83-joader-figure-15)
  - [9. Repository layout](#9-repository-layout)
  - [10. Troubleshooting](#10-troubleshooting)

---

## 1. What is reproduced

| Paper item | Claim it supports | Experiment id | Hardware | Data | Approx. time¹ |
|---|---|---|---|---|---|
| Figure 1 | Cloud vCPU:GPU ratios are low | `fig01` (plot only) | none | survey data in `paper_results/` | seconds |
| Figure 8 | Sharing raises throughput and GPU use and cuts CPU use (4 GPUs, 4 models) | `fig08` | A100 server | ImageNet | 6 h |
| Figure 9 | Shared loading keeps per-model throughput as collocation grows | `fig09` (+ `fig08`) | A100 server | ImageNet | 5 h |
| Table 3 | Sharing replaces disk and PCIe traffic with NVLink | `tab03` | A100 server | ImageNet | 1.5 h |
| Figure 10 | Flexible batch sizes cost no throughput and little CPU | `fig10` | H100 server | ImageNet | 1 h |
| Figure 11 | Sharing removes the need for large cloud instances (CLMR) | `fig11` | AWS g5.2x/4x/8xlarge | LibriSpeech | 3–5 h per instance |
| Figure 12 | Sharing GPU work (CLIP) for DALL-E 2 | `fig12` | H100 server | CC3M (100k) | 2.5 h |
| Figure 13 | Mixed models (model selection) on small cloud instances | `fig13` | AWS g5.2x/4x/8xlarge | ImageNet 10% | 1 h per instance |
| Table 4 | No overhead for GPU-bound LLM fine-tuning | `tab04` | A100 server | Qwen2.5-0.5B, Alpaca | 1.5 h |
| Figure 14 | Comparison with CoorDL | `fig14`, `fig14-coordl` | A100 server | ImageNet | 6 h + 2 h |
| Figure 15 | Comparison with Joader | `fig15`, `fig15-joader` | H100 server | ImageNet | 6 h + 3 h |

¹ One repetition, from the epoch times of the original runs. The paper ran every experiment twice (`--repeats 2`).

All experiments are scripted. The only manual steps are downloading the license-restricted datasets: ImageNet, and the CC3M URL list. See Section 4.

**Hardware dependence.** These are performance experiments, and absolute numbers depend on the machine. The paper's claims are relative: shared versus non-shared on the same hardware, or how that relation changes with the vCPU budget. Every plot draws the paper's values as black ticks or dotted lines next to the reproduced ones. Every table also lists `reproduced / paper` ratios, so the trends can be compared even on different hardware. Section 8.1 lists the parts that cannot be fully reproduced (CoorDL, and to a lesser extent Joader) and explains why the core thesis does not depend on them.

### 1.1 Validation of this package

We ran this package end to end on the paper's H100 server: 1× H100 PCIe 80 GB, 2× Xeon Silver 4410Y (24 cores / 48 threads) and 219 GiB RAM. We used `./reproduce.sh run-all-1gpu --fast`, which completes in about 4 hours. Fig. 12 and Fig. 15 (baseline and TensorSocket) were also run in full mode.

* The 4-GPU A100 server is no longer available. Its experiments therefore ran as single-GPU stand-ins (`*-1gpu`), with all models sharing the H100 via MPS.
* The AWS g5 instances were emulated by restricting the container to 8, 16 and 32 CPUs.
* All figures and tables of these runs are in `validation/h100_xena2_fast/` and `validation/h100_xena2_full/` (`REPORT.md` has every number next to the paper's).

| Item | Paper | Validation run (H100 server) |
|---|---|---|
| Fig. 10 flexible batching | 2,328 vs 2,332 samples/s; CPU 45 → 47% | 2,309 vs 2,218 samples/s; CPU 50 → 57% |
| Fig. 12 DALL-E 2, sharing gain at 2/4-way | 10–15% | +7% / +10% (full mode) |
| Fig. 15 TensorSocket per model, 1 → 4 / 8 models | 1,142 → 1,113 / 965 | 1,599 → 1,555 / 879 (full mode) |
| Fig. 15 baseline per model, 1 → 8 models | 1,128 → 138 | 1,561 → 209 (full mode) |
| Fig. 15 Joader per model, 1 → 4 / 8 models (8 CPUs, like the others) | 984 → 438 / 287 | 1,324 → 614 / 331 (fast mode) |
| Fig. 11 CLMR 4-way, shared at 8 vCPUs vs non-shared at 32 | equal (~50–59 samples/s) | 117–126 vs 116–118 samples/s |
| Fig. 13 aggregate, shared g5.2xlarge vs non-shared g5.4xlarge | 1,365 vs 1,548 samples/s | 1,840 vs 2,137 samples/s |
| Fig. 14 stand-in, per-model throughput at 2/3/4 models (baseline; TensorSocket) | 0.51/0.35/0.29; ≈1.0 | 0.46/0.29/0.21; 1.00/0.94/0.95 |
| Fig. 14 stand-in, CoorDL per-job throughput at 2/3/4 jobs (`fig14-coordl-1gpu`) | ≈1.0 (one job per GPU) | 0.83/0.54/0.38 (all jobs on one GPU) |
| Fig. 8 stand-in, CPU utilization non-shared → shared | ~3× lower (e.g. ResNet18 99 → 30%) | 2–3× lower (e.g. ResNet18 45 → 22%) |
| Table 3 stand-in, disk I/O non-shared → shared | 613 → 161 MB/s | 269 → 76 MB/s |
| Table 4 stand-in, tokens/s per model, baseline vs shared | 7.5k vs 7.5k | 4.9k vs 4.9k |

Notes:

* **Absolute numbers.** The H100 server's newer CPUs make most absolute numbers higher than in the paper. The relative effects match.
* **Joader**, restricted to the same 8-CPU data-loading budget as the other techniques, again lies between the baseline and TensorSocket. Its values are about 1.15–1.4× the paper's; the baseline's are 1.4–1.5×.
* **The single-GPU stand-ins** saturate the one GPU (SM activity ≈ 97%). They therefore show the CPU-side effect of sharing, and the throughput effect only for the most CPU-bound models (MobileNet S: +33%). The throughput gains of Fig. 8 for every model need one GPU per model.
* **PCIe counters** for the single-GPU Table 3 stand-in are not comparable to the paper's per-GPU A100 values.
* **CoorDL** runs in the `coordl` image, which reproduces the environment CoorDL was run in for the paper.
  * As in the paper, throughput is normalised per method to its own 1-job value. The raw images/s are in the `raw` column of `fig14_h100_1gpu.csv`.
  * With all jobs on one GPU, CoorDL's aggregate throughput plateaus at about 2,200–2,400 images/s from 2 jobs on, while it uses only 4.5–7.3 of 48 cores. Its normalised per-job decline therefore reflects the shared GPU rather than data loading.
  * The paper's comparison needs one GPU per job (`fig14-coordl`).
  * CoorDL's CPU use rises to 1.6–1.7× its 1-job value, more than TensorSocket's 1.1–1.5×.

## 2. Hardware

### 2.1 Machines used in the paper (Table 2)

| Machine | CPU | GPUs | GPU interconnect | Memory | Used for |
|---|---|---|---|---|---|
| A100 server: NVIDIA DGX Station A100 | AMD EPYC 7742, 64 cores / 128 threads, limited to 48 for all experiments (`--cpuset-cpus 0-47`) to mimic Azure's 12:1 vCPU:GPU A100 offerings | 4× NVIDIA A100 40 GB (SXM) | NVLink 3 (all-to-all) | 512 GB | Fig. 8, 9, 14; Tables 3, 4 |
| H100 server (xena2) | 2× Intel Xeon Silver 4410Y, 24 cores / 48 threads | 1× NVIDIA H100 PCIe 80 GB | – | 219 GiB usable | Fig. 10, 12, 15 |
| AWS `g5.2xlarge` | 8 vCPUs, AMD EPYC 7R32 | 1× NVIDIA A10G 24 GB | – | 32 GiB | Fig. 11, 13 |
| AWS `g5.4xlarge` | 16 vCPUs, AMD EPYC 7R32 | 1× NVIDIA A10G 24 GB | – | 64 GiB | Fig. 11, 13 |
| AWS `g5.8xlarge` | 32 vCPUs, AMD EPYC 7R32 | 1× NVIDIA A10G 24 GB | – | 128 GiB | Fig. 11, 13 |

On the DGX Station A100, GPU index 3 is the display GPU, so the original schedules used GPUs `0,1,2,4`. On such a machine pass `GPUS=0,1,2,4`; elsewhere the default logical GPUs `0,1,2,3` are used. Datasets were read from local NVMe storage (`/raid` on the DGX Station).

### 2.2 Minimum hardware to reproduce

* **A100 experiments:** a server with at least 4 NVLink-connected data-center GPUs (A100 or newer) and at least 48 cores. Table 3 needs NVLink for the shared case, and Table 4 needs at least 3 GPUs.
* **H100 experiments:** one GPU with ≥ 80 GB of memory (4× DALL-E collocation and 8× MobileNet under MPS) and at least 24 cores.
* **AWS experiments:** the three g5 instance sizes. Alternatively, on a single machine with an A10G-class GPU and ≥ 32 cores, set `EMULATE=1`, which restricts the container to 8, 16 or 32 CPUs with `--cpuset-cpus` (Section 5.4).
* **Smoke test:** any NVIDIA GPU supported by CUDA 12.1.

## 3. Software and setup

### 3.1 Host prerequisites

* Linux x86-64 with an NVIDIA driver ≥ 535 (CUDA 12.1 wheels; the LLM image's PyTorch 2.5.1 also runs on 535).
* Docker ≥ 24 with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check with `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`. If Docker answers that `--gpus` is not supported and asks for `--runtime=nvidia`, set `export DOCKER_GPU_ARGS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"`.
* About 60 GB of disk for the images and about 400 GB for the datasets (ImageNet, the Joader LMDB copy, LibriSpeech, CC3M subset, Qwen).

### 3.2 Images

All images are defined in `docker/Dockerfile` (one build target each). The CoorDL image has its own build in `docker/coordl/`.

| Image | Used by | Main pinned software |
|---|---|---|
| `tsrepro:timm` | fig08–10, 13, 14, 15, tab03, plotting | Python 3.10, PyTorch 2.1.2 + CUDA 12.1, timm 0.9.12 |
| `tsrepro:clmr` | fig11 | PyTorch 2.0.1 + CUDA 11.8, torchaudio 2.0.2, pytorch-lightning 1.9.5, CLMR @ `423414d` |
| `tsrepro:dalle2` | fig12 | PyTorch 2.1.2, dalle2-pytorch 1.15.6, webdataset 0.2.86, img2dataset 1.45.0 |
| `tsrepro:llm` | tab04 | PyTorch 2.5.1, torchtune 0.4.0, torchao 0.6.1 |
| `tsrepro:joader` | fig15 (Joader) | `tsrepro:timm` + Joader server @ `36beb2d` (Rust nightly-2021-12-02, libtorch 1.11) |
| `tsrepro:coordl` | fig14 (CoorDL) | CoorDL (DALI 0.20 fork) @ `bcde72d` (prebuilt wheel), DS-Analyzer @ `dbdff25`, Python 3.6, PyTorch 1.8.1 (CUDA 11.1) |

Every image except `coordl` contains:

* TensorSocket [0.0.5](https://pypi.org/project/tensorsocket/0.0.5/) from PyPI (commit [`ee231a2`](https://github.com/itu-rad/tensorsocket/commit/ee231a2449e7aa513a4c14fab7f1b35257cdb1c4)).
* DCGM 3.3 (`dcgmi`, `nv-hostengine`) for SM activity, PCIe and NVLink counters.
* The experiment harness (`harness/tsrepro`).

### 3.3 Build and smoke test

```bash
git clone https://github.com/itu-rad/tensorsocket-reproduction.git
cd tensorsocket-reproduction

./reproduce.sh build timm        # 15-30 min; add clmr, dalle2, llm, joader as needed, or `build all`
./reproduce.sh smoke             # ~5 min on any NVIDIA GPU
```

To check the paper's trends in limited time, run any experiment with `--fast` (about 10–30 minutes each; see "Fast mode" in Section 5)

`smoke` does the following:

1. Generates a small synthetic ImageNet-shaped dataset in `$DATA_DIR/synthetic_imagenet`.
2. Trains two MobileNets with their own data loaders.
3. Trains the same two MobileNets fed by one TensorSocket producer.
4. Prints the parsed throughput, CPU utilisation and the monitors that were recorded.

The monitor list should include `DCGMI - SM Active`. If it doesn't, see Section 10. The smoke test checks the toolchain only; its numbers mean nothing.

`./reproduce.sh list` shows every experiment with its image and requirements.

## 4. Datasets

Datasets live in one host directory, `DATA_DIR` (default `./data`), which is mounted at `/data` in the containers. Expected layout:

```
$DATA_DIR/
  imagenet/train/<wnid>/*.JPEG           ImageNet-1k (ILSVRC2012) training set, 1,281,167 images
  imagenet_10pct/train/...               10% subset (Figure 13)       <- prepare imagenet-10pct
  imagenet_joader/ILSVRC-train.lmdb      Joader's input format        <- prepare imagenet-joader
  librispeech/LibriSpeech/...            train-clean-100, test-clean  <- prepare librispeech
  cc3m_raw/Train_GCC-training.tsv        CC3M URL list (manual download)
  cc3m/{00000..}.tar                     CC3M webdataset shards       <- prepare cc3m
  imagenet/val/<wnid>/*.JPEG             validation set, only read by CoorDL's script (Figure 14)
  models/Qwen2.5-0.5B-Instruct/          Qwen2.5 weights              <- prepare qwen
  hf/                                    Hugging Face cache (Alpaca)  <- prepare qwen
  cache/                                 mounted as ~/.cache (CLIP weights, torch hub)
```

Datasets that already exist elsewhere on the host can be mounted read-only instead of copied, e.g.
`DATASET_MOUNTS="/raid/datasets/imagenet:/data/imagenet /raid/datasets/CC3M/train:/data/cc3m"`.

| Dataset | Experiments | How to obtain |
|---|---|---|
| ImageNet-1k | fig08–10, 13–15, tab03 | **Manual.** Download `ILSVRC2012_img_train.tar` from <https://image-net.org> (registration required). Extract it into `imagenet/train/<wnid>/` with one directory per class, e.g. with the [standard extraction script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh). Then run `./reproduce.sh prepare imagenet` to check it. The experiments train without validation; only CoorDL's evaluation script also expects `imagenet/val/<wnid>/` (validation images sorted into class folders, e.g. with the same script). |
| ImageNet 10% | fig13 | `./reproduce.sh prepare imagenet-10pct` hard-links the exact 128,115-image subset used in the paper (`datasets/imagenet_10pct.txt`). |
| ImageNet LMDB for Joader | fig15 (Joader) | `./reproduce.sh prepare imagenet-joader` (~1 h, ~140 GB). |
| LibriSpeech | fig11 | `./reproduce.sh prepare librispeech` (downloads train-clean-100 and test-clean from openslr.org, ~6.6 GB). |
| CC3M | fig12 | **Manual:** download the "Training split" TSV from <https://ai.google.com/research/ConceptualCaptions/download> to `cc3m_raw/Train_GCC-training.tsv`. Then `./reproduce.sh prepare cc3m` downloads the first 170k images with img2dataset. The experiment uses 100k (90k for training), and roughly a third of the URLs are dead. |
| Qwen2.5-0.5B + Alpaca | tab04 | `./reproduce.sh prepare qwen` (`tune download Qwen/Qwen2.5-0.5B-Instruct` and `tatsu-lab/alpaca` from the Hugging Face Hub). |

## 5. Running the experiments

Every experiment is one command. Results go to `results/<experiment>/<tag>/w<workload>/rep<k>/`. Completed workloads are skipped when a command is re-run, so an interrupted experiment resumes where it stopped (`--force` re-runs). Arguments after the experiment id go to the runner (`python -m tsrepro.run --help`). The useful ones:

| Option | Meaning |
|---|---|
| `--repeats 2` | Repeat each workload; the paper ran everything twice. Plots average over repetitions. |
| `--workloads 5,51` | Run only some workloads. Ids are listed in each `experiments/*.csv`. |
| `--var epochs=1` | Shorter runs. The analysis then uses the last epoch instead of the third. |
| `--dry-run` | Print the commands without running them. |

Each `experiments/*.csv` file is a readable list of the exact processes per workload: role, GPU, MPS or not, script, arguments. The CSVs were derived from the paper's original radT schedules.

### Fast mode

The full protocol (3 epochs of ImageNet per workload, 2 repetitions) takes several days of GPU time. Every experiment therefore also has a fast mode, which takes about 10–30 minutes per experiment:

```bash
./reproduce.sh run fig08 --fast       # any experiment; results go to results_fast/
./reproduce.sh plot --fast            # figures_fast/ and figures_fast/REPORT.md
```

* In fast mode each workload is stopped after the experiment's `fast_duration`: 50–300 s, listed in the experiment's CSV header. The clock starts once every trainer has reported its first throughput value, so dataset scanning and model set-up don't count.
* The analysis then measures over the last two thirds of each run, with the first third as warm-up, instead of over the third epoch.
* Throughput comes from `ML - Window Samples/s`, the rate since the previous log point; CPU and GPU metrics are averaged over the same window.
* Fast results and figures never mix with full ones, and every fast figure carries a note saying so.

Fast mode is meant for checking the trends and the toolchain when time is limited. The full mode, without `--fast`, follows the paper's protocol.

| Experiment | Workloads | `fast_duration` | Approx. fast time |
|---|---|---|---|
| fig08 / fig08-1gpu | 10 | 60 s | 20 min |
| fig09 / fig09-1gpu | 12 | 60 s | 25 min |
| tab03 / tab03-1gpu | 2 | 240 s | 15 min |
| fig10 | 2 | 300 s | 12 min |
| fig11 (per instance, 4-way) | 4 | 240 s | 20 min |
| fig12 | 6 | 150 s | 20 min |
| fig13 (per instance) | 2 | 300 s | 12 min |
| tab04 / tab04-1gpu | 2 | 300 s | 15 min |
| fig14 / fig14-1gpu | 8 | 75 s | 20 min |
| fig15 (baseline + TensorSocket) | 16 | 50 s | 25 min |
| fig15-joader | 8 | 90 s | 30 min (Joader registers 1.28M items per workload) |

### 5.1 A100 server (Figures 8, 9, 14; Tables 3, 4)

On a machine with more than 48 CPUs, `reproduce.sh` restricts the container to CPUs `0-47` as in the paper. Override with `CPUSET=...`.

```bash
export DATA_DIR=/path/to/datasets            # and GPUS=0,1,2,4 on a DGX Station A100
./reproduce.sh run fig08 --repeats 2         # Figure 8   (10 workloads)
./reproduce.sh run fig09 --repeats 2         # Figure 9   (12 workloads; 4x points reuse fig08)
./reproduce.sh run tab03                     # Table 3    (2 workloads, I/O measured from storage)
./reproduce.sh build llm && ./reproduce.sh prepare qwen
./reproduce.sh run tab04                     # Table 4    (2 workloads; GPUs 0-2)
./reproduce.sh run fig14 --repeats 2         # Figure 14, baseline + TensorSocket (8 workloads)
./reproduce.sh build coordl                  # see Section 8.3
./reproduce.sh run fig14-coordl              # Figure 14, CoorDL (1-4 jobs)
```

**Table 3.** Disk traffic is only visible if ImageNet is not served from the page cache. `tab03` therefore does two things:

* It evicts the dataset from the cache before each workload (`posix_fadvise`, no root needed).
* It caps the container's memory, and with it its page cache, at `TAB03_MEMORY` (default `96g`). This is below the 140 GB of ImageNet.

### 5.2 H100 server (Figures 10, 12, 15)

```bash
./reproduce.sh run fig10 --repeats 2         # Figure 10 (MPS)
./reproduce.sh build dalle2 && ./reproduce.sh prepare cc3m
./reproduce.sh run fig12 --repeats 2         # Figure 12 (MPS)
./reproduce.sh run fig15 --repeats 2         # Figure 15, baseline + TensorSocket (MPS)
./reproduce.sh build joader && ./reproduce.sh prepare imagenet-joader
./reproduce.sh run fig15-joader --repeats 2  # Figure 15, Joader
```

Figure 15 reports the median per-iteration throughput. Each of its workloads is therefore stopped after 20 minutes (`duration` in the CSV; `--duration 0` runs all 3 epochs). Without this, the 8-way baseline alone would take about 8 hours.

### 5.3 AWS g5 instances (Figures 11, 13)

Launch `g5.2xlarge`, `g5.4xlarge` and `g5.8xlarge` instances. We used the "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)", which comes with Docker and the NVIDIA container toolkit. On each one, run:

```bash
export INSTANCE=g5.2xlarge                   # results are tagged with the instance name
./reproduce.sh build clmr && ./reproduce.sh prepare librispeech
./reproduce.sh run fig11 --workloads 4,104,204,304   # the 4-way workloads shown in Figure 11
./reproduce.sh run fig13                     # Figure 13 (needs imagenet-10pct)
```

Copy the `results/` directories of the three instances into one checkout before plotting; they don't overlap. The full `fig11` experiment also contains the 1- and 2-way workloads that the paper ran but did not plot.

### 5.4 Emulating the g5 instances

On a single machine with one A10G (or similar) GPU and ≥ 32 cores:

```bash
for i in g5.2xlarge g5.4xlarge g5.8xlarge; do
  INSTANCE=$i EMULATE=1 ./reproduce.sh run fig11 --workloads 4,104,204,304
  INSTANCE=$i EMULATE=1 ./reproduce.sh run fig13
done
```

`EMULATE=1` restricts the container to the instance's vCPU count. The CPU model and memory bandwidth still differ from the real instances.

## 6. Collecting, parsing and plotting results

During a run, the harness collects everything automatically. For each workload and repetition, `results/.../rep<k>/` contains:

| File | Content |
|---|---|
| `meta.json` | Commands, GPU inventory, CPU count, start/end time, exit codes |
| `proc<i>_<role>.log` | stdout/stderr of each producer / trainer |
| `proc<i>_<role>.metrics.csv` | Training metrics in MLflow-export format (`run_id,key,value,step,timestamp`), with the radT metric keys used by the paper (`ML - Iterations/s`, `ML - Epoch`, ...) |
| `monitor_cpu.csv`, `monitor_io.csv`, `monitor_smi.csv`, `monitor_dcgm.csv` | 1 Hz system samples: CPU, disk reads, nvidia-smi, DCGM |

To parse and plot:

```bash
./reproduce.sh plot            # all figures/tables with results (and Figure 1)
./reproduce.sh plot fig08      # one of: fig01 fig08 fig09 tab03 fig10 fig11 fig12 fig13 tab04 fig14 fig15
```

This writes the following to `figures/`:

* `<figure>.pdf` / `.png`, with the paper's values drawn as black ticks or dotted lines.
* `<figure>.csv`, with reproduced values, paper values and their ratio.
* `REPORT.md`, which collects all comparison tables.

The parsing code is `harness/tsrepro/results.py` and the per-figure aggregation is `harness/tsrepro/plot.py`. Both mirror the paper's original plotting notebooks.

The paper's numbers are in `paper_results/paper_values.csv`. They were recomputed from the original radT/MLflow exports with the original notebook logic, or, where those exports aren't available, copied from the paper's tables and figure labels; the `source` column says which. Figure 13's original time series are in `paper_results/fig13_original_series.csv`.

## 7. Metrics

| Metric | Definition | Paper |
|---|---|---|
| Samples/s | Training-loop throughput per model as logged by the training script (`ML - Iterations/s`), averaged over the third epoch. The first two epochs are warm-up. | Fig. 8–12, 14, 15 |
| Aggregate samples/s | Sum over the collocated models | Fig. 13 |
| Tokens/s | torchtune's `tokens_per_second_per_gpu`, median over optimizer steps | Table 4 |
| % CPU utilization | Summed `%CPU` of all processes in the container (as `top` reports it), divided by the number of CPUs available to the container | Fig. 8, 10, 14 |
| % GPU utilization | DCGM SM activity (field 1002) | Fig. 8 |
| PCIe / NVLink | DCGM PCIe RX bytes/s (1010), NVLink RX bytes/s (1012), per GPU | Tables 3, 4 |
| Disk I/O | Host block-device reads (`/proc/diskstats`, as `iostat`), MB/s | Table 3 |
| VRAM | `nvidia-smi` memory used per GPU | Tables 3, 4 |

Figure 14 normalizes throughput and CPU utilization by the 1-model value of each method. Figure 15 uses the median per-iteration throughput per model, as in the paper.

## 8. Reproducibility of the paper's results and changes from the original runs

### 8.1 What can be reproduced (guideline C)

The SIGMOD reproducibility guidelines ask: *"If the submitted scripts do not reproduce some figures, add a concrete explanation as to why the core thesis of the paper can be verified without the missing experiments."* The scripts in this package reproduce every figure and table of the paper, with the following exceptions.

* **Figure 1** surveys the GPU instance types offered by AWS, Azure and GCP on 2023-11-30. Offerings change over time, so the survey is not re-collected; its counts are in `paper_results/cloud_instances.csv`, and `./reproduce.sh plot fig01` redraws the figure.
* **The two external baselines**, CoorDL (Figure 14) and Joader (Figure 15), are older research prototypes. Both are included and run (`fig14-coordl`, `fig15-joader`), but with the caveats in Sections 8.3 and 8.4.

The core claims of the paper can still be verified:

* Sharing one data loader increases training throughput and lowers CPU requirements (Figures 8–13, Tables 3–4).
* It reaches the same throughput as the baseline with far fewer vCPUs, which is the cloud cost argument of Figures 11 and 13.

All of these compare TensorSocket against non-shared training on the same hardware, and are fully reproducible with this package. For Figures 14 and 15 the baseline and TensorSocket sides are also fully reproducible, so a reviewer can verify that TensorSocket keeps per-model throughput flat as collocation grows (≈1.0× in Fig. 14, about 1100 samples/s up to 6-way in Fig. 15) while the baseline degrades.

Some results need the paper's hardware (Section 2). On a single GPU, the `*-1gpu` stand-ins verify the CPU-side claims but not the throughput gains that come from giving each model its own GPU (Section 1.1).

### 8.2 CoorDL (Figure 14)

* CoorDL is a fork of NVIDIA DALI 0.20 for Python 3.6 and CUDA 10, both long deprecated. The `coordl` image reproduces the one environment CoorDL was run in, for the paper's A100 runs and for our H100 validation alike.
* **Environment.** `docker/coordl/coordl-env.lock` is an exact `conda list --explicit` of that environment: Python 3.6.13, PyTorch 1.8.1 with CUDA 11.1 and cuDNN 8.0.5, and torchvision 0.9.1. On top of it come apex from ptrblck's `apex_no_distributed` branch and CoorDL's DALI wheel.
* **The wheel.** `docker/coordl/build.sh` downloads the prebuilt CoorDL wheel (63 MB) from Zenodo ([doi:10.5281/zenodo.23400672](https://doi.org/10.5281/zenodo.23400672)) and checks its SHA-256 (`594bce74…`). It is the wheel CoorDL was run with, built from CoorDL @ `bcde72d` with `patches/coordl.patch` (a still-available CUDA 10 build image, and the plugin import).
  * CoorDL's own build can no longer run as is: it downloads NVIDIA's CUDA 10.0 patch 1 installer (`cuda_10.0.130.1_linux.run`, which adds nvJPEG), and NVIDIA no longer hosts it.
  * `COORDL_BUILD_FROM_SOURCE=1 ./reproduce.sh build coordl` still runs that build (a few hours) for anyone who has the installer. `COORDL_WHEEL_URL` overrides the download location, e.g. for a mirror.
* **Evaluation script.** The image contains the authors' evaluation script (DS-Analyzer `hyperparam_imagenet_dali_mp.py`) in two variants:
  * `fig14-coordl` runs one job per GPU, as on the paper's A100 server;
  * `fig14-coordl-1gpu` puts all jobs on GPU 0, the single-GPU stand-in matching `fig14-1gpu`.
* **Fixes to the script** (`patches/ds-analyzer-*.patch`):
  * the iterator's module name;
  * `swapoff` removed, since it isn't possible in a container;
  * `accuracy()` uses `reshape` instead of `view`, as newer PyTorch requires.
* **Recent GPUs.** PyTorch 1.8.1 and the CUDA 10 DALI build ship no kernels for recent GPUs such as the H100, so the driver JIT-compiles them on first use. That takes about 30 minutes once. The image keeps the compiled kernels (about 3 GB) in the mounted cache (`$DATA_DIR/cache/nv`), so later runs start within minutes.
* **Throughput** is reported by CoorDL's script for the first job only (`##Perf`, or its running average in fast mode). That is what Figure 14 normalizes.
* **One job.** With a single job, the coordinated path (`--unified --pin`) stalls during set-up. There is also nothing to coordinate, so the 1-job point runs CoorDL's DALI pipeline without the coordination layer.

### 8.3 Joader (Figure 15)

* Joader is a 2022 proof of concept. It builds from upstream commit `36beb2d` with a pinned nightly Rust toolchain and libtorch 1.11 in the `joader` image.
* Upstream Joader hard-codes 48 server threads. To give it the same data-loading budget as the baseline and TensorSocket (8 workers in total), the image builds it with 8 tokio worker threads (`JOADER_THREADS`).
* `serve.sh` also pins the server process to 8 CPUs, which caps OpenCV's and libtorch's internal threads as well.
* It ships no dataset converter. `workloads/joader/convert_imagenet.py` writes the LMDB format that its reader expects, reconstructed from Joader's source. We checked that the server registers such an LMDB and serves decoded 224×224 samples to the training script's client.
* As in the paper, the TIMM script builds a tensor only from Joader's first batch and then waits for new data without re-batching.

## 9. Repository layout

```
reproduce.sh                  single entry point (build / prepare / run / plot / smoke)
docker/Dockerfile             images: timm, clmr, dalle2, llm, joader
docker/coordl/                CoorDL image build
experiments/*.csv             one file per figure/table: processes, GPUs, MPS, arguments
experiments/fig14_a100_coordl.sh
harness/tsrepro/run.py        experiment runner (launch, MPS, monitors, metadata)
harness/tsrepro/monitors.py   CPU / disk / nvidia-smi / DCGM sampling
harness/tsrepro/tracking.py   metric logging used by the workloads
harness/tsrepro/results.py    result parsing
harness/tsrepro/plot.py       figures and tables
workloads/timm/               image classification (train.py, producer.py, Joader client)
workloads/clmr/               CLMR audio classification
workloads/dalle2/             DALL-E 2 diffusion prior with online CLIP
workloads/llm/                torchtune Qwen2.5 fine-tuning
workloads/joader/             Joader server launch, registration, LMDB conversion
datasets/                     dataset preparation scripts
paper_results/                the paper's values for comparison
validation/                   our validation runs on the H100 server (figures + REPORT.md)
```

## 10. Troubleshooting

* **No `DCGMI - *` metrics.**
  * The containers start their own `nv-hostengine`, which needs `--cap-add SYS_ADMIN` (set by `reproduce.sh`) for profiling fields.
  * If the host already runs DCGM (e.g. `dcgm-exporter`), profiling metrics can be unavailable to a second host engine. Stop the host service during the experiments, or run `nv-hostengine` on the host and remove it from `docker/entrypoint.sh`.
  * Without DCGM, the harness falls back to nvidia-smi GPU utilization and leaves the PCIe/NVLink columns empty.
* **MPS.**
  * The runner starts and stops `nvidia-cuda-mps-control` inside the container for each workload that uses MPS (pipe directory `/tmp/tsrepro-mps`).
  * Make sure no MPS daemon is running on the host, and that the GPUs are in the `Default` compute mode.
* **Consumers wait forever.**
  * TensorSocket uses TCP ports 5556/5557 and 4444/4445 on localhost inside the container.
  * Only one workload runs at a time, but stale processes from an aborted run can hold the ports. Restart the container.
* **Datasets look empty inside the container.**
  * This happens when Docker is installed as a snap package (`snap list docker`). The snap's sandbox only lets it see `/home` and, after `sudo snap connect docker:removable-media`, `/mnt` and `/media`.
  * Keep `DATA_DIR` there, or bind-mount the dataset disk below `/mnt` (e.g. `sudo mount --bind /raid /mnt/raid`).
* **Logs.** Each workload prints the tail of failing trainers' logs. The full logs are in `results/.../proc*_*.log`.
