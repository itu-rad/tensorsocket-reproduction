#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct and the Alpaca dataset for Table 4.
# Models go to /data/models, Hugging Face datasets to $HF_HOME (/data/hf).
set -euo pipefail

tune download Qwen/Qwen2.5-0.5B-Instruct \
    --output-dir /data/models/Qwen2.5-0.5B-Instruct --ignore-patterns None
python3 -c "from datasets import load_dataset; print(load_dataset('tatsu-lab/alpaca', split='train'))"
