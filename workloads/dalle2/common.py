"""Shared pieces of the DALL-E 2 diffusion-prior workload (Section 4.4).

The paper trains the diffusion prior of DALLE2-pytorch "online": captions and
images are read from CC3M and embedded by a frozen CLIP ViT-B/32 during
training. This module provides the CC3M reader (webdataset shards written by
img2dataset, see datasets/prepare_cc3m.sh) and the model construction that the
trainer and the producer share.
"""

from __future__ import annotations

import json
from pathlib import Path

import clip
import torch
import webdataset as wds
from accelerate import Accelerator
from dalle2_pytorch import DiffusionPriorTrainer
from dalle2_pytorch.train_configs import DiffusionPriorConfig
from torchvision import transforms as T


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def train_shards(cc3m_dir: str, train_split: float) -> tuple[list[str], int]:
    """First ``train_split`` of the shards and the number of images they hold."""
    shards = sorted(Path(cc3m_dir).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"no CC3M webdataset shards (*.tar) in {cc3m_dir}")
    shards = shards[: max(1, int(len(shards) * train_split))]
    count = 0
    for s in shards:
        stats = s.with_name(s.stem + "_stats.json")
        count += json.loads(stats.read_text())["successes"] if stats.exists() else 0
    return [str(s) for s in shards], count


def cc3m_loader(cfg: dict, cc3m_dir: str, workers: int):
    """Batches of (image in [0, 1] at 224x224, CLIP token ids)."""
    data = cfg["data"]
    batch_size = data["batch_size"]
    shards, available = train_shards(cc3m_dir, data["splits"]["train"])
    num_samples = int(data["num_data_points"] * data["splits"]["train"])
    if available:
        num_samples = min(num_samples, available)
    num_batches = num_samples // batch_size

    # CLIP's preprocessing without the normalisation: OpenAIClipAdapter
    # normalises inside embed_image().
    preprocess = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Lambda(lambda im: im.convert("RGB")),
            T.ToTensor(),
        ]
    )

    def tokenize(caption: str) -> torch.Tensor:
        return clip.tokenize(caption, truncate=True)[0]

    dataset = (
        wds.WebDataset(shards, shardshuffle=False, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg;png;jpeg", "txt", handler=wds.warn_and_continue)
        .map_tuple(preprocess, tokenize)
        .batched(batch_size, partial=False)
    )
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=workers, pin_memory=True)
    # Every epoch restarts the stream and stops after num_batches batches.
    return loader.repeat().with_epoch(num_batches).with_length(num_batches)


def make_trainer(cfg: dict, device: torch.device) -> DiffusionPriorTrainer:
    prior = DiffusionPriorConfig(**cfg["prior"]).create()
    train = cfg["train"]
    trainer = DiffusionPriorTrainer(
        diffusion_prior=prior,
        lr=train["lr"],
        wd=train["wd"],
        max_grad_norm=train["max_grad_norm"],
        amp=train["amp"],
        use_ema=train["use_ema"],
        device=device,
        accelerator=Accelerator(),
        warmup_steps=train["warmup_steps"],
    )
    return trainer.to(device)


def make_clip(cfg: dict):
    return DiffusionPriorConfig(**cfg["prior"]).clip.create()
