"""TensorSocket producer for online DALL-E 2 prior training.

Moves both the CC3M data loading and the frozen CLIP inference into the
producer (Figure 7b): each batch is embedded once and the resulting
(text_embed, text_encodings, image_embed) tensors are shared with all
consumers on the GPU.
"""

import argparse

import torch
from tensorsocket.producer import TensorProducer

from common import cc3m_loader, load_config, make_clip
from tsrepro import tracking


class ClipEmbeddingLoader:
    def __init__(self, loader, clip, device):
        self.loader = loader
        self.clip = clip
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for image, text in self.loader:
            with torch.no_grad():
                image = image.to(self.device, non_blocking=True)
                text = text.to(self.device, non_blocking=True)
                image_embed, _ = self.clip.embed_image(image)
                text_embed, text_encodings = self.clip.embed_text(text)
            yield text_embed, text_encodings, image_embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prior_online.json")
    ap.add_argument("--cc3m", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["train"]["random_seed"])
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda")

    clip = make_clip(cfg).to(device).eval()
    loader = ClipEmbeddingLoader(cc3m_loader(cfg, args.cc3m, args.workers), clip, device)
    producer = TensorProducer(loader, "5556", "5557", heart_ports=(4444, 4445), rubber_band_pct=0.0001)

    with tracking.run():
        for epoch in range(epochs):
            print(f"Epoch: {epoch}", flush=True)
            for _ in producer:
                pass
        producer.join()
    print("Producer finished", flush=True)


if __name__ == "__main__":
    main()
