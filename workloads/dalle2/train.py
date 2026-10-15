"""Online DALL-E 2 diffusion-prior training, with or without TensorSocket.

Non-shared: each process reads CC3M and runs the CLIP image and text encoders
itself (inside the prior's forward). Shared: the producer runs the data loading
and CLIP once and the consumer receives (text_embed, text_encodings,
image_embed) batches.

The training step and the logged ``ML - Iterations/s`` follow the original
script: samples per second of the training step, logged every 10 steps with
the epoch as step.
"""

import argparse
import time

import torch
from dalle2_pytorch.utils import Timer
from tensorsocket import TensorConsumer

from common import cc3m_loader, load_config, make_trainer
from tsrepro import tracking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prior_online.json")
    ap.add_argument("--cc3m", required=True, help="directory with CC3M webdataset shards")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--shared-loader", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["train"]["random_seed"])
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda")

    trainer = make_trainer(cfg, device)

    if args.shared_loader:
        loader = TensorConsumer("5556", "5557", (4444, 4445), batch_size=cfg["data"]["batch_size"])
    else:
        loader = cc3m_loader(cfg, args.cc3m, args.workers)

    samples_timer = Timer()
    with tracking.run() as run:
        for epoch in range(epochs):
            print(f"Starting epoch {epoch}", flush=True)
            run.log_metric("ML - Epoch", epoch + 1, epoch + 1)
            epoch_start, epoch_samples = time.time(), 0
            for idx, out in enumerate(loader):
                trainer.train()
                samples_timer.reset()

                if args.shared_loader:
                    text_embed, text_encodings, image_embed = out
                    loss = trainer(text_embed=text_embed, text_encodings=text_encodings, image_embed=image_embed)
                    n = len(text_embed)
                else:
                    img, txt = out
                    img = img.to(device, non_blocking=True)
                    txt = txt.to(device, non_blocking=True)
                    loss = trainer(text=txt, image=img)
                    n = len(txt)

                trainer.update()
                samples_per_sec = n / samples_timer.elapsed()
                epoch_samples += n

                if idx % 10 == 0:
                    run.log_metrics(
                        {
                            "ML - Iterations/s": samples_per_sec,
                            "ML - Loss": loss,
                            "PT - Mem Reserved": float(torch.cuda.memory_reserved(0)),
                            "PT - Mem Allocated": float(torch.cuda.memory_allocated(0)),
                        },
                        epoch,
                    )
                    print(f"epoch {epoch} step {idx}: {samples_per_sec:.1f} samples/s, loss {loss:.4f}", flush=True)
            # Wall-clock rate including time spent waiting for data.
            run.log_metric("ML - Epoch Samples/s", epoch_samples / (time.time() - epoch_start), epoch)
        run.log_metric("ML - Epoch", epochs + 1, epochs + 1)
    print("Training done", flush=True)


if __name__ == "__main__":
    main()
