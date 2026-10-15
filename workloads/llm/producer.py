"""TensorSocket producer wrapping torchtune's (Hugging Face backed) Alpaca loader.

Builds the data loader exactly as the recipe's ``_setup_data`` does and shares
its batches as (tokens, labels) tuples. Usage::

    python producer.py --config config.yaml [key=value ...]
"""

import sys
from functools import partial

from omegaconf import DictConfig
from tensorsocket.producer import TensorProducer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config
from torchtune.config._utils import _get_component_from_path

from tsrepro import tracking

KEYS = ("tokens", "labels")


class TupleLoader:
    def __init__(self, loader, sampler):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.sampler.set_epoch(self.epoch)
        self.epoch += 1
        for batch in self.loader:
            yield tuple(batch[k] for k in KEYS)


@config.parse
def main(cfg: DictConfig) -> None:
    tokenizer = config.instantiate(cfg.tokenizer)
    loss = config.instantiate(cfg.loss)
    ds = config.instantiate(cfg.dataset, tokenizer)
    collate = _get_component_from_path(cfg.collate_fn)
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=cfg.shuffle, seed=0)
    loader = DataLoader(
        dataset=ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=partial(collate, padding_idx=tokenizer.pad_id, ignore_idx=loss.ignore_index),
    )

    producer = TensorProducer(TupleLoader(loader, sampler), "5556", "5557", heart_ports=(4444, 4445))
    with tracking.run():
        for epoch in range(cfg.epochs):
            print(f"Epoch: {epoch}", flush=True)
            for _ in producer:
                pass
        producer.join()
    print("Producer finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
