"""torchtune full single-device fine-tuning, optionally fed by TensorSocket.

Runs torchtune 0.4.0's ``FullFinetuneRecipeSingleDevice`` unchanged, except
that with ``shared_loader=True`` the recipe's data loader is replaced by a
TensorConsumer. Usage (torchtune config syntax)::

    python train.py --config config.yaml [shared_loader=True] [key=value ...]
"""

import importlib.util
import sys
from pathlib import Path

import torchtune
from omegaconf import DictConfig
from tensorsocket import TensorConsumer
from torchtune import config

KEYS = ("tokens", "labels")


def _load_recipe():
    # torchtune ships its recipes next to the package but blocks `import recipes`.
    path = Path(torchtune.__file__).resolve().parents[1] / "recipes" / "full_finetune_single_device.py"
    spec = importlib.util.spec_from_file_location("full_finetune_single_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FullFinetuneRecipeSingleDevice


FullFinetuneRecipeSingleDevice = _load_recipe()


class _EpochSampler:
    """The recipe calls ``set_epoch`` on its sampler; the producer owns shuffling."""

    def set_epoch(self, epoch: int) -> None:
        pass


class SharedBatches:
    """Turns the consumer's (tokens, labels) tuples back into torchtune batch dicts."""

    def __init__(self, batch_size: int):
        self.consumer = TensorConsumer("5556", "5557", (4444, 4445), batch_size=batch_size)

    def __len__(self):
        return len(self.consumer)

    def __iter__(self):
        for batch in self.consumer:
            yield dict(zip(KEYS, batch))


class SharedRecipe(FullFinetuneRecipeSingleDevice):
    def _setup_data(self, cfg_dataset, shuffle, batch_size, collate_fn):
        return _EpochSampler(), SharedBatches(batch_size)


@config.parse
def main(cfg: DictConfig) -> None:
    recipe_cls = SharedRecipe if cfg.get("shared_loader", False) else FullFinetuneRecipeSingleDevice
    recipe = recipe_cls(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(main())
