import argparse
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

from clmr.data import ContrastiveDataset
from clmr.datasets import get_dataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning, SupervisedLearning
from clmr.utils import yaml_config_hook
import sys
from tensorsocket import TensorConsumer
from tsrepro import tracking
from pytorch_lightning.callbacks import Callback
import torch
import os
import tempfile
import time
import logging
#_logger = logging.getLogger("mlflow")
#_logger.setLevel(logging.INFO)

#logging.basicConfig(level=logging.WARN)
logging.disable(logging.CRITICAL)

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

class SharedLoader:
    """Gives Lightning a sized iterable over a TensorConsumer (one epoch per iteration)."""

    def __init__(self, port="5556", ack_port="5557", heart_ports=(4444, 4445), batch_size=128):
        self.consumer = TensorConsumer(port, ack_port, heart_ports, batch_size=batch_size)

    def __iter__(self):
        return iter(self.consumer)

    def __len__(self):
        return len(self.consumer)


class RADTLogger(Callback):
    def __init__(self, mlflow, global_epoch, log_interval, batch_size, verbose=True):
        self.mlflow = mlflow
        self.global_epoch = global_epoch
        self.log_interval = log_interval
        self.verbose = verbose
        self.batch_size = batch_size
        self.state = {"epochs": 0, "batches": 0}
        self.mlflow.log_metric("ML - Epoch", self.state["epochs"], self.state["epochs"])

    @property
    def state_key(self) -> str:
        # note: we do not include `verbose` here on purpose
        return f"RADTLogger[{self.state}]"
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.window_start, self.window_batches = self.start_time, 0
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, *args, **kwargs):
        end_time = time.time() - self.start_time
        batches_processed = self.state["batches"]
        batches_per_second = batches_processed / end_time
        iterations_per_second = batches_per_second * self.batch_size
        self.mlflow.log_metric("ML - Iterations/s", iterations_per_second)

        self.state["epochs"] += 1
        self.mlflow.log_metric("ML - Epoch", self.state["epochs"], self.state["epochs"])
        self.state["batches"] = 0

    def on_train_batch_end(self, *args, **kwargs):
        self.state["batches"] += 1
        self.window_batches += 1
        if self.window_batches == 20:
            # Within-epoch rate, used by short (--fast) runs; ML - Iterations/s is per epoch.
            now = time.time()
            self.mlflow.log_metric("ML - Window Samples/s", 20 * self.batch_size / (now - self.window_start), self.state["epochs"])
            self.window_start, self.window_batches = now, 0
        if self.state["batches"] % self.log_interval == 0:
            ml_metrics = {}
            ml_metrics["PT - Mem Reserved"] = float(torch.cuda.memory_reserved(0))
            ml_metrics["PT - Mem Allocated"] = float(torch.cuda.memory_allocated(0))
            self.mlflow.log_metrics(ml_metrics, self.state["epochs"])

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

def main():
    global mlflow
    global global_epoch
    mlflow = None
    global_epoch = 0

    parser = argparse.ArgumentParser(description="CLMR")
    config_file = sys.argv[1]

    config = yaml_config_hook(config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument(
        "--shared-loader",
        type=int,
        default=0,
        help="whether to use the shared data loader. remember to instantiate producer.",
    )
    args = parser.parse_known_args()[0]
    pl.seed_everything(args.seed)

    # ------------
    # data augmentations
    # ------------
    if args.supervised:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        num_augmented_samples = 1
    else:
        train_transform = [
            RandomResizedCrop(n_samples=args.audio_length),
            RandomApply([PolarityInversion()], p=args.transforms_polarity),
            RandomApply([Noise()], p=args.transforms_noise),
            RandomApply([Gain()], p=args.transforms_gain),
            RandomApply(
                [HighLowPass(sample_rate=args.sample_rate)], p=args.transforms_filters
            ),
            RandomApply([Delay(sample_rate=args.sample_rate)], p=args.transforms_delay),
            RandomApply(
                [
                    PitchShift(
                        n_samples=args.audio_length,
                        sample_rate=args.sample_rate,
                    )
                ],
                p=args.transforms_pitch,
            ),
            RandomApply(
                [Reverb(sample_rate=args.sample_rate)], p=args.transforms_reverb
            ),
        ]
        num_augmented_samples = 2

    # ------------
    # dataloaders
    # ------------
    if args.shared_loader:
        train_loader = SharedLoader(batch_size=args.batch_size)
        valid_loader = None
    else:
        train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train")
        valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")
        contrastive_train_dataset = ContrastiveDataset(
            train_dataset,
            input_shape=(1, args.audio_length),
            transform=ComposeMany(
                train_transform, num_augmented_samples=num_augmented_samples
            ),
        )

        contrastive_valid_dataset = ContrastiveDataset(
            valid_dataset,
            input_shape=(1, args.audio_length),
            transform=ComposeMany(
                train_transform, num_augmented_samples=num_augmented_samples
            ),
        )

        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            shuffle=True,
        )

        valid_loader = DataLoader(
            contrastive_valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            shuffle=False,
        )


    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
        strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
        supervised=args.supervised,
        #out_dim=train_dataset.n_classes,
        out_dim=251,
    )

    # ------------
    # model
    # ------------
    if args.supervised:
        module = SupervisedLearning(
            args, 
            encoder, 
            #output_dim=train_dataset.n_classes,
            output_dim=251,
            )
    else:
        module = ContrastiveLearning(args, encoder)

    # Per-process log directory: collocated trainers otherwise race on the same one.
    logger = TensorBoardLogger(tempfile.mkdtemp(prefix="clmr-tb-"), name="CLMRv2-{}".format(args.dataset), default_hp_metric=False)
    if args.checkpoint_path:
        module = module.load_from_checkpoint(
            args.checkpoint_path, 
            encoder=encoder, 
            #output_dim=train_dataset.n_classes,
            output_dim=251
        )

    else:
        # ------------
        # training
        # ------------

        if args.supervised:
            early_stopping = EarlyStopping(monitor="Valid/loss", patience=20)
        else:
            early_stopping = None

        with tracking.run() as run:
            #global mlflow
            #global global_epoch
            global_epoch = 0
            mlflow = run
            trainer = Trainer.from_argparse_args(
                args,
                logger=logger,
                sync_batchnorm=True,
                max_epochs=args.max_epochs,
                log_every_n_steps=100,
                check_val_every_n_epoch=1,
                accelerator=args.accelerator,
                callbacks=[RADTLogger(mlflow, global_epoch, 100, args.batch_size, verbose=False), TQDMProgressBar(refresh_rate=20)],
                limit_val_batches=0,
                num_sanity_val_steps=0,
            )

            trainer.fit(module, train_loader)

    if args.supervised:
        test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test")

        contrastive_test_dataset = ContrastiveDataset(
            test_dataset,
            input_shape=(1, args.audio_length),
            transform=None,
        )

        device = "cuda:0" if args.gpus else "cpu"
        results = evaluate(
            module.encoder,
            None,
            contrastive_test_dataset,
            args.dataset,
            args.audio_length,
            device=device,
        )
        print(results)

if __name__ == "__main__":
    main()