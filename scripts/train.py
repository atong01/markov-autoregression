"""Train a MarS model on tetrapeptide or MD-CATH trajectories."""

import argparse
import logging
import os

import pytorch_lightning as pl
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from mars.utils import set_seed
from mars.model.module import MarSModule
from scripts.training_args import parse_train_args

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


def _find_latest_checkpoint(model_dir):
    ckpt_files = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".ckpt")
    ]
    if not ckpt_files:
        return None
    return max(ckpt_files, key=os.path.getctime)


def _build_dataloaders(args):
    if args.mdcath:
        from mars.data.dataset import MarSDatasetMDCath as MarSDataset
    else:
        from mars.data.dataset import MarSDataset4AA as MarSDataset

    trainset = MarSDataset(args, split=args.train_split)
    valset = MarSDataset(args, split=args.val_split, repeat=args.val_repeat)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    return train_loader, val_loader


def _build_trainer(args):
    model_dir = os.environ["MODEL_DIR"]

    pl_logger = False
    if args.wandb:
        pl_logger = WandbLogger(
            project="mars",
            name=args.run_name,
            config=vars(args),
            save_dir=model_dir,
        )

    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches or 1.0,
        limit_val_batches=args.val_batches or 1.0,
        num_sanity_val_steps=0,
        gradient_clip_val=args.grad_clip,
        default_root_dir=model_dir,
        callbacks=[
            ModelCheckpoint(dirpath=model_dir, save_top_k=-1, every_n_epochs=args.ckpt_freq),
        ],
        logger=pl_logger,
    )


def _resolve_checkpoint(args):
    if args.ckpt is not None:
        logger.info(f"Using user-supplied checkpoint: {args.ckpt}")
        return args.ckpt
    latest = _find_latest_checkpoint(os.environ["MODEL_DIR"])
    if latest:
        logger.info(f"Resuming from existing checkpoint: {latest}")
    else:
        logger.info("No checkpoint found. Training from scratch.")
    return latest


if __name__ == "__main__":
    args = parse_train_args()
    set_seed(args.seed)

    train_loader, val_loader = _build_dataloaders(args)
    model = MarSModule(args)
    trainer = _build_trainer(args)
    ckpt_path = _resolve_checkpoint(args)

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
