"""Train a MarS model on tetrapeptide or MD-CATH trajectories."""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

import pytorch_lightning as pl
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from markov_autoregression.utils import set_seed
from scripts.training_args import parse_train_args

# Use Lightning's seed_everything when available; it also seeds DataLoader
# workers deterministically so the dataset's RNG is reproducible across ranks.
try:
    from pytorch_lightning import seed_everything as _seed_everything
except ImportError:
    _seed_everything = None

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cuda.preferred_linalg_library("magma")


def _find_latest_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        return None
    # Prefer last.ckpt — it is overwritten every epoch and is the canonical
    # resume point for SLURM-array training.
    last = os.path.join(model_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    ckpt_files = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".ckpt")
    ]
    if not ckpt_files:
        return None
    return max(ckpt_files, key=os.path.getctime)


def _build_module(args):
    if getattr(args, "ar", False):
        from markov_autoregression.model.autoregressive_model import MarSARModule
        return MarSARModule(args)
    if getattr(args, "mean_flow", False):
        from markov_autoregression.model.module import MarSMeanFlowModule
        return MarSMeanFlowModule(args)
    from markov_autoregression.model.module import MarSModule
    return MarSModule(args)


def _build_dataloaders(args):
    if args.mdcath:
        from markov_autoregression.data.dataset import MarSDatasetMDCath as MarSDataset
    else:
        from markov_autoregression.data.dataset import MarSDataset4AA as MarSDataset

    trainset = MarSDataset(args, split=args.train_split, translate=True)
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
            project="markov-autoregression",
            name=args.wandb_name or args.run_name,
            config=vars(args),
            save_dir=model_dir,
        )

    precision = "bf16-mixed" if getattr(args, "bf16", False) else "32-true"

    # Multi-process detection: under `srun --ntasks-per-node=N`, each task is
    # a separate Python process with one visible GPU. SLURM_NTASKS>1 means we
    # need DDP to sync them; without an explicit strategy Lightning may not
    # initialise the process group, causing rank 0 to wait at on_fit_start.
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    use_ddp = slurm_ntasks > 1 or torch.cuda.device_count() > 1
    if use_ddp:
        # find_unused_parameters=True is needed because zero-init residual
        # gates (cross-attn gate, IPA linear_out) make some params receive
        # zero gradient on step 1 — DDP would otherwise hang waiting for them.
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    return pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices="auto",
        strategy=strategy,
        precision=precision,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches or 1.0,
        limit_val_batches=args.val_batches or 1.0,
        num_sanity_val_steps=0,
        gradient_clip_val=args.grad_clip,
        default_root_dir=model_dir,
        callbacks=[
            # Periodic permanent checkpoints, kept indefinitely.
            ModelCheckpoint(dirpath=model_dir, save_top_k=-1, every_n_epochs=args.ckpt_freq),
            # Rolling last.ckpt — overwritten every epoch, used for SLURM-array resume.
            ModelCheckpoint(
                dirpath=model_dir,
                save_top_k=0,
                save_last=True,
                every_n_epochs=1,
            ),
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
    # seed_everything(workers=True) seeds Python/numpy/torch on the main proc
    # AND installs a worker_init_fn so DataLoader workers receive deterministic
    # per-rank seeds. Without this, np.random.* calls inside dataset.__getitem__
    # diverge across worker procs and across DDP ranks.
    if _seed_everything is not None:
        _seed_everything(args.seed, workers=True)
    else:
        set_seed(args.seed)

    train_loader, val_loader = _build_dataloaders(args)
    model = _build_module(args)

    # Pre-calibrate the AR discretizer BEFORE trainer.fit() — every rank runs
    # the same code with identical torch seeds, so they all derive identical
    # per-channel bounds without any cross-rank communication. Doing this
    # inside Lightning's on_fit_start risked deadlock under DDP (the loader
    # iterator and NCCL collectives don't always cooperate cleanly there).
    if (
        getattr(args, "ar", False)
        and getattr(args, "auto_discretize_range", False)
        and not getattr(model, "_calibrated", True)
    ):
        logger.info("Pre-fit calibration of AR discretizer (CPU, deterministic)…")
        model.calibrate_discretizer(
            train_loader,
            num_batches=int(getattr(args, "calibration_batches", 32)),
            k=float(getattr(args, "discretize_std_k", 4.0)),
            verbose=True,
        )

    trainer = _build_trainer(args)
    ckpt_path = _resolve_checkpoint(args)

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
