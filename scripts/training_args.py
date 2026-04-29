"""Command-line argument parser for MarS training."""

import os
from argparse import ArgumentParser


def parse_train_args():
    parser = ArgumentParser(description="Train a MarS model.")

    # ---- Seed ----
    parser.add_argument("--seed", type=int, default=42)

    # ---- Data ----
    data = parser.add_argument_group("Data")
    data.add_argument("--train_split", type=str, required=True)
    data.add_argument("--val_split", type=str, default=None)
    data.add_argument("--data_dir", type=str, required=True)
    data.add_argument("--mdcath", action="store_true")
    data.add_argument("--data_temperature", type=int, default=320)
    data.add_argument("--crop", type=int, default=256)
    data.add_argument("--euclidean", action="store_true")
    data.add_argument("--ca_only", action="store_true")
    data.add_argument("--backbone", action="store_true")
    data.add_argument("--s_translation", type=float, default=1.0)

    # ---- MSM / clustering ----
    msm = parser.add_argument_group("MSM clustering")
    msm.add_argument("--msm_num_states", type=int, default=10)
    msm.add_argument("--clusters_per_batch", type=int, default=10)
    msm.add_argument("--samples_per_cluster", type=int, default=100)
    msm.add_argument("--msm_lagtime", type=int, default=1)

    # ---- Model ----
    model = parser.add_argument_group("Model")
    model.add_argument("--num_layers", type=int, default=5)
    model.add_argument("--embed_dim", type=int, default=384)
    model.add_argument("--mha_heads", type=int, default=16)
    model.add_argument("--ipa_heads", type=int, default=4)
    model.add_argument("--ipa_head_dim", type=int, default=32)
    model.add_argument("--ipa_qk", type=int, default=8)
    model.add_argument("--ipa_v", type=int, default=8)
    model.add_argument("--abs_pos_emb", action="store_true")

    # ---- Optimization ----
    optim = parser.add_argument_group("Optimization")
    optim.add_argument("--epochs", type=int, default=100)
    optim.add_argument("--batch_size", type=int, default=8)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--grad_clip", type=float, default=1.0)
    optim.add_argument("--ema", action="store_true", default=True)
    optim.add_argument("--ema_decay", type=float, default=0.999)

    # ---- Training loop ----
    loop = parser.add_argument_group("Training loop")
    loop.add_argument("--ckpt", type=str, default=None)
    loop.add_argument("--num_workers", type=int, default=4)
    loop.add_argument("--train_batches", type=int, default=None)
    loop.add_argument("--val_batches", type=int, default=None)
    loop.add_argument("--val_repeat", type=int, default=1)

    # ---- Autoregressive ----
    ar = parser.add_argument_group("Autoregressive")
    ar.add_argument("--ar", action="store_true",
                    help="Use the autoregressive model instead of flow matching.")
    ar.add_argument("--num_bins", type=int, default=8192)
    ar.add_argument("--discretize_min", type=float, default=-5.0)
    ar.add_argument("--discretize_max", type=float, default=5.0)
    ar.add_argument("--ar_temperature", type=float, default=1.0)
    ar.add_argument("--ar_top_k", type=int, default=None)
    ar.add_argument("--ar_top_p", type=float, default=None)
    ar.add_argument("--label_smoothing", type=float, default=0.0)
    ar.add_argument("--use_empirical_offsets", action="store_true")
    ar.add_argument("--auto_discretize_range", action="store_true",
                    help="Calibrate per-channel discretization bounds from the "
                         "training data as μ ± k·σ before training starts.")
    ar.add_argument("--discretize_std_k", type=float, default=4.0,
                    help="Std-multiplier for auto-calibrated bounds (default 4σ).")
    ar.add_argument("--calibration_batches", type=int, default=32,
                    help="Number of training batches sampled during calibration.")
    ar.add_argument("--bf16", action="store_true",
                    help="Use bf16-mixed precision (recommended for AR training).")

    # ---- Logging ----
    log = parser.add_argument_group("Logging")
    log.add_argument("--ckpt_freq", type=int, default=1)
    log.add_argument("--wandb", action="store_true")
    log.add_argument("--run_name", type=str, default="default")
    log.add_argument("--wandb_name", type=str, default=None)
    log.add_argument("--model_dir", type=str, default="workdir")

    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join(args.model_dir, args.run_name)

    return args
