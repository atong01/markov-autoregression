"""Inject a hyper_parameters['args'] Namespace into a Lightning checkpoint
that was saved without one.

The mdcath frames-AR checkpoint epoch=124-step=180625.ckpt has an empty
'hyper_parameters' dict, so MarSARModule.load_from_checkpoint fails with
"BaseModule.__init__() missing 1 required positional argument: 'args'".
This script reconstructs args from train_ar.sh defaults and writes a
patched copy alongside the original.
"""

import argparse
import os

import torch

torch.serialization.add_safe_globals([argparse.Namespace])


# Mirrors train_ar.sh + scripts/training_args.py defaults for the mdcath
# frames-AR run that produced epoch=124-step=180625.ckpt.
FRAMES_AR_ARGS = dict(
    seed=42,
    train_split="splits/mdCATH_train.csv",
    val_split="splits/mdCATH_val.csv",
    data_dir="/home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed",
    mdcath=True,
    data_temperature=450,
    crop=256,
    euclidean=False,
    ca_only=False,
    s_translation=1.0,
    deterministic_dataset=False,
    msm_num_states=10,
    clusters_per_batch=2,
    samples_per_cluster=12,
    msm_lagtime=50,
    num_layers=5,
    embed_dim=384,
    mha_heads=16,
    ipa_heads=4,
    ipa_head_dim=32,
    ipa_qk=8,
    ipa_v=8,
    abs_pos_emb=False,
    epochs=1000,
    batch_size=1,
    lr=1e-4,
    grad_clip=1.0,
    ema=True,
    ema_decay=0.999,
    ckpt=None,
    num_workers=4,
    train_batches=None,
    val_batches=None,
    val_repeat=5,
    ar=True,
    num_bins=8192,
    discretize_min=-5.0,
    discretize_max=5.0,
    ar_temperature=1.0,
    ar_top_k=None,
    ar_top_p=None,
    label_smoothing=0.05,
    use_empirical_offsets=False,
    auto_discretize_range=True,
    discretize_std_k=4.0,
    calibration_batches=32,
    bf16=True,
    ckpt_freq=25,
    wandb=True,
    run_name="mdcath_ar_v6",
    wandb_name="mdcath_ar_v6",
    model_dir="/home/mila/d/danyal.rehman/scratch/mdcath/workdir",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", required=True)
    p.add_argument("--out_ckpt", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.out_ckpt) and not args.overwrite:
        raise SystemExit(f"{args.out_ckpt} exists; pass --overwrite to replace.")

    ck = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    # Match the on-disk format used by working ckpts: hyper_parameters is a
    # dict {'args': Namespace(...)} with hparams_name='kwargs', so PL unpacks
    # it as **hp into BaseModule.__init__(args=...).
    ck["hyper_parameters"] = {"args": argparse.Namespace(**FRAMES_AR_ARGS)}
    ck["hparams_name"] = "kwargs"
    torch.save(ck, args.out_ckpt)
    print(f"wrote {args.out_ckpt}")


if __name__ == "__main__":
    main()
