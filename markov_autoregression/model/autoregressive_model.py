"""Autoregressive variant of the MarS module.

Predicts the per-residue 21-dim latent (7 frame offsets + 14 torsion sin/cos)
of the next MSM state autoregressively, one discrete bin at a time, conditioned
on the starting structure and protein sequence.

Discretization range: per-channel ranges are computed once at the start of
training as μ ± k·σ from a calibration pass over the training data. This is
critical because channels have heterogeneous dynamic ranges — quaternions span
~[-1, 1] while frame translations span tens of Ångströms.

Loss: per-token cross-entropy over `num_bins` classes, with optional label
smoothing and residue-padding masking. The targets are bin indices in
[0, num_bins); inputs are shifted right and offset by +1 (index 0 reserved as
START), so the embedding table has `num_bins + 1` rows and the output head
produces `num_bins` logits aligned with the targets.
"""

import logging
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..data.discretize import DiscretizeTransform
from ..neural_networks.causal_transformer import CausalARModel
from ..vendored.ema import ExponentialMovingAverage
from .module import BaseModule

logger = logging.getLogger(__name__)


def _filter_logits(logits: torch.Tensor, top_k=None, top_p=None) -> torch.Tensor:
    """Top-k and/or top-p filtering applied along the last dim. Works for any
    number of leading dims (used for both 2-D sampling and 3-D likelihood)."""
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        threshold = v[..., -1:].expand_as(logits)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(sorted_logits).scatter_(-1, sorted_idx, sorted_logits)

    return logits


class MarSARModule(BaseModule):
    """Autoregressive analogue of MarSModule.

    Reuses BaseModule for rigid/latent construction, EMA, and atom14 decode.
    Overrides general_step (cross-entropy over discretized bins) and inference
    (KV-cached AR sampling). Adds compute_log_likelihood for exact likelihood
    evaluation, and an on_fit_start hook that calibrates per-channel
    discretization bounds from the training data (μ ± k·σ).
    """

    # Discretized-then-undiscretized (sin, cos) torsion components do not
    # naturally satisfy sin²+cos²=1, so the rotation matrices built from them
    # in frames_torsions_to_atom14 would be non-orthogonal — distorting bond
    # lengths/angles in the decoded atom14. Normalising recovers a valid
    # rotation. Critical for downstream drug-discovery use of generated geom.
    normalize_torsions = True

    def __init__(self, args):
        # Skip BaseModule.__init__ so we don't instantiate the flow MarSModel.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.args = args
        self.latent_dim = 21

        self.model = CausalARModel(args, self.latent_dim)
        self.discretize = DiscretizeTransform(
            num_bins=args.num_bins,
            min_val=args.discretize_min,
            max_val=args.discretize_max,
            latent_dim=self.latent_dim,
        )
        self._calibrated = False

        if getattr(args, "ema", False):
            self.ema = ExponentialMovingAverage(model=self.model, decay=args.ema_decay)
            self.cached_weights = None

    # ----- Batch reshaping ---------------------------------------------------

    @staticmethod
    def _pair_with_tau(batch):
        for key in ["trans", "rots", "torsions"]:
            B, T = batch[key].shape[:2]
            rest = batch[key].shape[2:]
            batch[key] = torch.stack(
                [
                    batch[key].reshape(B * T, *rest),
                    batch[key + "_plus_tau"].reshape(B * T, *rest),
                ],
                dim=1,
            )
        for key in ["torsion_mask", "mask", "seqres"]:
            batch[key] = (
                batch[key]
                .unsqueeze(1)
                .expand(B, T, *batch[key].shape[1:])
                .reshape(B * T, *batch[key].shape[1:])
            )

    # ----- Encoding building blocks -----------------------------------------

    def _build_encodings(self, batch, rigids, x_cond):
        return {
            "aatype": batch["seqres"].long(),
            "x_cond": x_cond,
            "frames": rigids[:, 0],
            "mask": batch["mask"].float(),
        }

    def _shifted_input(self, target_disc):
        """[START, target_disc[0]+1, ..., target_disc[L*D - 2]+1].
        +1 makes room for the START token at embedding index 0."""
        Bt = target_disc.shape[0]
        return torch.cat(
            [
                torch.zeros(Bt, 1, dtype=torch.long, device=target_disc.device),
                target_disc[:, :-1] + 1,
            ],
            dim=1,
        )

    # ----- Per-channel discretization helpers --------------------------------

    def _discretize_target(self, target_3d: torch.Tensor) -> torch.Tensor:
        """target_3d: (Bt, L, D) → bin indices flat (Bt, L*D).

        Discretization is per-channel (channel dim is last in target_3d) so
        each of the 21 latent channels uses its own [min, max] range.
        """
        Bt, L, D = target_3d.shape
        idx = self.discretize.discretize(target_3d)  # (Bt, L, D)
        return idx.reshape(Bt, L * D)

    def _undiscretize(self, idx_flat: torch.Tensor, L: int) -> torch.Tensor:
        """idx_flat: (Bt, L*D) → continuous (Bt, L, D)."""
        Bt = idx_flat.shape[0]
        D = self.latent_dim
        idx = idx_flat.reshape(Bt, L, D)
        if (
            getattr(self.args, "use_empirical_offsets", False)
            and self.discretize.has_empirical_offsets()
        ):
            return self.discretize.undiscretize_with_empirical_offsets(idx)
        return self.discretize.undiscretize_with_noise(idx)

    # ----- Calibration -------------------------------------------------------

    @torch.no_grad()
    def calibrate_discretizer(
        self,
        loader,
        num_batches: int = 32,
        k: float = 4.0,
        verbose: bool = True,
    ) -> None:
        """Compute per-channel μ ± k·σ on the *target* latent and set bounds.

        Single-process. Intended to be called BEFORE `trainer.fit()` from the
        training script — i.e. before Lightning has wrapped the loader with a
        DistributedSampler and before NCCL is initialised. With an identical
        torch RNG seed across SLURM ranks, the dataloader's RandomSampler
        produces the same shuffle order, so each rank computes the same
        bounds independently — no cross-rank communication needed.
        """
        sum_x = torch.zeros(self.latent_dim, dtype=torch.float64)
        sum_x2 = torch.zeros(self.latent_dim, dtype=torch.float64)
        n = 0

        loader_iter = iter(loader)
        try:
            for _ in range(num_batches):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                batch = dict(batch)
                self._pair_with_tau(batch)
                rigids = self._build_rigids(batch)
                latents = self._build_latents(batch, rigids)
                target = latents[:, 1]                              # (Bt, L, 21)
                valid = batch["mask"].bool()                        # (Bt, L)
                x = target[valid].to(torch.float64)                 # (M, 21)
                if x.numel() == 0:
                    continue
                sum_x += x.sum(dim=0)
                sum_x2 += (x * x).sum(dim=0)
                n += x.shape[0]
        finally:
            # Explicitly tear down the iterator so DataLoader workers exit
            # cleanly before Lightning re-creates the iterator inside fit().
            del loader_iter

        if n < 2:
            if verbose:
                logger.warning("calibrate_discretizer: insufficient samples (%d), skipping", n)
            return

        mean = sum_x / n
        var = (sum_x2 / n - mean * mean).clamp(min=1e-12)
        std = torch.sqrt(var)
        mn = (mean - k * std).float()
        mx = (mean + k * std).float()
        self.discretize.set_range(mn, mx)
        self._calibrated = True

        if verbose:
            for c in range(self.latent_dim):
                logger.info(
                    "  channel %2d: μ=%+0.4f σ=%0.4f → range=[%+0.4f, %+0.4f] (bw=%0.5f)",
                    c, float(mean[c]), float(std[c]),
                    float(mn[c]), float(mx[c]),
                    float((mx[c] - mn[c]) / (self.discretize.num_bins - 1)),
                )

    def on_fit_start(self) -> None:
        """No-op if calibration was already done before fit() (recommended).
        Kept as a safety fallback — but bounds will then differ across DDP
        ranks; for production runs always call `calibrate_discretizer` from
        the training script before `trainer.fit(...)`.
        """
        if not getattr(self.args, "auto_discretize_range", False):
            return
        if self._calibrated:
            return
        logger.warning(
            "Discretizer was not pre-calibrated before fit(); falling back to "
            "in-fit calibration. Under DDP, ranks may compute slightly different "
            "bounds. Pre-calibrate from train.py main for full determinism."
        )
        try:
            loader = self.trainer.train_dataloader
        except Exception:
            return
        if loader is None:
            return
        self.calibrate_discretizer(
            loader,
            num_batches=int(getattr(self.args, "calibration_batches", 32)),
            k=float(getattr(self.args, "discretize_std_k", 4.0)),
            verbose=True,
        )

    # ----- Checkpointing: persist discretizer state -------------------------

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["discretize"] = {
            "num_bins": self.discretize.num_bins,
            "latent_dim": self.discretize.latent_dim,
            "min_val": self.discretize.min_val.detach().clone(),
            "max_val": self.discretize.max_val.detach().clone(),
            "calibrated": bool(self._calibrated),
        }

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        d = checkpoint.get("discretize", None)
        if d is None:
            return
        # Only restore if shape-compatible — guards against config drift.
        if (
            d.get("num_bins") == self.discretize.num_bins
            and d.get("latent_dim") == self.discretize.latent_dim
        ):
            self.discretize.set_range(d["min_val"], d["max_val"])
            self._calibrated = bool(d.get("calibrated", True))
            logger.info("Restored discretizer bounds from checkpoint (calibrated=%s)",
                        self._calibrated)

    # ----- Training step -----------------------------------------------------

    def general_step(self, batch, stage: str = "train"):
        batch = dict(batch)
        self._pair_with_tau(batch)
        rigids = self._build_rigids(batch)              # (Bt, 2, L)
        latents = self._build_latents(batch, rigids)    # (Bt, 2, L, 21)

        Bt, _, L, D = latents.shape
        target_3d = latents[:, 1]                       # (Bt, L, 21)
        x_cond = latents[:, 0]                          # (Bt, L, 21)

        target_disc = self._discretize_target(target_3d)        # (Bt, L*D)
        x_input = self._shifted_input(target_disc)              # (Bt, L*D)
        encodings = self._build_encodings(batch, rigids, x_cond)

        logits, _ = self.model(x_input, encodings=encodings)    # (Bt, L*D, V)

        # Cross-entropy: predicted bin distribution vs. ground-truth bin index,
        # per token, then masked-mean over valid residues' tokens.
        ce = F.cross_entropy(
            logits.reshape(-1, self.args.num_bins),
            target_disc.reshape(-1),
            reduction="none",
            label_smoothing=getattr(self.args, "label_smoothing", 0.0),
        ).reshape(Bt, L * D)

        coord_mask = batch["mask"].repeat_interleave(D, dim=1).float()
        loss = (ce * coord_mask).sum() / coord_mask.sum().clamp(min=1)

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ----- Likelihood --------------------------------------------------------

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        batch,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Exact log-likelihood of x_{t+τ} given x_t under the model.

        If samples were drawn with non-default `temperature`/`top_k`/`top_p`,
        passing the same values here gives the likelihood under the actual
        sampling distribution.

        Returns: (Bt,) summed log-prob across all unmasked tokens.
        """
        batch = dict(batch)
        self._pair_with_tau(batch)
        rigids = self._build_rigids(batch)
        latents = self._build_latents(batch, rigids)
        Bt, _, L, D = latents.shape

        target_3d = latents[:, 1]
        x_cond = latents[:, 0]

        target_disc = self._discretize_target(target_3d)
        x_input = self._shifted_input(target_disc)
        encodings = self._build_encodings(batch, rigids, x_cond)

        logits, _ = self.model(x_input, encodings=encodings)
        logits = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)

        log_probs = F.log_softmax(logits, dim=-1)
        log_p = log_probs.gather(-1, target_disc.unsqueeze(-1)).squeeze(-1)

        coord_mask = batch["mask"].repeat_interleave(D, dim=1).float()
        return (log_p * coord_mask).sum(dim=-1)

    # ----- Inference ---------------------------------------------------------

    @torch.no_grad()
    def inference(self, batch, num_steps: Optional[int] = None):
        """Autoregressive sample of x_{t+τ} given x_t. num_steps is unused."""
        if "trans_plus_tau" in batch:
            batch = dict(batch)
            self._pair_with_tau(batch)
        rigids = self._build_rigids(batch)
        latents = self._build_latents(batch, rigids)
        Bt, _, L, D = latents.shape

        x_cond = latents[:, 0]
        seqres = batch["seqres"].long()
        encodings = self._build_encodings(batch, rigids, x_cond)

        generated = self.model.generate(
            batch_size=Bt,
            num_residues=L,
            encodings=encodings,
            temperature=getattr(self.args, "ar_temperature", 1.0),
            top_k=getattr(self.args, "ar_top_k", None),
            top_p=getattr(self.args, "ar_top_p", None),
            device=self.device,
            use_cache=True,
        )

        cont = self._undiscretize(generated, L)         # (Bt, L, D)
        samples = cont.unsqueeze(1)                     # (Bt, 1, L, 21)
        return self._decode_samples(samples, rigids[:, 0:1], seqres)
