# Adapted from https://github.com/bjing2016/mdgen

import logging
from functools import partial

import pytorch_lightning as pl
import torch

from ..data.geometry import frames_torsions_to_atom14
from ..vendored.openfold.rigid_utils import Rigid, Rotation
from ..vendored.openfold.tensor_utils import tensor_tree_map
from ..vendored.ema import ExponentialMovingAverage
from ..transport import Transport, Sampler
from ..utils import get_offsets
from .model import MarSModel

logger = logging.getLogger(__name__)


class BaseModule(pl.LightningModule):
    normalize_torsions = False

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.latent_dim = 21
        self.model = MarSModel(args, self.latent_dim)
        self.transport = Transport()
        self.sampler = Sampler(self.transport)

        if getattr(args, "ema", False):
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay,
            )
            self.cached_weights = None

    # -- EMA helpers ----------------------------------------------------------

    def load_ema_weights(self):
        logger.info("Loading EMA weights")
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        logger.info("Restoring cached weights")
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if getattr(self.args, "ema", False):
            self.ema.update(self.model)

    def on_load_checkpoint(self, checkpoint):
        if getattr(self.args, "ema", False):
            logger.info("Loading EMA state dict")
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if getattr(self.args, "ema", False):
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    # -- PL hooks -------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if getattr(self.args, "ema", False):
            if self.ema.device != self.device:
                self.ema.to(self.device)
        return self.general_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        if getattr(self.args, "ema", False):
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()
        self.general_step(batch, stage="val")

    def on_validation_epoch_end(self):
        if getattr(self.args, "ema", False):
            self.restore_cached_weights()

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
        )

    # -- Shared building blocks -----------------------------------------------

    def _build_rigids(self, batch):
        return Rigid(trans=batch["trans"], rots=Rotation(rot_mats=batch["rots"]))

    def _build_latents(self, batch, rigids):
        B, T, L = rigids.shape
        offsets = get_offsets(rigids[:, 0:1], rigids)
        offsets[..., :4] *= torch.where(offsets[:, :, :, 0:1] < 0, -1, 1)
        torsions = batch["torsions"].view(B, T, L, 14)
        return torch.cat([offsets, torsions], -1)

    def _build_conditioning(self, latents):
        """First-frame conditioning: expose latents[:, 0] to the model."""
        x_cond = torch.zeros_like(latents)
        x_cond[:, 0] = latents[:, 0]
        B, T, L, _ = latents.shape
        x_cond_mask = torch.zeros(B, T, L, dtype=int, device=latents.device)
        x_cond_mask[:, 0] = 1
        return x_cond, x_cond_mask

    def _build_model_kwargs(self, batch, rigids, latents):
        B, T, L = rigids.shape
        x_cond, x_cond_mask = self._build_conditioning(latents)
        return {
            "start_frames": rigids[:, 0],
            "mask": batch["mask"].unsqueeze(1).expand(-1, T, -1),
            "aatype": batch["seqres"],
            "x_cond": x_cond,
            "x_cond_mask": x_cond_mask,
        }

    def _decode_samples(self, samples, rigids, seqres):
        """Decode ODE output (offsets + torsions) into atom14 coordinates."""
        B, T, L = rigids.shape
        offsets = samples[..., :7]
        torsions = samples[..., 7:21].reshape(B, T, L, 7, 2)
        if self.normalize_torsions:
            torsions = torsions / torch.linalg.norm(torsions, dim=-1, keepdims=True)
        frames = rigids[:, 0:1].compose(
            Rigid.from_tensor_7(offsets, normalize_quats=True)
        )
        atom14 = frames_torsions_to_atom14(
            frames, torsions, seqres[:, None].expand(B, T, L)
        )
        return atom14, seqres[:, None].expand(B, T, L)

    # -- Inference (shared) ---------------------------------------------------

    def inference(self, batch, num_steps=50):
        rigids = self._build_rigids(batch)
        B, T, L = rigids.shape
        latents = self._build_latents(batch, rigids)
        model_kwargs = self._build_model_kwargs(batch, rigids, latents)

        zs = torch.randn(B, T, L, self.latent_dim, device=self.device)
        sample_fn = self.sampler.sample_ode(num_steps=num_steps)
        samples = sample_fn(zs, partial(self.model.forward, **model_kwargs))[-1]

        return self._decode_samples(samples, rigids, batch["seqres"])


class MDGenModule(BaseModule):
    """MDGen baseline module (inference only)."""
    normalize_torsions = True


class MarSModule(BaseModule):

    def _pair_with_tau(self, batch):
        """Stack (x_t, x_{t+tau}) pairs and tile per-protein fields to match."""
        for key in ["trans", "rots", "torsions"]:
            B, T = batch[key].shape[:2]
            rest = batch[key].shape[2:]
            batch[key] = torch.stack(
                [batch[key].reshape(B * T, *rest),
                 batch[key + "_plus_tau"].reshape(B * T, *rest)],
                dim=1,
            )
        for key in ["torsion_mask", "mask", "seqres"]:
            batch[key] = (
                batch[key].unsqueeze(1).expand(B, T, *batch[key].shape[1:])
                .reshape(B * T, *batch[key].shape[1:])
            )

    def _build_loss_mask(self, batch, rigids):
        B, T, L = rigids.shape
        frame_loss_mask = batch["mask"].unsqueeze(-1).expand(-1, -1, 7)
        torsion_loss_mask = (
            batch["torsion_mask"].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, L, 14)
        )
        loss_mask = torch.cat([frame_loss_mask, torsion_loss_mask], -1)
        return loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

    def general_step(self, batch, stage="train"):
        self._pair_with_tau(batch)
        rigids = self._build_rigids(batch)
        latents = self._build_latents(batch, rigids)
        loss_mask = self._build_loss_mask(batch, rigids)
        model_kwargs = self._build_model_kwargs(batch, rigids, latents)

        out_dict = self.transport.training_losses(
            model=self.model, x1=latents, mask=loss_mask,
            model_kwargs=model_kwargs,
        )
        loss = out_dict["loss"].mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss
