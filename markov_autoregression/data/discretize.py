"""Discretization for autoregressive coordinate generation (PixelCNN-style).

A continuous value x ∈ [min_val, max_val] is mapped to one of `num_bins` evenly
spaced *anchor points*:

    bin_center(i) = min_val + i * (max_val - min_val) / (num_bins - 1),  i ∈ [0, N)

Discretize: round to the nearest anchor.
Undiscretize: return the anchor (bin center).
Undiscretize-with-noise: anchor + uniform(-bw/2, bw/2), where
    bw = (max_val - min_val) / (num_bins - 1)
is the spacing between consecutive anchors.

Per-channel ranges are supported by passing 1-D tensors of shape (latent_dim,)
for `min_val` / `max_val`. This matters because protein-state latents have
heterogeneous channels (quaternions ≈ [-1, 1] vs translations in Å).

The scheme matches the original (PixelCNN-style) idea while using a single
consistent convention across discretize / undiscretize / empirical offsets.
"""

from typing import List, Optional, Union

import torch


def _as_1d_or_scalar(v, expected_dim: Optional[int]) -> torch.Tensor:
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.dim() == 0:
        return t
    if t.dim() == 1:
        if expected_dim is not None:
            assert t.shape[0] == expected_dim, (
                f"per-channel min/max length {t.shape[0]} != expected {expected_dim}"
            )
        return t
    raise ValueError(f"min_val/max_val must be scalar or 1-D, got shape {tuple(t.shape)}")


class DiscretizeTransform:
    """Round-to-N-anchor discretizer with optional per-channel ranges."""

    def __init__(
        self,
        num_bins: int = 8192,
        min_val: Union[float, torch.Tensor] = -4.0,
        max_val: Union[float, torch.Tensor] = 4.0,
        latent_dim: Optional[int] = None,
    ):
        assert num_bins >= 2, "num_bins must be >= 2"
        self.num_bins = int(num_bins)
        self.latent_dim = latent_dim
        self.min_val = _as_1d_or_scalar(min_val, latent_dim)
        self.max_val = _as_1d_or_scalar(max_val, latent_dim)
        assert torch.all(self.max_val > self.min_val), "max_val must exceed min_val"
        # Spacing between consecutive anchor points.
        self.bin_width = (self.max_val - self.min_val) / (self.num_bins - 1)
        # List of within-bin offsets per bin, each in [-0.5, +0.5] (deviation
        # from anchor as a fraction of bin_width). One list element per bin.
        self._empirical_offsets: Optional[List[torch.Tensor]] = None

    # ----- range update (from calibration) ----------------------------------

    def set_range(
        self,
        min_val: Union[float, torch.Tensor],
        max_val: Union[float, torch.Tensor],
    ) -> None:
        self.min_val = _as_1d_or_scalar(min_val, self.latent_dim)
        self.max_val = _as_1d_or_scalar(max_val, self.latent_dim)
        assert torch.all(self.max_val > self.min_val), "max_val must exceed min_val"
        self.bin_width = (self.max_val - self.min_val) / (self.num_bins - 1)
        self._empirical_offsets = None  # invalidated

    # ----- core transforms --------------------------------------------------

    def _to(self, x: torch.Tensor):
        return (
            self.min_val.to(x.device),
            self.max_val.to(x.device),
            self.bin_width.to(x.device),
        )

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous → bin index in [0, num_bins).

        Channel dim must be the LAST dim of x when per-channel min/max is used
        (broadcasting along the trailing axis).
        """
        mn, mx, _ = self._to(x)
        x_clipped = torch.clamp(x, mn, mx)
        norm = (x_clipped - mn) / (mx - mn)             # in [0, 1]
        idx = (norm * (self.num_bins - 1)).round().long()
        return idx.clamp(0, self.num_bins - 1)

    def undiscretize(self, idx: torch.Tensor) -> torch.Tensor:
        """Bin index → bin center (anchor point)."""
        mn, _, bw = self._to(idx)
        return mn + idx.float() * bw

    def undiscretize_with_noise(
        self, idx: torch.Tensor, noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Anchor + uniform(-bw/2, +bw/2) * noise_scale."""
        _, _, bw = self._to(idx)
        center = self.undiscretize(idx)
        noise = (torch.rand_like(center) - 0.5) * bw * noise_scale
        return center + noise

    # ----- empirical within-bin offsets -------------------------------------

    def has_empirical_offsets(self) -> bool:
        return self._empirical_offsets is not None

    def compute_empirical_offsets(
        self, samples: torch.Tensor, max_samples_per_bin: int = 10_000,
    ) -> None:
        """Bin training data and store, per bin, the within-bin SIGNED offsets
        (in [-0.5, +0.5] as fractions of bin_width). Used at inference to
        sample more realistic dequantization than uniform noise.

        `samples` shape: (..., D) when per-channel; (...) when scalar range.
        Channel dim must be last.
        """
        if self.latent_dim is not None:
            x = samples.reshape(-1, self.latent_dim)
        else:
            x = samples.reshape(-1, 1)

        idx = self.discretize(x)                         # (N, D) or (N, 1)
        center = self.undiscretize(idx)                  # same shape
        bw = self.bin_width.to(x.device)
        # Signed fractional offsets in [-0.5, +0.5].
        u = ((x - center) / bw).clamp(-0.5, 0.5)

        idx_flat = idx.reshape(-1).cpu()
        u_flat = u.reshape(-1).cpu().float()

        per_bin: List[torch.Tensor] = []
        for b in range(self.num_bins):
            mask = idx_flat == b
            offsets = u_flat[mask]
            if offsets.numel() > max_samples_per_bin:
                perm = torch.randperm(offsets.numel())[:max_samples_per_bin]
                offsets = offsets[perm]
            if offsets.numel() == 0:
                offsets = torch.zeros(1)  # default = anchor center
            per_bin.append(offsets)
        self._empirical_offsets = per_bin

    def undiscretize_with_empirical_offsets(self, idx: torch.Tensor) -> torch.Tensor:
        """Sample within-bin offsets from the empirical distribution, then
        return anchor + sampled_u * bin_width.

        idx must have channel dim as its last axis when per-channel min/max is
        in use (so that bin_width / min_val broadcast properly).
        """
        assert self._empirical_offsets is not None, (
            "compute_empirical_offsets must be called first"
        )

        device = idx.device
        shape = idx.shape
        flat = idx.reshape(-1)

        # Sample per-bin offsets, then reshape so channel-dim broadcasting works.
        sampled_flat = torch.zeros(flat.shape, device=device, dtype=torch.float32)
        for b, offsets in enumerate(self._empirical_offsets):
            mask = flat == b
            n = int(mask.sum().item())
            if n == 0:
                continue
            pool = offsets.to(device=device, dtype=torch.float32)
            pick = torch.randint(0, pool.numel(), (n,), device=device)
            sampled_flat[mask] = pool[pick]
        sampled = sampled_flat.reshape(shape)

        center = self.undiscretize(idx)        # respects per-channel min/max
        bw = self.bin_width.to(device)
        return center + sampled * bw
