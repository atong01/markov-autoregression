# Adapted from https://github.com/willisma/SiT/

import math

import torch
from torchdiffeq import odeint


# ---------------------------------------------------------------------------
# Flow-matching path
# ---------------------------------------------------------------------------


def _expand_t_like_x(t, x):
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


class GVPCPlan:
    """Geodesic Velocity Preserving Conditional Plan (trigonometric interpolation)."""

    def _alpha(self, t):
        return torch.sin(t * math.pi / 2), math.pi / 2 * torch.cos(t * math.pi / 2)

    def _sigma(self, t):
        return torch.cos(t * math.pi / 2), -math.pi / 2 * torch.sin(t * math.pi / 2)

    def plan(self, t, x0, x1):
        t_x = _expand_t_like_x(t, x1)
        alpha_t, d_alpha_t = self._alpha(t_x)
        sigma_t, d_sigma_t = self._sigma(t_x)
        xt = alpha_t * x1 + sigma_t * x0
        ut = d_alpha_t * x1 + d_sigma_t * x0
        return t, xt, ut


# ---------------------------------------------------------------------------
# ODE integrator
# ---------------------------------------------------------------------------


class _ODE:
    """Thin wrapper around torchdiffeq.odeint."""

    def __init__(self, drift, *, t0, t1, num_steps, atol, rtol):
        self.drift = drift
        self.t = torch.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol

    def sample(self, x, model, **model_kwargs):
        device = x.device

        def _fn(t, x):
            t = torch.ones(x.size(0), device=device) * t
            return self.drift(x, t, model, **model_kwargs)

        return odeint(
            _fn, x, self.t.to(device),
            method="dopri5", atol=[self.atol], rtol=[self.rtol],
        )


# ---------------------------------------------------------------------------
# Transport & Sampler
# ---------------------------------------------------------------------------


def _mean_flat(x, mask):
    return torch.sum(x * mask, dim=list(range(1, len(x.size())))) / torch.sum(
        mask, dim=list(range(1, len(x.size())))
    )


class Transport:

    def __init__(self):
        self.path_sampler = GVPCPlan()

    def training_losses(self, model, x1, mask=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0],)).to(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        xt = xt[:, 1, :, :].unsqueeze(1)
        ut = ut[:, 1, :, :].unsqueeze(1)
        model_kwargs["mask"] = model_kwargs["mask"][:, 1, :].unsqueeze(1)
        model_kwargs["x_cond"] = model_kwargs["x_cond"][:, 0, :, :].unsqueeze(1)
        model_kwargs["x_cond_mask"] = model_kwargs["x_cond_mask"][:, 0, :].unsqueeze(1)

        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        return {
            "t": t,
            "pred": model_output,
            "loss": _mean_flat(((model_output - ut) ** 2), mask),
        }

    @staticmethod
    def _drift(x, t, model, **model_kwargs):
        out = model(x, t, **model_kwargs)
        assert out.shape == x.shape
        return out


class Sampler:

    def __init__(self, transport):
        self.drift = transport._drift

    def sample_ode(self, *, num_steps=50, atol=1e-6, rtol=1e-3):
        integrator = _ODE(
            drift=self.drift, t0=0, t1=1,
            num_steps=num_steps, atol=atol, rtol=rtol,
        )
        return integrator.sample


# ---------------------------------------------------------------------------
# Mean-flow (flow-map) training & sampling
# ---------------------------------------------------------------------------
#
# MeanFlow learns u_θ(z, r, t): the *average* drift from time r to time t along
# a linear interpolant z = t·ε + (1−t)·x with ε ~ N(0, I). Time direction is
# opposite to the GVP-C flow-matching path used elsewhere in this file:
#
#   t = 0 → data,    t = 1 → noise.
#
# Training target (stop-grad):
#
#   v       = ε − x                        (instantaneous velocity at t)
#   u, ∂u   = JVP of u_θ at (z, r, t) along (v, 0, 1)
#   u_tgt   = v − (t − r) · ∂u             (detached)
#   loss    = ‖u − u_tgt‖²
#
# Sampling: Euler steps in *decreasing* t from 1 to 0 with mean-velocity step
# z ← z − (t_i − t_{i+1})·u_θ(z, t_{i+1}, t_i).


def _expand(t, x):
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


class MeanFlowTransport:
    """Mean-flow training objective (meanflow_loss only)."""

    def __init__(
        self,
        neq_frac: float = 0.25,
        use_lognormal: bool = False,
        uniform_temporal_sampling: bool = True,
    ):
        self.neq_frac = neq_frac
        self.use_lognormal = use_lognormal
        self.uniform_temporal_sampling = uniform_temporal_sampling

    def _sample_rt(self, B: int, device):
        if self.uniform_temporal_sampling:
            t = torch.rand(B, device=device)
            r = torch.rand(B, device=device) * t
            return r, t

        if self.use_lognormal:
            dist = torch.distributions.LogNormal(-0.4, 1.0)
            t = dist.sample((B,)).to(device)
            r = dist.sample((B,)).to(device)
        else:
            t = torch.rand(B, device=device)
            r = torch.rand(B, device=device)

        # neq_frac of the batch keeps r ≠ t; the remainder is collapsed to r = t
        # (boundary regression: u(z, t, t) = v).
        eq_mask = torch.rand(B, device=device) > self.neq_frac
        r = torch.where(eq_mask, t, r)

        # Enforce r ≤ t.
        swap = r > t
        r, t = torch.where(swap, t, r), torch.where(swap, r, t)
        return r, t

    def training_losses(self, model, x1, mask=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # Mirror Transport.training_losses' pair slicing: target = state at t+τ,
        # condition = state at t.
        x = x1[:, 1, :, :].unsqueeze(1)
        target_mask = mask[:, 1, :, :].unsqueeze(1) if mask is not None else None
        kwargs = dict(model_kwargs)
        kwargs["mask"] = kwargs["mask"][:, 1, :].unsqueeze(1)
        kwargs["x_cond"] = kwargs["x_cond"][:, 0, :, :].unsqueeze(1)
        kwargs["x_cond_mask"] = kwargs["x_cond_mask"][:, 0, :].unsqueeze(1)

        B = x.shape[0]
        r, t = self._sample_rt(B, x.device)

        eps = torch.randn_like(x)
        t_x = _expand(t, x)
        z = t_x * eps + (1 - t_x) * x
        v = eps - x

        def _fn(z_, r_, t_):
            return model(z_, r_, t_, **kwargs)

        u, dudt = torch.func.jvp(
            _fn,
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        diff = _expand(t - r, x)
        u_target = (v - diff * dudt).detach()

        loss = _mean_flat((u - u_target) ** 2, target_mask)
        return {"r": r, "t": t, "pred": u, "loss": loss}


class MeanFlowSampler:
    """Multi-step mean-flow Euler sampler from t=1 (noise) to t=0 (data)."""

    def __init__(self, n_steps: int = 8):
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(self, z, model, n_steps: int = None, **model_kwargs):
        n = n_steps if n_steps is not None else self.n_steps
        ts = torch.linspace(1.0, 0.0, n + 1, device=z.device, dtype=torch.float32)
        B = z.shape[0]
        for i in range(n):
            t = ts[i].expand(B)
            r = ts[i + 1].expand(B)
            u = model(z, r, t, **model_kwargs)
            z = z - _expand(t - r, z) * u
        return z
