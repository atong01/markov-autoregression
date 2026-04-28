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
