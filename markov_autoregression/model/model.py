# Adapted from https://github.com/bjing2016/mdgen
# DiT layers adapted from https://github.com/facebookresearch/DiT

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vendored.openfold.ipa import InvariantPointAttention
from ..vendored.mha import MultiheadAttention


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


TIME_MULTIPLIER = 100.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    """Final adaptive-LayerNorm + linear projection layer."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=1 - mask)
        x = x.transpose(0, 1)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MarSModel(nn.Module):
    def __init__(self, args, latent_dim):
        super().__init__()
        self.args = args
        self.use_ipa = not getattr(args, "euclidean", False)

        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim)
        self.cond_to_emb = nn.Linear(latent_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)

        ipa_args = {
            "c_s": args.embed_dim,
            "c_z": 0,
            "c_hidden": args.ipa_head_dim,
            "no_heads": args.ipa_heads,
            "no_qk_points": args.ipa_qk,
            "no_v_points": args.ipa_v,
        }

        self.aatype_to_emb = nn.Embedding(21, args.embed_dim)
        self.ipa_layers = nn.ModuleList(
            [
                IPALayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    ipa_args=ipa_args,
                    use_ipa=self.use_ipa,
                )
                for _ in range(args.num_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                MarSLayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                )
                for _ in range(args.num_layers)
            ]
        )

        self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim)

        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, args.crop, args.embed_dim),
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_ipa:
            for block in self.ipa_layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.abs_pos_emb:
            pos_embed = _sincos_pos_embed(
                self.pos_embed.shape[-1], np.arange(self.args.crop)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.emb_to_latent.linear.weight, 0)
        nn.init.constant_(self.emb_to_latent.linear.bias, 0)

    def run_ipa(self, t, mask, start_frames, aatype):
        x = self.aatype_to_emb(aatype)

        for layer in self.ipa_layers:
            x = layer(x, t, mask, frames=start_frames)

        return x

    def forward(
        self,
        x,
        t,
        mask,
        start_frames=None,
        x_cond=None,
        x_cond_mask=None,
        aatype=None,
    ):
        x = self.latent_to_emb(x)
        if self.args.abs_pos_emb:
            x = x + self.pos_embed

        if x_cond is not None:
            x = (
                x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)
            )

        t = self.t_embedder(t * TIME_MULTIPLIER)[:, None]

        x = (
            x
            + self.run_ipa(
                t[:, 0], mask[:, 0], start_frames, aatype
            )[:, None]
        )

        for layer in self.layers:
            x = layer(x, t, mask)

        latent = self.emb_to_latent(x, t)
        return latent


class MarSMeanFlowModel(MarSModel):
    """MarSModel that conditions on two times (r, t) instead of one.

    Used for mean-flow / flow-map training. The conditioning vector is the sum
    of independent embeddings of r and t, fed everywhere t is fed in MarSModel.
    """

    def __init__(self, args, latent_dim):
        super().__init__(args, latent_dim)
        self.r_embedder = TimestepEmbedder(args.embed_dim)
        nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

    def forward(
        self,
        x,
        r,
        t,
        mask,
        start_frames=None,
        x_cond=None,
        x_cond_mask=None,
        aatype=None,
    ):
        x = self.latent_to_emb(x)
        if self.args.abs_pos_emb:
            x = x + self.pos_embed

        if x_cond is not None:
            x = x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)

        c = (
            self.t_embedder(t * TIME_MULTIPLIER)
            + self.r_embedder(r * TIME_MULTIPLIER)
        )[:, None]

        x = x + self.run_ipa(c[:, 0], mask[:, 0], start_frames, aatype)[:, None]

        for layer in self.layers:
            x = layer(x, c, mask)

        return self.emb_to_latent(x, c)


class IPALayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, ipa_args=None, use_ipa=True):
        super().__init__()
        self.use_ipa = use_ipa

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )

        if self.use_ipa:
            self.ipa_norm = nn.LayerNorm(embed_dim)
            self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = AttentionWithRoPE(
            embed_dim, mha_heads, add_bias_kv=True, use_rotary_embeddings=True,
        )

        self.mha_layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

        self.final_layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=-1)
        )
        if self.use_ipa:
            x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class MarSLayer(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, mha_heads):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 9 * embed_dim, bias=True)
        )

        self.mha_t = AttentionWithRoPE(
            embed_dim, mha_heads, add_bias_kv=True, use_rotary_embeddings=True,
        )

        self.mha_l = AttentionWithRoPE(
            embed_dim, mha_heads, add_bias_kv=True, use_rotary_embeddings=True,
        )

        self.mha_layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

        self.final_layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None):
        B, T, L, C = x.shape
        (
            shift_msa_l,
            scale_msa_l,
            gate_msa_l,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(t).chunk(9, dim=-1)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(
            x.reshape(B * T, L, C),
            mask=mask.reshape(B * T, L),
        ).reshape(B, T, L, C)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        x = (
            self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C),
                mask=mask.transpose(1, 2).reshape(B * L, T),
            )
            .reshape(B, L, T, C)
            .transpose(1, 2)
        )
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x
