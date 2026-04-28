"""Causal autoregressive transformer over per-residue, per-channel tokens.

Treats each residue's 21-dim latent (7 frame offsets + 14 torsion sin/cos) as
21 discrete tokens, predicts them sequentially conditioned on a bidirectional
encoding of the starting structure, sequence, and prior-state latent.

Efficiency / quality features:
- F.scaled_dot_product_attention (Flash Attention compatible) everywhere.
- KV cache during generation; the encoder context is computed once and reused.
- RMSNorm + SwiGLU FFN (LLaMA-style decoder blocks).
- Cross-attention from causal tokens to bidirectional per-residue context, with
  QK-norm for attention stability, zero-initialised output projection, and a
  learnable tanh gate (starts at 0, opens during training).
- IPA + RoPE-MHA bidirectional encoder reuses existing OpenFold IPA.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vendored.openfold.ipa import InvariantPointAttention
from ..model.model import AttentionWithRoPE


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _make_ffn(dim: int, ffn_dim: int, use_swiglu: bool = True) -> nn.Module:
    if use_swiglu:
        # 2/3 sizing keeps param count comparable to a 1-projection FFN.
        return SwiGLU(dim, int(2 * ffn_dim / 3))
    return nn.Sequential(
        nn.Linear(dim, ffn_dim),
        nn.GELU(),
        nn.Linear(ffn_dim, dim),
    )


def _norm(use_rms: bool, dim: int) -> nn.Module:
    return RMSNorm(dim) if use_rms else nn.LayerNorm(dim)


def _sincos_pos(seq_len: int, dim: int, device) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32, device=device) / (dim / 2)
    omega = 1.0 / (10000 ** omega)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([out.sin(), out.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Bidirectional structure encoder (per-residue context)
# ---------------------------------------------------------------------------


class StructureEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, ipa_args,
                 use_rms_norm=True, use_swiglu=True):
        super().__init__()
        self.ipa_norm = _norm(use_rms_norm, embed_dim)
        self.ipa = InvariantPointAttention(**ipa_args)
        self.attn_norm = _norm(use_rms_norm, embed_dim)
        self.attn = AttentionWithRoPE(
            embed_dim, num_heads, add_bias_kv=True, use_rotary_embeddings=True,
        )
        self.ffn_norm = _norm(use_rms_norm, embed_dim)
        self.ffn = _make_ffn(embed_dim, ffn_dim, use_swiglu=use_swiglu)

    def forward(self, x, mask, frames):
        x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StructureEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ipa_args, latent_dim,
                 use_rms_norm=True, use_swiglu=True):
        super().__init__()
        self.aatype_emb = nn.Embedding(21, embed_dim)
        self.cond_proj = nn.Linear(latent_dim, embed_dim)
        self.layers = nn.ModuleList([
            StructureEncoderLayer(
                embed_dim, 4 * embed_dim, num_heads, ipa_args,
                use_rms_norm=use_rms_norm, use_swiglu=use_swiglu,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = _norm(use_rms_norm, embed_dim)

    def forward(self, aatype, x_cond, frames, mask):
        x = self.aatype_emb(aatype) + self.cond_proj(x_cond)
        for layer in self.layers:
            x = layer(x, mask, frames)
        return self.out_norm(x)


# ---------------------------------------------------------------------------
# Causal decoder
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Causal MHA with KV cache via F.scaled_dot_product_attention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            new_cache = (k, v)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn = attn.transpose(1, 2).reshape(B, T, C)
        return self.out(attn), new_cache


class CrossAttention(nn.Module):
    """Cross-attention (decoder tokens → encoder context) with optional QK-norm."""

    def __init__(self, embed_dim, num_heads, qk_norm: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, context, context_mask=None):
        B, T, C = x.shape
        L = context.shape[1]
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv_proj(context)
            .reshape(B, L, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class CausalDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        num_heads,
        use_cross_attn: bool = True,
        cross_attn_qk_norm: bool = True,
        cross_attn_gate: bool = True,
        use_rms_norm: bool = True,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.attn_norm = _norm(use_rms_norm, embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        if use_cross_attn:
            self.cross_norm = _norm(use_rms_norm, embed_dim)
            self.cross_attn = CrossAttention(embed_dim, num_heads, qk_norm=cross_attn_qk_norm)
            self.cross_gate = nn.Parameter(torch.zeros(1)) if cross_attn_gate else None
        else:
            self.cross_attn = None
            self.cross_gate = None

        self.ffn_norm = _norm(use_rms_norm, embed_dim)
        self.ffn = _make_ffn(embed_dim, ffn_dim, use_swiglu=use_swiglu)

    def forward(self, x, kv_cache=None, context=None, context_mask=None):
        h, new_kv = self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + h

        if self.use_cross_attn and context is not None:
            cross = self.cross_attn(self.cross_norm(x), context, context_mask)
            if self.cross_gate is not None:
                cross = torch.tanh(self.cross_gate) * cross
            x = x + cross

        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class CausalARModel(nn.Module):
    """Per-residue 21-channel autoregressive model.

    Sequence length per example = num_residues * NUM_CHANNELS, residue-major.
    """

    NUM_CHANNELS = 21

    def __init__(self, args, latent_dim: int = 21):
        super().__init__()
        assert latent_dim == self.NUM_CHANNELS
        self.args = args
        self.embed_dim = args.embed_dim
        self.num_bins = args.num_bins

        use_rms_norm = getattr(args, "use_rms_norm", True)
        use_swiglu = getattr(args, "use_swiglu", True)
        use_cross_attn = getattr(args, "use_cross_attention", True)
        cross_qk_norm = getattr(args, "cross_attn_qk_norm", True)
        cross_gate = getattr(args, "cross_attn_gate", True)

        self.use_cross_attn = use_cross_attn

        # Token embedding: index 0 = START; indices 1..num_bins = bin labels 0..num_bins-1.
        self.token_emb = nn.Embedding(self.num_bins + 1, self.embed_dim)
        self.channel_emb = nn.Embedding(self.NUM_CHANNELS, self.embed_dim)

        max_seq = args.crop * self.NUM_CHANNELS
        self.register_buffer(
            "seq_pos_embed",
            _sincos_pos(max_seq, self.embed_dim, "cpu"),
            persistent=False,
        )

        ipa_args = dict(
            c_s=self.embed_dim,
            c_z=0,
            c_hidden=args.ipa_head_dim,
            no_heads=args.ipa_heads,
            no_qk_points=args.ipa_qk,
            no_v_points=args.ipa_v,
        )
        self.encoder = StructureEncoder(
            embed_dim=self.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.mha_heads,
            ipa_args=ipa_args,
            latent_dim=latent_dim,
            use_rms_norm=use_rms_norm,
            use_swiglu=use_swiglu,
        )

        self.decoder = nn.ModuleList([
            CausalDecoderLayer(
                self.embed_dim,
                4 * self.embed_dim,
                args.mha_heads,
                use_cross_attn=use_cross_attn,
                cross_attn_qk_norm=cross_qk_norm,
                cross_attn_gate=cross_gate,
                use_rms_norm=use_rms_norm,
                use_swiglu=use_swiglu,
            )
            for _ in range(args.num_layers)
        ])

        self.out_norm = _norm(use_rms_norm, self.embed_dim)
        self.out_head = nn.Linear(self.embed_dim, self.num_bins, bias=False)

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic)

        # IPA stability: zero output projection so the residual block starts as identity.
        for layer in self.encoder.layers:
            nn.init.zeros_(layer.ipa.linear_out.weight)
            nn.init.zeros_(layer.ipa.linear_out.bias)

        # Cross-attention onboarding: cross_gate=0 alone is sufficient. Zero-initialising
        # both the gate AND cross_attn.out.weight produces a gradient deadlock —
        # ∂L/∂cross_gate ∝ out_proj(...) = 0, so the gate never moves off zero.

    # ----- Conditioning ------------------------------------------------------

    def encode_context(self, aatype, x_cond, frames, mask):
        return self.encoder(aatype, x_cond, frames, mask)

    def _build_decoder_input(self, x_input, context, offset: int = 0):
        B, T_in = x_input.shape
        C = self.NUM_CHANNELS
        device = x_input.device

        positions = torch.arange(offset, offset + T_in, device=device)
        ch_idx = positions % C

        h = self.token_emb(x_input)
        h = h + self.channel_emb(ch_idx).unsqueeze(0)
        h = h + self.seq_pos_embed[offset:offset + T_in].to(device).unsqueeze(0)

        if not self.use_cross_attn:
            # Fall back: additive per-residue context.
            res_idx = positions // C
            h = h + context[:, res_idx]

        return h

    # ----- Forward / generate ------------------------------------------------

    def forward(
        self,
        x_input: torch.Tensor,
        encodings: Optional[dict] = None,
        node_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_caches: Optional[list] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ):
        if context is None:
            assert encodings is not None
            mask = encodings["mask"] if node_mask is None else node_mask
            context = self.encode_context(
                encodings["aatype"], encodings["x_cond"], encodings["frames"], mask,
            )
            if context_mask is None:
                context_mask = mask

        if kv_caches is not None and kv_caches[0] is not None:
            offset = kv_caches[0][0].shape[2]
        else:
            offset = 0

        h = self._build_decoder_input(x_input, context, offset=offset)

        new_caches: list = []
        for i, block in enumerate(self.decoder):
            past = kv_caches[i] if (kv_caches is not None and use_cache) else None
            h, new_kv = block(
                h,
                kv_cache=past,
                context=context if self.use_cross_attn else None,
                context_mask=context_mask if self.use_cross_attn else None,
            )
            if use_cache:
                new_caches.append(new_kv)

        h = self.out_norm(h)
        logits = self.out_head(h)
        return logits, (new_caches if use_cache else None, context, context_mask)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_residues: int,
        encodings: dict,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        L = num_residues
        C = self.NUM_CHANNELS
        S = L * C
        device = device if device is not None else next(self.parameters()).device

        context = self.encode_context(
            encodings["aatype"], encodings["x_cond"], encodings["frames"], encodings["mask"],
        )
        context_mask = encodings["mask"]

        out = torch.zeros(batch_size, S, dtype=torch.long, device=device)
        kv_caches: Optional[list] = [None] * len(self.decoder) if use_cache else None
        current = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # START

        for step in range(S):
            logits, (new_caches, _, _) = self.forward(
                current,
                use_cache=use_cache,
                kv_caches=kv_caches,
                context=context,
                context_mask=context_mask,
            )
            if use_cache:
                kv_caches = new_caches
            next_tok = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            out[:, step] = next_tok
            current = (next_tok + 1).unsqueeze(1)

        return out

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=None, top_p=None):
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(
                logits < v[:, [-1]], torch.full_like(logits, float("-inf")), logits,
            )

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            mask = cum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
