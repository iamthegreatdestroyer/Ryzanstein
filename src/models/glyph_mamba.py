"""
Mamba-Glyph Fusion — Innovation #2
====================================

Architecture insight (from Ryot-updates.md):

    "Instead of Mamba's selective mechanism working on raw embeddings, let it
    work on glyph coordinates in a learned manifold. The SSM's 'selectivity'
    becomes 'which glyph subspace is relevant?' rather than 'which tokens
    matter?' Result: sub-linear attention equivalent that never materializes
    a full attention matrix."

Standard Mamba selective mechanism:
    B_t, C_t, Δ_t = projections of x_t  ∈ ℝ^d_model

Glyph-Mamba selective mechanism:
    B_t, C_t, Δ_t = projections of g_t  ∈ ℝ^d_glyph    (d_glyph << d_model)

Where g_t is the glyph coordinate — a learned embedding of the Σ-glyph
primitive that token t maps to. Because d_glyph << d_model, the selectivity
parameters are derived from a compressed, semantically meaningful signal
rather than the full token embedding space.

Complexity comparison (seq length L, d_model, d_state, d_glyph):
  Standard attention:  O(L² × d_model)
  Standard Mamba SSM:  O(L × d_model × d_state)
  Glyph-Mamba SSM:     O(L × d_glyph × d_state)  ← selectivity is sub-model-dim

Components
----------
GlyphCoordinateEmbedding
    token_ids → primitive_ids → learned d_glyph-dim coordinate vectors
    Works with or without sigmalang (falls back to token_id % 256 bucketing)

selective_scan_sequential(x, Δ, A, B, C, D)
    Pure-PyTorch ZOH-discretized SSM recurrence (O(L) sequential scan)
    Production: swap for mamba_ssm's CUDA parallel scan for 10-100× speedup

GlyphSSMLayer
    Core innovation: single SSM layer with glyph-driven selectivity
    Drop-in attention head replacement: same (B, L, d_model) → (B, L, d_model)

GlyphMambaBlock
    Full Mamba-style block: in_proj → conv1d → GlyphSSMLayer → gate → out_proj
    Residual-wrapped, layer-normed, plug into any Transformer stack

GlyphMambaModel
    Minimal end-to-end model using N GlyphMambaBlocks for benchmarking
    Not a trained model — use for architecture validation and latency benchmarks
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Σ-Glyph coordinate dimension (learned embedding over 256 primitives)
DEFAULT_D_GLYPH = 32
# SSM state dimension (Mamba paper default)
DEFAULT_D_STATE = 16
# Local conv kernel width (Mamba paper default)
DEFAULT_D_CONV = 4
# Number of primitive IDs in the glyph alphabet
N_PRIMITIVES = 256


# ---------------------------------------------------------------------------
# Glyph Coordinate Embedding
# ---------------------------------------------------------------------------

class GlyphCoordinateEmbedding(nn.Module):
    """
    Maps token IDs to learned d_glyph-dimensional glyph coordinate vectors.

    Each token ID is first mapped to a glyph primitive ID (0-255), then looked
    up in a learned embedding table of shape [N_PRIMITIVES, d_glyph].

    Because many tokens share the same primitive (semantic deduplication),
    the embedding table has far fewer rows than a standard vocab embedding —
    and crucially, the SSM selectivity network operates on d_glyph dims
    rather than d_model dims.

    Token → primitive mapping (sigmalang-free fallback):
        token_id % 256   (uniform bucketing into 256 primitives)

    With sigmalang:
        tier-0 (0-15):   direct primitive_id = token_id
        tier-1 (16-127): primitive_id = 0x10 + (token_id - 16)
        tier-2 (128+):   primitive_id = 0x80 + ((token_id - 128) % 128)
    """

    def __init__(self, d_glyph: int = DEFAULT_D_GLYPH, use_sigmalang: bool = True):
        super().__init__()
        self.d_glyph = d_glyph
        self.embed = nn.Embedding(N_PRIMITIVES, d_glyph)
        # Small init so glyph coords start near zero (they're a *bias* on selectivity)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Try to load sigmalang mapper; fall back gracefully
        self._mapper = None
        if use_sigmalang:
            try:
                import sys
                from pathlib import Path
                _sl_root = Path(__file__).parent.parent.parent.parent / "sigmalang"
                if _sl_root.exists():
                    sys.path.insert(0, str(_sl_root))
                from sigmalang.core.primitives import Glyph, GlyphType
                from src.recycler.glyph_kv_cache import TokenGlyphMapper
                self._mapper = TokenGlyphMapper()
            except Exception:
                pass

    def token_to_primitive(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map token IDs → primitive IDs  [same shape as token_ids].
        Pure-Python loop over vocab; cached for inference (see forward()).
        """
        if self._mapper is not None:
            # Vectorised via mapper._map_one — still Python loop but fast enough
            flat = token_ids.reshape(-1).tolist()
            prims = [self._mapper._map_one(t).primitive_id for t in flat]
            return torch.tensor(prims, dtype=torch.long, device=token_ids.device).reshape(token_ids.shape)
        else:
            # Fallback: uniform bucketing
            return token_ids % N_PRIMITIVES

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long tensor of token IDs

        Returns:
            glyph_coords: [B, L, d_glyph] float tensor of glyph coordinates
        """
        primitive_ids = self.token_to_primitive(token_ids)   # [B, L]
        return self.embed(primitive_ids)                       # [B, L, d_glyph]


# ---------------------------------------------------------------------------
# SSM selective scan — pure PyTorch (CPU-compatible, CUDA-swappable)
# ---------------------------------------------------------------------------

def selective_scan_sequential(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-order-hold discretized selective SSM scan.

    Args:
        x:     [B, L, d_model]  — input (after conv)
        delta: [B, L, d_model]  — step size Δ (from glyph coords)
        A:     [d_model, d_state] — learned state matrix (negative)
        B:     [B, L, d_state]  — input projection (from glyph coords)
        C:     [B, L, d_state]  — output projection (from glyph coords)
        D:     [d_model]        — skip connection (learned)

    Returns:
        y: [B, L, d_model]

    Complexity: O(L × d_model × d_state) sequential.
    For production: replace with mamba_ssm.selective_scan_cuda (parallel,
    10-100× faster, same numerics).

    ZOH discretization:
        Ā_t  = exp(Δ_t ⊙ A)            [B, L, d_model, d_state]
        B̄_t  = Δ_t ⊙ B_t               [B, L, d_model, d_state]
        h_t  = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
        y_t  = C_t^T h_t  +  D ⊙ x_t
    """
    B_batch, L, d_model = x.shape
    d_state = A.shape[-1]

    # Discretize A: [B, L, d_model, d_state]
    deltaA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )

    # Discretize B×x: [B, L, d_model, d_state]
    # B̄_t * x_t = (Δ_t ⊙ B_t) ⊙ x_t — but B is d_state, x is d_model
    # We broadcast: Δ [B,L,d_m] × B [B,L,d_s] × x [B,L,d_m] unsqueeze to align
    deltaB_x = (
        delta.unsqueeze(-1)         # [B, L, d_model, 1]
        * B.unsqueeze(2)            # [B, L, 1, d_state]
        * x.unsqueeze(-1)           # [B, L, d_model, 1]
    )  # → [B, L, d_model, d_state]

    # Sequential scan over L
    h = torch.zeros(B_batch, d_model, d_state, device=x.device, dtype=x.dtype)
    ys: List[torch.Tensor] = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_x[:, t]          # [B, d_model, d_state]
        # y_t = Σ_s  C_t[s] * h[d, s]  for each d
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)   # [B, d_model]
        ys.append(y_t)

    y = torch.stack(ys, dim=1)    # [B, L, d_model]
    return y + x * D.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Core SSM Layer with glyph-driven selectivity
# ---------------------------------------------------------------------------

class GlyphSSMLayer(nn.Module):
    """
    Single selective SSM layer where B, C, Δ are functions of glyph coordinates.

    Standard Mamba: B_proj, C_proj, dt_proj take d_model inputs.
    Glyph-Mamba:   B_proj, C_proj, dt_proj take d_glyph inputs  (d_glyph << d_model).

    This means the selectivity parameters encode *semantic glyph neighbourhood*
    rather than *syntactic token context* — the SSM asks "which glyph subspace
    is most relevant?" at every step.

    The x (content) stream still operates in d_model space for expressiveness;
    only the *gating* parameters are glyph-derived.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = DEFAULT_D_STATE,
        d_glyph: int = DEFAULT_D_GLYPH,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_glyph = d_glyph
        dt_rank = dt_rank or max(1, d_glyph // 4)

        # Core SSM parameters (learned, A must stay negative → log parameterisation)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_model, -1))
        )  # [d_model, d_state]
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        # ── Glyph-driven selectivity ──────────────────────────────────────────
        # B, C: project from d_glyph (NOT d_model) → d_state
        self.B_proj  = nn.Linear(d_glyph, d_state,   bias=False)
        self.C_proj  = nn.Linear(d_glyph, d_state,   bias=False)
        # Δ (dt): project from d_glyph → dt_rank → d_model
        self.dt_proj_down = nn.Linear(d_glyph, dt_rank, bias=False)
        self.dt_proj_up   = nn.Linear(dt_rank, d_model, bias=True)
        # Init dt bias so Δ starts near a good operating range
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_proj_up.bias.data = torch.log(torch.expm1(dt_init))
        # ─────────────────────────────────────────────────────────────────────

        nn.init.normal_(self.B_proj.weight, std=0.02)
        nn.init.normal_(self.C_proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        glyph_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:            [B, L, d_model] — token content stream
            glyph_coords: [B, L, d_glyph] — glyph coordinate stream

        Returns:
            y: [B, L, d_model]
        """
        A = -torch.exp(self.A_log)   # [d_model, d_state] — always negative

        # Selectivity from glyph coordinates ← the key innovation
        B   = self.B_proj(glyph_coords)                          # [B, L, d_state]
        C   = self.C_proj(glyph_coords)                          # [B, L, d_state]
        dt  = F.softplus(
            self.dt_proj_up(self.dt_proj_down(glyph_coords))
        )                                                          # [B, L, d_model]

        return selective_scan_sequential(x, dt, A, B, C, self.D)


# ---------------------------------------------------------------------------
# Full Mamba-style Block
# ---------------------------------------------------------------------------

class GlyphMambaBlock(nn.Module):
    """
    Complete Mamba-style transformer block with glyph-driven selectivity.

    Structure:
        in_proj(x)                      → x_inner [B,L,d_model], z [B,L,d_model]
        conv1d(x_inner)                 → local context
        GlyphSSMLayer(x_conv, g)        → SSM output with glyph selectivity
        y * silu(z)                     → gated output
        out_proj(gated)                 → [B, L, d_model]
        residual + LayerNorm            → final output

    Can replace one attention head in any Transformer with:
        block = GlyphMambaBlock(d_model=model.d_model)
        # swap: model.layers[k].attention → block
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = DEFAULT_D_STATE,
        d_glyph: int = DEFAULT_D_GLYPH,
        d_conv: int = DEFAULT_D_CONV,
        expand: int = 2,
        use_sigmalang: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # Expand projection (content + gating branch)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Local context conv (causal, grouped depthwise)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # Glyph coordinate embedding
        self.glyph_embed = GlyphCoordinateEmbedding(
            d_glyph=d_glyph, use_sigmalang=use_sigmalang
        )
        # Project glyph coords to match d_inner for SSM
        self.glyph_proj = nn.Linear(d_glyph, d_glyph, bias=False)

        # Core SSM (glyph-driven, operating at d_inner)
        self.ssm = GlyphSSMLayer(
            d_model=d_inner,
            d_state=d_state,
            d_glyph=d_glyph,
        )

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._d_conv = d_conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, d_model] — input from previous layer
            token_ids:     [B, L] long — original token IDs for glyph mapping.
                           If None, a dummy primitive-0 coordinate is used
                           (useful for transfer from attention-only models).

        Returns:
            output: [B, L, d_model] with residual added
        """
        residual = hidden_states
        B, L, _ = hidden_states.shape

        # LayerNorm pre-activation
        h = self.norm(hidden_states)

        # Expand + split into content and gate
        xz = self.in_proj(h)                           # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)                    # each [B, L, d_inner]

        # Causal conv for local context (strip extra padding)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                        # [B, L, d_inner]

        # Glyph coordinates for selectivity
        if token_ids is not None:
            glyph_coords = self.glyph_proj(
                self.glyph_embed(token_ids)
            )                                           # [B, L, d_glyph]
        else:
            # Fallback: uniform glyph (no selectivity signal)
            dummy_ids = torch.zeros(B, L, dtype=torch.long, device=hidden_states.device)
            glyph_coords = self.glyph_proj(
                self.glyph_embed(dummy_ids)
            )

        # Glyph-driven SSM scan
        y = self.ssm(x_conv, glyph_coords)             # [B, L, d_inner]

        # SiLU gate
        y = y * F.silu(z)                              # [B, L, d_inner]

        # Project back
        out = self.out_proj(y)                         # [B, L, d_model]

        return out + residual


# ---------------------------------------------------------------------------
# End-to-end model (benchmark target)
# ---------------------------------------------------------------------------

class GlyphMambaModel(nn.Module):
    """
    Minimal Glyph-Mamba model for benchmarking and integration testing.

    Architecture: token_embed → N × GlyphMambaBlock → LayerNorm → lm_head
    Not trained — use for latency / memory / expressiveness benchmarks.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = DEFAULT_D_STATE,
        d_glyph: int = DEFAULT_D_GLYPH,
        d_conv: int = DEFAULT_D_CONV,
        expand: int = 2,
        use_sigmalang: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GlyphMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_glyph=d_glyph,
                d_conv=d_conv,
                expand=expand,
                use_sigmalang=use_sigmalang,
            )
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (standard)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            token_ids:     [B, L] long
            return_hidden: if True, return last hidden state instead of logits

        Returns:
            logits: [B, L, vocab_size]   or
            hidden: [B, L, d_model]      (if return_hidden=True)
        """
        h = self.token_embed(token_ids)           # [B, L, d_model]
        for block in self.blocks:
            h = block(h, token_ids=token_ids)
        h = self.norm_f(h)
        if return_hidden:
            return h
        return self.lm_head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def glyph_selectivity_ratio(self) -> float:
        """
        Ratio of glyph-selectivity params to total params.

        Higher = more of the model's selectivity is glyph-driven.
        A ratio of 1.0 would mean the entire selective mechanism is glyph-native.
        """
        total = self.count_parameters()
        glyph_params = sum(
            p.numel()
            for block in self.blocks
            for name, p in block.named_parameters()
            if any(k in name for k in ("B_proj", "C_proj", "dt_proj", "glyph"))
        )
        return round(glyph_params / total, 4) if total else 0.0
