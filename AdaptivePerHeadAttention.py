from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class AdaptivePerHeadAttention(torch.nn.Module):
    """
    Per-head adaptive attention with learnable sink budget and temperature.

    The module does not implement projection layers. Instead, it takes pre-computed
    attention scores and returns a renormalised probability tensor that can be fed
    into the value aggregation step inside the model.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_idx: int = 0,
        num_layers: int = 24,
        use_adaptive: bool = True,
        enable_two_pool: bool = True,
    ) -> None:
        super().__init__()
        if dim % max(num_heads, 1) != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.num_layers = max(num_layers, 1)
        self.use_adaptive = use_adaptive
        self.enable_two_pool = enable_two_pool
        self.head_dim = dim // max(num_heads, 1)

        # Normalised depth in the stack so we can scale epsilon/temperature.
        depth = layer_idx / max(self.num_layers - 1, 1)
        self.register_buffer("layer_depth", torch.tensor(depth, dtype=torch.float32))

    def get_adaptive_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-head epsilon and temperature vectors."""
        if not self.use_adaptive:
            epsilon = torch.full((self.num_heads,), 0.02)
            temperature = torch.ones(self.num_heads)
            return epsilon, temperature

        layer_depth = float(self.layer_depth.item())

        # Epsilon increases with depth: 0.01 → 0.05.
        epsilon = torch.full((self.num_heads,), 0.01 + 0.04 * layer_depth)

        # Temperature increases with depth: 1.0 → 1.8.
        temperature = torch.full((self.num_heads,), 1.0 + 0.8 * layer_depth)

        return epsilon, temperature

    @staticmethod
    def _prepare_epsilon(epsilon: torch.Tensor, num_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        epsilon = epsilon.to(device=device, dtype=dtype)
        epsilon = epsilon.view(1, num_heads, 1, 1)
        epsilon = epsilon.clamp_min(1e-6).clamp_max(1.0 - 1e-6)
        return epsilon

    def two_pool_renormalization(
        self,
        weights: torch.Tensor,
        epsilon: torch.Tensor,
        sink_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Apply two-pool renormalisation to attention weights.

        Args:
            weights: (batch, num_heads, tgt_len, src_len)
            epsilon: (num_heads,)
            sink_indices: locations treated as sinks (e.g. BOS)
        """
        if not sink_indices:
            return weights

        sink_indices = list(sink_indices)
        src_len = weights.size(-1)
        sink_mask = torch.zeros(src_len, device=weights.device, dtype=torch.bool)
        sink_mask[sink_indices] = True
        content_mask = ~sink_mask

        # Sum in fp32 for stability, then convert back.
        sink_mass = weights[..., sink_mask].sum(dim=-1, keepdim=True, dtype=torch.float32)
        content_mass = weights[..., content_mask].sum(dim=-1, keepdim=True, dtype=torch.float32)

        epsilon_expanded = self._prepare_epsilon(epsilon, self.num_heads, weights.device, weights.dtype)

        output = torch.zeros_like(weights)
        output[..., content_mask] = (
            weights[..., content_mask]
            * (1.0 - epsilon_expanded)
            / (content_mass + 1e-9)
        )
        output[..., sink_mask] = (
            weights[..., sink_mask]
            * epsilon_expanded
            / (sink_mass + 1e-9)
        )
        return output

    def forward(
        self,
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sink_indices: Optional[Iterable[int]] = (0,),
    ) -> torch.Tensor:
        """
        Convert raw attention scores into probability distribution.

        Args:
            scores: (batch, num_heads, tgt_len, src_len) raw dot-product scores.
            attention_mask: broadcast-compatible additive mask.
            sink_indices: indices that should receive the sink mass budget.
        """
        if attention_mask is not None:
            scores = scores + attention_mask

        epsilon, temperature = self.get_adaptive_params()
        temperature = temperature.to(device=scores.device, dtype=scores.dtype)
        temperature = temperature.view(1, self.num_heads, 1, 1).clamp_min(1e-4)

        scores = scores / temperature
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=scores.dtype)

        if self.enable_two_pool:
            probs = self.two_pool_renormalization(probs, epsilon, sink_indices=sink_indices)

        return probs


# Usage example:
# attn = AdaptivePerHeadAttention(dim=768, num_heads=12, layer_idx=10, num_layers=24)
# output = attn(query, key, value)
