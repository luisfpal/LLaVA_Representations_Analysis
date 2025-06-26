from typing import List, Dict, Optional, Union
import torch
from utils import ModelType, resolve_layer_indices


class HeadProjectionTracer:
    def __init__(
        self,
        model: ModelType,
        target_layers: Union[int, List[int], str],
        target_heads: Optional[Dict] = None,
        tokens_mode: str = "last",
    ):
        """
        Extracts per-head representations from the residual stream projections
        after the attention mechanism, before the MLP normalization.

        Parameters:
        - model: HuggingFace transformer or Multimodal model (e.g., LLaVA)
        - target_layers: list of integers (layer indices to trace) or string (e.g., "all")
        - target_heads: dict[layer_idx] = list of head indices to trace, or None for all
        - tokens_mode: 'last' or 'mean' -- how to pool across time (tokens)
        """
        if hasattr(model, "language_model"):
            self.model = model.language_model
        else:
            self.model = model

        self.target_layers = resolve_layer_indices(self.model, target_layers)
        self.target_layers = [layer - 1 for layer in self.target_layers]
        self.target_heads = target_heads or {}
        self.tokens_mode = tokens_mode  # 'last' or 'mean'
        self.head_dim = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.residual_stream_projections = {}
        self.handles = []

        # Pre-compute weight slices and head indices for each layer
        self._precomputed_weights = {}
        self._layer_head_indices = {}

    def _precompute_layer_info(self, layer_idx: int, module: torch.nn.Module):
        """Pre-compute weight slices and head indices for efficient processing."""
        # Determine head indices for this layer
        head_indices = self.target_heads.get(layer_idx, None)
        if head_indices is None:
            local_head_indices = list(range(self.num_heads))
        else:
            local_head_indices = head_indices

        self._layer_head_indices[layer_idx] = local_head_indices

        # Pre-slice weight matrix for all heads
        W = module.weight.data  # (hidden_size, hidden_size)
        weight_slices = []

        for h in local_head_indices:
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            W_head = W[start:end, :]  # (head_dim, hidden_size)
            weight_slices.append(W_head)

        # Stack weight slices for vectorized operations
        # Shape: (num_active_heads, head_dim, hidden_size)
        if weight_slices:
            self._precomputed_weights[layer_idx] = torch.stack(weight_slices, dim=0)
        else:
            self._precomputed_weights[layer_idx] = torch.empty(
                0, self.head_dim, self.hidden_size
            )

    def _make_hook(self, layer_idx):
        """
        Returns an optimized hook that computes efficient head projections using vectorized operations.
        """
        local_head_indices = self._layer_head_indices[layer_idx]
        W_heads = self._precomputed_weights[
            layer_idx
        ]  # (num_heads, head_dim, hidden_size)

        def hook(module, input, output):
            if len(local_head_indices) == 0:
                return

            x = input[0]  # (batch, seq_len, hidden_size)

            # Efficient pooling
            if self.tokens_mode == "mean":
                x_pooled = x.mean(dim=1)  # (batch, hidden_size)
            elif self.tokens_mode == "last":
                x_pooled = x[:, -1, :]  # (batch, hidden_size)
            else:
                raise ValueError("mode must be 'last' or 'mean'")

            # Vectorized head processing
            # Extract all head features at once
            head_features = []
            for h in local_head_indices:
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                head_features.append(x_pooled[:, start:end])

            # Stack head features: (num_heads, batch, head_dim)
            x_heads = torch.stack(head_features, dim=0)

            # Batched matrix multiplication
            # x_heads: (num_heads, batch, head_dim)
            # W_heads: (num_heads, head_dim, hidden_size)
            # Result: (num_heads, batch, hidden_size)
            projections = torch.bmm(x_heads, W_heads)

            # Efficient result storage
            # Store results with minimal dictionary operations
            with torch.no_grad():
                for i, h in enumerate(local_head_indices):
                    key = f"layer_{layer_idx}/head_{h}"
                    # Extract the projection for this head: (batch, hidden_size)
                    self.residual_stream_projections[key] = projections[i].detach()

        return hook

    def trace(self):
        """Set up hooks for all target layers with pre-computation."""
        print(f"Setting up optimized hooks for {len(self.target_layers)} layers...")

        for layer_idx in self.target_layers:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj

            # Pre-compute layer-specific information
            self._precompute_layer_info(layer_idx, o_proj)

            # Register optimized hook
            hook = o_proj.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(hook)

        total_heads = sum(len(indices) for indices in self._layer_head_indices.values())
        print(
            f"Hooks registered for {total_heads} heads across {len(self.target_layers)} layers"
        )

    def get_residual_stream_projections(self) -> Dict[str, torch.Tensor]:
        return self.residual_stream_projections
        # residual_stream_projections is a dictionary of shape:
        # {
        #     "layer_idx/head_idx": (batch_size, hidden_size)
        # }

    def clear(self):
        """Clean up hooks and cached data."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.residual_stream_projections.clear()

        # Clean up pre-computed data
        self._precomputed_weights.clear()
        self._layer_head_indices.clear()
