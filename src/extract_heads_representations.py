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
        Efficiently extract per-head projections from o_proj using slicing.

        Parameters:
        - model: HuggingFace transformer or Multimodal model (e.g., LLaVA)
        - target_layers: list of integers (layer indices to trace)
        - target_heads: dict[layer_idx] = list of head indices to trace, or None for all
        - tokens_mode: 'last' or 'mean' -- how to pool across time (tokens)
        """
        if hasattr(model, "language_model"):
            self.model = model.language_model
        else:
            self.model = model

        self.target_layers = resolve_layer_indices(self.model, target_layers)
        self.target_heads = target_heads or {}
        self.tokens_mode = tokens_mode  # 'last' or 'mean'
        self.head_dim = (
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        self.hidden_size = self.model.config.hidden_size
        self.residual_stream_projections = {}
        self.handles = []

    def _make_hook(self, layer_idx, head_indices):
        """
        Returns a hook that computes efficient head projections.
        """

        def hook(module, input, output):
            x = input[0]  # (batch, seq_len, hidden_size)
            W = module.weight.data  # (hidden_size, hidden_size)
            # Transpose W because we want to make a projection with x @ W
            # W comes from o_proj which is applied like this:
            # self.o_proj(x) = x @ W.T
            # since I am using slices of W multiple times I just transpose it once here

            # Choose pooling strategy
            if self.tokens_mode == "mean":
                x_pooled = x.mean(dim=1)  # (batch, hidden_size)
            elif self.tokens_mode == "last":
                x_pooled = x[:, -1, :]  # (batch, hidden_size)
            else:
                raise ValueError("mode must be 'last' or 'mean'")

            # Determine head indices
            if head_indices is None:
                local_head_indices = list(range(self.hidden_size // self.head_dim))
            else:
                local_head_indices = head_indices

            for h in local_head_indices:
                start = h * self.head_dim
                end = (h + 1) * self.head_dim

                # Get submatrices
                x_head = x_pooled[:, start:end]  # (batch, head_dim)
                W_head = W[start:end, :]  # (head_dim, hidden_size)

                # Efficient linear projection
                proj = x_head @ W_head  # (batch, hidden_size)
                # proj = torch.matmul(x_head.view(-1, self.head_dim), W_head.view(self.head_dim, -1))  # (batch, hidden_size)
                # torch.matmul can broadcast unexpectedly; shape-sensitive

                key = f"layer_{layer_idx}/head_{h}"
                self.residual_stream_projections[key] = proj.detach()

        return hook

    def trace(self):
        for layer_idx in self.target_layers:
            head_indices = self.target_heads.get(layer_idx, None)
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_hook(
                self._make_hook(layer_idx, head_indices)
            )
            self.handles.append(hook)

    def get_residual_stream_projections(self) -> Dict[str, torch.Tensor]:
        return self.residual_stream_projections

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.residual_stream_projections.clear()
