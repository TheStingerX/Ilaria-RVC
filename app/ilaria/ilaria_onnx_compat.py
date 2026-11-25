"""Tracing adapters for the legacy RVC ONNX network."""

import math
from types import MethodType
from typing import Optional

import torch
from torch.nn import functional as F

from infer.lib.infer_pack.attentions import MultiHeadAttention
from infer.lib.infer_pack.models_onnx import ResidualCouplingBlock


def prepare_onnx_trace(model):
    for layer in model.modules():
        if isinstance(layer, MultiHeadAttention):
            layer.attention = MethodType(_export_attention, layer)
        elif isinstance(layer, ResidualCouplingBlock):
            layer.forward = MethodType(_export_flow, layer)


def _export_flow(self, x, x_mask, g=None, reverse=False):
    flows = reversed(self.flows) if reverse else self.flows
    for flow in flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    return x


def _export_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s = key.size()
    t_t = query.size(2)
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    relative_index: Optional[torch.Tensor] = None
    relative_mask: Optional[torch.Tensor] = None
    if self.window_size is not None:
        positions = torch.arange(t_s, device=query.device)
        offsets = positions.unsqueeze(0) - positions.unsqueeze(1) + self.window_size
        relative_mask = (offsets >= 0) & (offsets <= 2 * self.window_size)
        relative_index = offsets.clamp(0, 2 * self.window_size).expand_as(scores)
        relative_logits = self._matmul_with_relative_keys(
            query / math.sqrt(self.k_channels), self.emb_rel_k
        )
        scores = scores + relative_logits.gather(3, relative_index) * relative_mask
    if self.proximal_bias:
        assert t_s == t_t, "Proximal bias is only available for self-attention."
        scores = scores + self._attention_bias_proximal(t_s).to(
            device=scores.device, dtype=scores.dtype
        )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
        if self.block_length is not None:
            assert (
                t_s == t_t
            ), "Local attention is only available for self-attention."
            block_mask = (
                torch.ones_like(scores)
                .triu(-self.block_length)
                .tril(self.block_length)
            )
            scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if relative_index is not None and relative_mask is not None and self.window_size is not None:
        relative_weights = p_attn.new_zeros((b, self.n_heads, t_t, 2 * self.window_size + 1))
        relative_weights = relative_weights.scatter_add(3, relative_index, p_attn * relative_mask)
        output = output + self._matmul_with_relative_values(relative_weights, self.emb_rel_v)
    output = (
        output.transpose(2, 3).contiguous().view(b, d, t_t)
    )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

