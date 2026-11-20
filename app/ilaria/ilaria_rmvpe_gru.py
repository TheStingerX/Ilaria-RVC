"""RMVPE GRU recovery shared by new and existing installations."""

from copy import deepcopy
from types import MethodType

import torch


def run_gru(gru, x):
    x = x.contiguous()
    try:
        return gru(x)[0]
    except RuntimeError as error:
        if not x.is_cuda or "CUDNN_STATUS_NOT_SUPPORTED" not in str(error):
            raise
    # Leave the live CUDA model and process-wide cuDNN settings intact.
    cpu_gru = deepcopy(gru).to(device="cpu", dtype=torch.float32)
    output = cpu_gru(x.to(device="cpu", dtype=torch.float32))[0]
    return output.to(device=x.device, dtype=x.dtype)


def _legacy_forward(layer, x):
    return run_gru(layer.gru, x)


def configure_rmvpe(model, gru_type):
    """Apply the fix to RMVPE loaded from a release with the older source file."""
    for layer in model.modules():
        if isinstance(layer, gru_type) and not hasattr(layer, "_forward_with_fallback"):
            layer.forward = MethodType(_legacy_forward, layer)
