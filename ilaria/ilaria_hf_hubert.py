"""Load Hugging Face HuBERT Base embedders using the bundled TorchAudio."""

import json
from pathlib import Path
from pickle import UnpicklingError
import zipfile

import torch
from torch import nn


_BASE_CONFIG = {
    "model_type": "hubert",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "conv_dim": [512] * 7,
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "layer_norm_eps": 1e-5,
    "feat_extract_activation": "gelu",
    "hidden_act": "gelu",
    "feat_proj_layer_norm": True,
    "conv_pos_batch_norm": False,
}

_POSITIONAL_WEIGHT_NAMES = {
    "parametrizations.weight.original0": "weight_g",
    "parametrizations.weight.original1": "weight_v",
}


class _HubertEmbedder(nn.Module):
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    @property
    def has_final_proj(self):
        return self.projection is not None

    def final_proj(self, features):
        if self.projection is None:
            raise ValueError("This embedder has no 256-feature projection for RVC v1. Select an RVC v2 model or an embedder that supports v1.")
        return self.projection(features)

    def extract_features(self, source, padding_mask=None, output_layer=None):
        # These HuBERT embedders use the final hidden state for both RVC versions.
        lengths = None if padding_mask is None else (~padding_mask).sum(dim=-1)
        features, feature_lengths = self.encoder(source, lengths)
        feature_mask = None
        if feature_lengths is not None:
            positions = torch.arange(features.size(1), device=features.device)
            feature_mask = positions.unsqueeze(0) >= feature_lengths.unsqueeze(1)
        return features, feature_mask


def _build_encoder(config_path):
    from torchaudio.models import wav2vec2_model

    with config_path.open("rb") as config_file:
        raw_config = config_file.read(256 * 1024 + 1)
    if len(raw_config) > 256 * 1024:
        raise ValueError("The HuBERT config.json is too large.")
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("The HuBERT config.json must contain a JSON object.")
    for setting, supported_value in _BASE_CONFIG.items():
        if config.get(setting, supported_value) != supported_value:
            raise ValueError(f"The HuBERT Base setting {setting} must be {supported_value!r}.")
    norm = config.get("feat_extract_norm", "group")
    if norm not in ("group", "layer"):
        raise ValueError("The HuBERT feature extractor must use group or layer normalization.")
    conv_bias = config.get("conv_bias", False)
    stable_layer_norm = config.get("do_stable_layer_norm", False)
    if not isinstance(conv_bias, bool) or not isinstance(stable_layer_norm, bool):
        raise ValueError("HuBERT convolution bias and stable layer normalization must be boolean settings.")
    return wav2vec2_model(
        extractor_mode=f"{norm}_norm",
        extractor_conv_layer_config=list(zip(_BASE_CONFIG["conv_dim"], _BASE_CONFIG["conv_kernel"], _BASE_CONFIG["conv_stride"])),
        extractor_conv_bias=conv_bias,
        encoder_embed_dim=_BASE_CONFIG["hidden_size"],
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=_BASE_CONFIG["num_conv_pos_embeddings"],
        encoder_pos_conv_groups=_BASE_CONFIG["num_conv_pos_embedding_groups"],
        encoder_num_layers=_BASE_CONFIG["num_hidden_layers"],
        encoder_num_heads=_BASE_CONFIG["num_attention_heads"],
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=_BASE_CONFIG["intermediate_size"],
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.0,
        encoder_layer_norm_first=stable_layer_norm,
        encoder_layer_drop=0.0,
        aux_num_out=None,
    )


def _load_weights(weight_path):
    size_limit = 1024**3
    with weight_path.open("rb") as file:
        if file.seek(0, 2) > size_limit:
            raise ValueError("The HuBERT pytorch_model.bin is larger than 1 GiB.")
        file.seek(0)
        # Torch 2.0 also accepts legacy pickle formats and TorchScript via torch.load.
        # Require a ZIP state dictionary and pass the same checked file to the restricted loader.
        if file.read(4) != b"PK\x03\x04" or not zipfile.is_zipfile(file):
            raise ValueError("Upload pytorch_model.bin saved in the PyTorch ZIP state dictionary format.")
        with zipfile.ZipFile(file) as archive:
            entries = archive.infolist()
            paths = [entry.filename.split("/") for entry in entries]
            if (
                not paths
                or sum(entry.file_size for entry in entries) > size_limit
                or any(len(path) < 2 or path[0] != paths[0][0] for path in paths)
                or any("constants.pkl" in path or "code" in path for path in paths)
                or [paths[0][0], "data.pkl"] not in paths
                or [paths[0][0], "version"] not in paths
            ):
                raise ValueError("Upload a PyTorch state dictionary ZIP containing tensor weights.")
        file.seek(0)
        try:
            state = torch.load(file, map_location="cpu", weights_only=True)
        except (RuntimeError, TypeError, ValueError, OSError, EOFError, UnpicklingError) as exc:
            raise ValueError("The HuBERT weights could not be read as a tensor-only state dictionary.") from exc
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("The HuBERT weights must be a Hugging Face tensor state dictionary.")
    return state


def _load_hubert_state(encoder, state):
    expected_weights = encoder.state_dict()
    encoder_weights = {}
    projection_weights = {}
    positional_prefix = "encoder.transformer.pos_conv_embed.conv."
    for name, value in state.items():
        key = name.removeprefix("module.").removeprefix("hubert.")
        if key.startswith("final_proj."):
            projection_key = key.removeprefix("final_proj.")
            if projection_key in projection_weights:
                raise ValueError(f"The HuBERT checkpoint contains duplicate weights for {key}.")
            projection_weights[projection_key] = value
            continue
        if key.startswith("feature_projection."):
            key = f"encoder.{key}"
        elif key.startswith("encoder."):
            key = f"encoder.transformer.{key.removeprefix('encoder.')}"
        elif not key.startswith("feature_extractor."):
            continue
        if key.startswith(positional_prefix) and key not in expected_weights:
            positional_name = key.removeprefix(positional_prefix)
            for modern_name, legacy_name in _POSITIONAL_WEIGHT_NAMES.items():
                if positional_name in (modern_name, legacy_name):
                    replacement = legacy_name if positional_name == modern_name else modern_name
                    key = f"{positional_prefix}{replacement}"
                    break
        if key in encoder_weights:
            raise ValueError(f"The HuBERT checkpoint contains duplicate weights for {key}.")
        encoder_weights[key] = value
    try:
        encoder.load_state_dict(encoder_weights, strict=True)
        projection = None
        if projection_weights:
            projection = nn.Linear(_BASE_CONFIG["hidden_size"], 256)
            projection.load_state_dict(projection_weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("The weights do not match the supported HuBERT Base encoder or its 256-feature projection.") from exc
    return _HubertEmbedder(encoder, projection)


def load_hf_hubert(folder, device="cpu", half=False):
    """Load a local Hugging Face HuBERT Base pair for RVC feature extraction."""
    folder = Path(folder)
    state = _load_weights(folder / "pytorch_model.bin")
    encoder = _build_encoder(folder / "config.json")
    embedder = _load_hubert_state(encoder, state)
    precision = torch.float16 if half and torch.device(device).type == "cuda" else torch.float32
    return embedder.to(device=device, dtype=precision).eval()
