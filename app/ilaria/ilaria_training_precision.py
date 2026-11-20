"""Resolve training precision independently from inference and feature extraction."""

import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

PRECISIONS = ("BF16", "FP16", "FP32")


def config_precision(config):
    get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    value = get("precision")
    if value is None:
        return "FP16" if get("fp16_run", False) else "FP32"
    if value not in PRECISIONS:
        raise ValueError("Choose BF16, FP16 or FP32.")
    return value


def describe_cuda(cuda, selection=None):
    count = cuda.device_count()
    if not count:
        return []
    indices = list(range(count))
    if selection:
        ids = str(selection).split("-")
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        mapping = [item.strip() for item in visibility.split(",")] if visibility else []
        if mapping and all(i in mapping for i in ids):
            indices = [mapping.index(i) for i in ids]
        elif all(i.isdecimal() and int(i) < count for i in ids):
            indices = [int(i) for i in ids]
        else:
            raise ValueError("Select valid CUDA GPU IDs for training.")
    devices = []
    for index in indices:
        major, _ = cuda.get_device_capability(index)
        native = False
        if major >= 8:
            with cuda.device(index):
                try:
                    native = cuda.is_bf16_supported(including_emulation=False)
                except TypeError:
                    native = cuda.is_bf16_supported()
        devices.append((cuda.get_device_name(index), bool(native)))
    return devices


def available_precisions(devices):
    if not devices:
        return ["FP32"]
    return list(PRECISIONS if all(native for _, native in devices) else ("FP16", "FP32"))


def automatic_precision(devices, notebook=False):
    if devices and all(native and re.search(r"\bRTX\s+[345]\d{3}\b", name, re.I) for name, native in devices):
        return "BF16"
    if notebook and devices and all(re.search(r"\bT4\b", name) for name, _ in devices):
        return "FP16"
    return "FP32"


def validate_device_precision(precision, devices):
    if precision not in available_precisions(devices):
        raise ValueError(f"{precision} is unavailable for this GPU selection. BF16 requires native support on every selected GPU; FP16 requires CUDA.")


def saved_precision(root, model):
    if not model or Path(model).name != model or any(char in model for char in ("/", "\\", ":")) or model in (".", ".."):
        return None
    path = Path(root) / "logs" / model / "config.json"
    if path.is_file():
        return config_precision(json.loads(path.read_text(encoding="utf-8"))["train"])
    return None


def precision_preference(path):
    try:
        choice = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return choice if choice in PRECISIONS else None


def save_precision_preference(path, choice):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(choice or "Auto")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
