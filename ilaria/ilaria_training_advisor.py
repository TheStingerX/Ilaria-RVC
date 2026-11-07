"""Conservative, transparent training suggestions; never edits a training config."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import html
import math
import re

AUTO_GPU = "Auto (largest compatible GPU)"
SOURCES = (
    ("RVC precision policy", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/config.py"),
    ("Applio training guide", "https://docs.applio.org/getting-started/training/"),
    ("RVC out-of-memory FAQ", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-%28Frequently-Asked-Questions%29"),
    ("PyTorch mixed precision", "https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html"),
)


def _cell(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ").replace("`", "&#96;")


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _gib(value):
    number = _number(value)
    return "Unknown" if number is None else "%.1f GiB" % number


def gpu_choices(hardware):
    return [AUTO_GPU] + ["GPU %s: %s" % (gpu["index"], gpu["name"])
                         for gpu in hardware.get("devices", [])]


def _select_gpu(hardware, selection):
    devices = hardware.get("devices", [])
    if selection and selection != AUTO_GPU:
        match = re.match(r"GPU (\d+):", str(selection))
        if not match:
            raise ValueError("Scan the PC and select one of the detected GPUs.")
        gpu = next((item for item in devices if item["index"] == int(match.group(1))), None)
        if gpu is None:
            raise ValueError("That GPU is no longer in this scan. Scan the PC again.")
        return gpu
    eligible = [gpu for gpu in devices if gpu.get("compatible") is not False]
    return max(eligible, key=lambda gpu: _number(gpu.get("total_gib")) or 0) if eligible else None


def hardware_report(hardware):
    cpu, ram, disk = (hardware.get(key, {}) for key in ("cpu", "ram", "disk"))
    lines = ["### Detected hardware", "",
             "Checked at **%s** (PC local time). Free memory is a snapshot; scan again after closing other GPU applications." % _cell(hardware.get("checked_at", "Unknown")), "",
             "| Component | Detected value |", "| --- | --- |",
             "| CPU | %s |" % _cell(cpu.get("name") or "Unknown"),
             "| CPU cores / logical threads | %s / %s |" % (cpu.get("physical") or "Unknown", cpu.get("logical") or "Unknown"),
             "| RAM total / available | %s / %s |" % (_gib(ram.get("total_gib")), _gib(ram.get("available_gib"))),
             "| Free disk space at the installation | %s |" % _gib(disk.get("free_gib")),
             "| System / Python | %s / %s |" % (_cell(hardware.get("os", "Unknown")), _cell(hardware.get("python", "Unknown"))),
             "| PyTorch / bundled CUDA | %s / %s |" % (_cell(hardware.get("torch_version") or "Unavailable"), _cell(hardware.get("cuda_version") or "Not a CUDA build")),
             "| CUDA available to this installation | %s |" % ("Yes" if hardware.get("cuda_available") else "No"), ""]
    if hardware.get("devices"):
        lines.extend(["| GPU ID | Graphics card | Total VRAM | Free VRAM now | Runtime check |", "| --- | --- | --- | --- | --- |"])
        for gpu in hardware["devices"]:
            lines.append("| %s | %s | %s | %s | %s |" % (gpu["index"], _cell(gpu["name"]),
                _gib(gpu.get("total_gib")), _gib(gpu.get("free_gib")), _cell(gpu.get("reason") or "Detected; training has not been tested")))
    else:
        lines.append("No CUDA GPU could be inspected by this PyTorch installation. This can mean a missing/incompatible driver or runtime, not necessarily a missing graphics card.")
    for warning in hardware.get("warnings", []):
        lines.append("\n- " + _cell(warning))
    return "\n".join(lines)


def suggest_settings(hardware, sample_rate="32k", precision="FP32", selection=AUTO_GPU):
    if sample_rate not in ("32k", "40k", "48k") or precision not in ("FP32", "FP16"):
        raise ValueError("Choose 32k, 40k or 48k and FP32 or FP16.")
    gpu = _select_gpu(hardware, selection)
    logical = int(_number(hardware.get("cpu", {}).get("logical")) or 1)
    threads = min(8, max(1, logical - 2))
    ram = hardware.get("ram", {})
    available_ram, total_ram = _number(ram.get("available_gib")), _number(ram.get("total_gib"))
    notes = []
    if (total_ram is not None and total_ram < 8) or (available_ram is not None and available_ram < 3):
        threads = min(threads, 2)
        notes.append("RAM is tight: use fewer preparation CPU threads and close other applications. The trainer's DataLoader workers are separate from this CPU Threads control.")
    elif available_ram is not None and available_ram < 6:
        threads = min(threads, 4)
    disk_free = _number(hardware.get("disk", {}).get("free_gib"))
    if disk_free is not None and disk_free < 10:
        notes.append("Less than 10 GiB is free on the installation drive. Check space for prepared audio, features and resumable G/D checkpoints before training; the required amount depends on your dataset.")
    result = {"gpu": gpu, "precision": precision, "sample_rate": sample_rate, "batch_size": None,
              "cpu_threads": threads, "pitch_method": "rmvpe", "notes": notes, "ready": False}
    if not hardware.get("cuda_available") or gpu is None or gpu.get("compatible") is False:
        result["precision"] = "FP32"
        notes.append("No compatible CUDA GPU was confirmed. A GPU batch-size recommendation is unavailable. Check this installation's driver/PyTorch compatibility before GPU training; CPU-only training can be very slow.")
        return result
    cc = gpu.get("compute_capability")
    cc_text = ".".join(map(str, cc)) if isinstance(cc, (list, tuple)) else str(cc or "")
    legacy = bool(re.search(r"\b(?:GTX\s*)?16\d{2}\b", gpu.get("name", ""), re.I)) or cc_text == "6.1"
    if legacy and precision == "FP16":
        result["precision"] = "FP32"
        notes.append("FP16 was requested, but FP32 is recommended for this GTX 16-series/Pascal SM 6.1 GPU, following mainline RVC's precision policy. Mixed precision can be unstable here.")
    elif legacy:
        notes.append("Keep FP32 on this GTX 16-series/Pascal SM 6.1 GPU. If you run out of VRAM, reduce Batch Size rather than assuming FP16 will be reliable.")
    total, free = _number(gpu.get("total_gib")), _number(gpu.get("free_gib"))
    if not total or total < 1:
        notes.append("GPU VRAM size could not be read. No batch size is guessed; scan again after fixing the runtime or driver.")
        return result
    if free is None:
        budget = total * 0.75
        notes.append("Free VRAM could not be read. The suggestion assumes only 75% of total VRAM is available; this is a conservative assumption, not a measurement.")
    else:
        budget = min(total, free)
        if budget < total * 0.75:
            notes.append("A substantial part of VRAM is already occupied. The suggested batch is reduced using currently free memory. Unload the Inference voice and close other GPU programs, then scan again.")
    # Effective budget tiers reserve space for the desktop/runtime implicitly.
    # These are local starting heuristics, NOT official limits or a benchmark.
    tiers = [(21.0, 16, 16), (14.0, 12, 16), (9.5, 8, 12),
             (7.0, 4, 6), (4.5, 2, 4), (2.0, 1, 1)]
    base = next((fp16 if result["precision"] == "FP16" else fp32
                 for minimum, fp32, fp16 in tiers if budget >= minimum), None)
    if base is None:
        notes.append("Less than 2 GiB of VRAM is available. Free GPU memory and scan again; a workable batch cannot be recommended from this snapshot.")
        return result
    result["batch_size"] = max(1, min(16, int(base * {"32k": 1.0, "40k": 0.75, "48k": 0.5}[sample_rate])))
    result["ready"] = True
    if gpu.get("training_index", gpu["index"]) is None:
        notes.append("Custom CUDA visibility uses UUIDs or another nonnumeric mapping. This trainer expects numeric GPU IDs; confirm the correct training ID before launching. The visible scan ID alone is not a reliable replacement.")
    if sample_rate != "32k":
        notes.append("%s needs a smaller initial batch in this advisor. Match both pretrained G/D files to this sample rate and use a new experiment if prepared data was created at another rate." % sample_rate)
    if budget >= 4.5:
        result["pitch_method"] = "rmvpe_gpu"
    notes.extend([
        "Batch Size is a starting estimate, not a tested maximum or a speed guarantee. If CUDA reports out of memory, halve it (minimum 1), keep GPU caching off and retry. A larger batch is not always faster or better.",
        "FP32 remains ILARIA's default. On a compatible modern CUDA GPU, FP16 is an optional memory/speed comparison; it is not guaranteed to improve training speed or quality.",
        "CPU Threads applies to data preparation, not the training DataLoader. Increasing it beyond available CPU/RAM resources may slow preparation down.",
        "Use one selected GPU and one GPU pitch-extraction process initially. VRAM from different cards is not simply added together; GPUs should be evaluated individually.",
    ])
    return result


def recommendation_report(hardware, sample_rate="32k", precision="FP32", selection=AUTO_GPU):
    result = suggest_settings(hardware, sample_rate, precision, selection)
    gpu = result["gpu"]
    rows = [
        ("Sample Rate", result["sample_rate"], "32k is the initial baseline; higher rates require matching pretrains and more resources."),
        ("Training Precision", result["precision"], "FP32 default; FP16 is optional only where appropriate."),
        ("Batch Size", result["batch_size"] if result["batch_size"] is not None else "Not recommended yet", "Conservative estimate for this scan, rate and precision; not a benchmark."),
        ("CPU Threads", result["cpu_threads"], "Leaves CPU headroom and is reduced when available RAM is low."),
        ("Pitch Extraction", result["pitch_method"], "RMVPE on GPU when memory permits, otherwise CPU RMVPE."),
        ("Training GPU ID", str(gpu.get("training_index", gpu["index"])) if gpu and result["ready"] and gpu.get("training_index", gpu["index"]) is not None else "Confirm GPU ID / runtime first", "Use this single ID in Training Settings & Other; numeric CUDA visibility mappings are accounted for."),
        ("rmvpe_gpu process IDs", str(gpu.get("training_index", gpu["index"])) if gpu and result["ready"] and result["pitch_method"] == "rmvpe_gpu" and gpu.get("training_index", gpu["index"]) is not None else "Not needed / confirm GPU ID", "One process initially; avoid duplicate IDs on a memory-limited card."),
        ("Cache data to GPU", "No", "Leaves VRAM for model training; cache size cannot be inferred from the GPU alone."),
        ("Total Epochs", "300 (Ilaria preset)", "Not calculated from hardware; compare held-out conversions and previews."),
        ("Save frequency", "100 (Ilaria preset)", "Saves every 100 epochs; lower to 50 for more frequent recovery points. Saves take time and disk space."),
        ("Save last ckpt as final Model", "Yes", "Keeps the latest managed training checkpoint generation."),
        ("Create model at save frequency", "No", "Enable only if you also want intermediate inference exports."),
        ("Pitch Guidance / Speaker id", "True / 0", "Baseline for a normal single-voice, pitch-guided experiment."),
        ("Training previews", "Optional", "Useful for comparing epochs; CPU conversions add time. Disable for uninterrupted training."),
    ]
    lines = ["### Suggested starting settings", "",
             "**These are ILARIA's conservative estimates, not official optimal values or a measured training benchmark. No Training setting has been changed.**", ""]
    if gpu:
        lines.append("Selected GPU: **%s**. Scan time: **%s**." % (_cell(gpu["name"]), _cell(hardware.get("checked_at", "Unknown"))))
        lines.append("")
    lines.extend(["| Training setting | Suggested value | Why / how to use it |", "| --- | --- | --- |"])
    lines.extend("| %s | **%s** | %s |" % (_cell(key), _cell(value), _cell(why)) for key, value, why in rows)
    lines.extend(["", "### What to do next", "",
        "Open **Training > Training Settings & Other** and enter the suggested values manually. Use a **new Model Name** for a fresh experiment. To resume an existing checkpoint, keep its original compatible dataset, batch size, GPU count and architecture; do not apply a new preset halfway through that experiment.", "",
        "The 300-epoch target and save frequency of 100 are Ilaria's baseline for her RTX 4070 and usual **about 35 minutes of clean, expressive speech**. Your dataset length, variety, recording quality and listening preference can require different choices. Hardware alone cannot decide the best epoch, predict voice quality or produce a reliable completion time.", ""])
    lines.extend("- " + note for note in result["notes"])
    lines.extend(["", "**Sources used for the advice**", "",
                  " · ".join("[%s](%s)" % item for item in SOURCES), "",
                  "Sources explain precision, VRAM, batch size and OOM handling. The numeric batch/CPU tiers above are this app's own cautious rules, not a table published by RVC or Applio. No hardware information is uploaded; scans run locally without a training benchmark."])
    return "\n".join(lines)
