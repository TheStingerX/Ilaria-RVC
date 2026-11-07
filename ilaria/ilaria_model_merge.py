"""CPU-only interpolation of compatible RVC inference checkpoints (Python 3.9+)."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from collections import OrderedDict
from pathlib import Path
import math
import os
import re
import tempfile


class MergeProblem(ValueError):
    pass


def _torch():
    import torch
    return torch


def _rate(value):
    choices = {"32k": 32000, "40k": 40000, "48k": 48000}
    if isinstance(value, str) and value.lower() in choices:
        return choices[value.lower()]
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise MergeProblem("Choose a supported sample rate: 32k, 40k, or 48k.")
    if result not in choices.values():
        raise MergeProblem("Only 32k, 40k, and 48k RVC models are supported.")
    return result


from ilaria_model_paths import voice_path


def _model_path(name, root):
    try:
        return voice_path(Path(root) / "models", name)
    except ValueError as exc:
        raise MergeProblem(str(exc)) from exc


def _normal(value):
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    return value


def _load(path):
    torch = _torch()
    try:
        # RVC inference exports contain tensors and primitive metadata only.
        model = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise MergeProblem("Could not read %s as a supported inference model: %s" % (path.name, error))
    if not isinstance(model, dict) or not isinstance(model.get("weight"), dict) or not model["weight"]:
        raise MergeProblem("%s is not an inference export with a weight dictionary. G_/D_ training checkpoints cannot be merged here." % path.name)
    if not isinstance(model.get("config"), (list, tuple)) or len(model["config"]) != 18:
        raise MergeProblem("%s has a missing or unsupported RVC model configuration." % path.name)
    config = _normal(model["config"])
    rate = _rate(config[-1])
    if "sr" in model and _rate(model["sr"]) != rate:
        raise MergeProblem("%s has inconsistent sample-rate metadata." % path.name)
    version = model.get("version", "v1")
    if version not in ("v1", "v2"):
        raise MergeProblem("%s has an unsupported RVC version." % path.name)
    if model.get("f0") not in (0, 1, False, True):
        raise MergeProblem("%s has missing or invalid pitch-guidance metadata." % path.name)
    for key, value in model["weight"].items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise MergeProblem("%s contains unsupported weight values." % path.name)
        if value.layout != torch.strided or value.is_complex():
            raise MergeProblem("%s contains unsupported tensor types." % path.name)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise MergeProblem("%s contains NaN or infinite weights." % path.name)
    from ilaria_vocoder import checkpoint_vocoder, weights_for_runtime
    model["vocoder"] = checkpoint_vocoder(model)
    model["weight"] = weights_for_runtime(model["weight"])
    return model, config, rate, version, int(model["f0"])


def _compatible(a, b):
    torch = _torch()
    if a[2] != b[2]:
        raise MergeProblem("The models have different sample rates. Merge 32k with 32k, 40k with 40k, or 48k with 48k. This tool does not resample model architectures.")
    if a[3] != b[3]:
        raise MergeProblem("RVC versions differ. Both models must use the same version.")
    if a[4] != b[4]:
        raise MergeProblem("Pitch guidance differs. Both models must have the same F0 setting.")
    if a[1] != b[1]:
        raise MergeProblem("The model configurations differ (architecture or speaker layout). Choose compatible models.")
    if a[0].get("vocoder", "HiFi-GAN") != b[0].get("vocoder", "HiFi-GAN"):
        raise MergeProblem("The models use different vocoders and cannot be merged.")
    if (a[0].get("embedder_model") or "contentvec") != (b[0].get("embedder_model") or "contentvec"):
        raise MergeProblem("The models use different embedders and cannot be merged.")
    aw, bw = a[0]["weight"], b[0]["weight"]
    if set(aw) != set(bw):
        raise MergeProblem("The models have different weight keys and cannot be merged.")
    for key in aw:
        if aw[key].shape != bw[key].shape:
            raise MergeProblem("Weight shape mismatch: " + key)
        if aw[key].is_floating_point() != bw[key].is_floating_point():
            raise MergeProblem("Weight type mismatch: " + key)
        if not aw[key].is_floating_point() and not torch.equal(aw[key], bw[key]):
            raise MergeProblem("Non-floating model buffers differ: " + key)


def _describe(name, model):
    return "%s: %s kHz | RVC %s | Pitch guidance: %s | Speakers: %s | Tensors: %s" % (
        name, model[2] // 1000, model[3].upper(), "Yes" if model[4] else "No",
        model[1][-3], len(model[0]["weight"]))


def inspect_model_pair(model_a, model_b, root):
    """Yield human-readable metadata and compatibility, without loading a network."""
    progress = 0.0
    try:
        yield "Inspecting Model A... | 0.0%"
        path_a = _model_path(model_a, root)
        a = _load(path_a)
        progress = 40.0
        yield _describe(path_a.name, a) + " | 40.0%"
        path_b = _model_path(model_b, root)
        b = _load(path_b)
        progress = 80.0
        yield _describe(path_b.name, b) + " | 80.0%"
        _compatible(a, b)
        yield "Compatible models | 100.0%\n" + _describe(path_a.name, a) + "\n" + _describe(path_b.name, b) + "\nChoose %sk for the output sample rate." % (a[2] // 1000)
    except Exception as error:
        yield "Inspection failed: %s | %.1f%%" % (error, progress)


def _output_path(name, root):
    name = str(name or "").strip()
    if name.lower().endswith(".pth"):
        name = name[:-4]
    if not name or name in (".", "..") or re.search(r'[\x00-\x1f<>:"/\\|?*]', name) or name.endswith((".", " ")):
        raise MergeProblem("Use a simple output model name without folder paths or reserved characters.")
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *("COM%d" % i for i in range(1, 10)), *("LPT%d" % i for i in range(1, 10))}:
        raise MergeProblem("That output name is reserved by Windows.")
    base = (Path(root) / "models").resolve()
    path = (base / name / (name + ".pth")).resolve()
    if path.parent != base / name:
        raise MergeProblem("The output path must stay inside models.")
    if path.exists():
        raise MergeProblem("The output model already exists. Choose a new name; existing files are not overwritten.")
    return path


def merge_models(model_a, model_b, alpha, sample_rate, output_name, info, precision, root):
    """Merge compatible inference models: alpha*A + (1-alpha)*B, on CPU.

    alpha is Model A's contribution (0..1). sample_rate is auto/32k/40k/48k;
    explicit rates must match both inputs. precision is FP16 or FP32.
    """
    temporary = None
    progress = 0.0
    try:
        torch = _torch()
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise MergeProblem("Model A's contribution must be between 0 and 1.")
        requested_rate = None if str(sample_rate).lower() in ("auto", "auto (from models)") else _rate(sample_rate)
        precision = str(precision).upper()
        if precision not in ("FP16", "FP32"):
            raise MergeProblem("Choose FP16 or FP32 output precision.")
        output = _output_path(output_name, root)
        yield "Merge: loading Model A on CPU... | 0.0%"
        path_a = _model_path(model_a, root)
        a = _load(path_a)
        progress = 15.0
        yield "Merge: loading Model B on CPU... | 15.0%"
        path_b = _model_path(model_b, root)
        b = _load(path_b)
        progress = 30.0
        yield "Merge: validating architecture, sample rate and pitch guidance... | 30.0%"
        _compatible(a, b)
        if requested_rate is not None and a[2] != requested_rate:
            raise MergeProblem("Selected output rate does not match the models. Choose %sk; this tool does not convert sample rates." % (a[2] // 1000))
        result = OrderedDict()
        result["weight"] = OrderedDict()
        keys = list(a[0]["weight"])
        last_percent = -1
        for number, key in enumerate(keys, 1):
            av, bv = a[0]["weight"][key], b[0]["weight"][key]
            if av.is_floating_point():
                value = av.float().mul(alpha).add(bv.float(), alpha=1 - alpha)
                value = value.half() if precision == "FP16" else value.float()
                if not torch.isfinite(value).all():
                    raise MergeProblem("The merged weights exceed the chosen precision: " + key)
            else:
                value = av.clone()
            result["weight"][key] = value
            percent = int(35 + 50 * number / len(keys))
            if percent != last_percent or number == len(keys):
                progress = float(percent)
                yield "Merge: blending tensor %s/%s (%s)... | %.1f%%" % (number, len(keys), key, percent)
                last_percent = percent
        result["config"] = a[1]
        result["sr"] = "%sk" % (a[2] // 1000)
        result["f0"] = a[4]
        result["version"] = a[3]
        result["vocoder"] = a[0].get("vocoder", "HiFi-GAN")
        result["embedder_model"] = a[0].get("embedder_model") or "contentvec"
        result["info"] = str(info or "Merged with ILARIA RVC.").strip()
        result["merge"] = {"model_a": path_a.name, "model_b": path_b.name,
                           "model_a_ratio": alpha, "model_b_ratio": 1 - alpha,
                           "precision": precision}
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".ilaria-merge-", suffix=".tmp", dir=str(output.parent))
        os.close(handle)
        progress = 90.0
        yield "Merge: saving output model... | 90.0%"
        torch.save(result, temporary)
        progress = 95.0
        yield "Merge: verifying saved model... | 95.0%"
        verified = _load(Path(temporary))
        if verified[2:] != a[2:] or verified[1] != a[1] or set(verified[0]["weight"]) != set(result["weight"]):
            raise MergeProblem("Saved model verification failed.")
        if any(not torch.equal(verified[0]["weight"][key], value) for key, value in result["weight"].items()):
            raise MergeProblem("Saved model tensor verification failed.")
        # Windows rename is atomic and refuses to replace an existing file.
        # POSIX rename replaces existing files, so use exclusive hard linking.
        if os.name == "nt":
            os.rename(temporary, output)
        else:
            os.link(temporary, output)
        yield ("Merge done! | 100.0%%\nSaved: %s\nModel A: %.1f%% | Model B: %.1f%% | %sk | %s\nClick Refresh in Inference and select the merged model. Index files are not merged."
               % (output.relative_to(Path(root).resolve()).as_posix(), alpha * 100, (1 - alpha) * 100, a[2] // 1000, precision))
    except Exception as error:
        yield "Merge failed: %s | %.1f%%" % (error, progress)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
