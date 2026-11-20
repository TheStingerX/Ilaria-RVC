"""Convert tensor weights on CPU and retain the metadata of RVC exports."""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp, TemporaryDirectory
from threading import Lock

_METADATA = "ilaria_rvc_checkpoint"
_dependency_lock = Lock()


def _ensure_safetensors(app_root):
    """Install the optional wheel outside the bundled Python environment."""
    with _dependency_lock:
        try:
            importlib.import_module("safetensors")
            return
        except ModuleNotFoundError as error:
            if error.name != "safetensors":
                raise

        dependencies = app_root / "assets" / "weight_converter" / "dependencies"
        if not dependencies.is_dir():
            dependencies.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".safetensors-", dir=dependencies.parent) as temporary:
                stage = Path(temporary) / "dependencies"
                result = subprocess.run(
                    [sys.executable, "-I", "-m", "pip", "--isolated", "install",
                     "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
                     "--no-deps", "--no-input", "--disable-pip-version-check",
                     "--target", str(stage), "safetensors==0.8.0"],
                    capture_output=True, text=True, errors="replace", timeout=180,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode:
                    raise RuntimeError("Could not download Safetensors. Check your internet connection and retry.")
                stage.rename(dependencies)
        sys.path.insert(0, str(dependencies))
        importlib.invalidate_caches()
        importlib.import_module("safetensors")



def convert_model_file(upload, output_root):
    """Yield status and the downloadable file after successful conversion."""
    output_dir = None
    try:
        if not upload:
            raise ValueError("Drop a .pth or .safetensors file first.")
        source = Path(upload.name)
        original_name = Path(str(getattr(upload, "orig_name", None) or source.name).replace("\\", "/")).name
        suffix = source.suffix.lower()
        if suffix not in {".pth", ".safetensors"}:
            raise ValueError("Choose a .pth or .safetensors file.")

        yield "5% | Preparing Safetensors...", None
        _ensure_safetensors(Path(output_root).resolve().parents[1])

        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        yield "10% | Reading weights on CPU...", None
        if suffix == ".pth":
            if tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 6):
                raise ValueError("PyTorch 2.6 or newer is required to read uploaded .pth files.")
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Expected a dictionary of weights.")
            wrapper = next((key for key in ("weight", "state_dict") if isinstance(checkpoint.get(key), dict)), None)
            weights = checkpoint[wrapper] if wrapper else checkpoint
            extra = {key: value for key, value in checkpoint.items() if key != wrapper} if wrapper else {}
            if not weights or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                                  for key, value in weights.items()):
                raise ValueError("Expected a voice export or a plain dictionary of tensors.")
            try:
                metadata = json.dumps({"wrapper": wrapper, "extra": extra}, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("This checkpoint contains training state or unsupported metadata. Export a voice model first.") from error
            target_suffix = ".safetensors"
        else:
            with safe_open(source, framework="pt", device="cpu") as model:
                weights = {key: model.get_tensor(key) for key in model.keys()}
                metadata = model.metadata() or {}
                stored = metadata.get(_METADATA)
            if not weights:
                raise ValueError("The file contains no tensors.")
            checkpoint = weights
            if stored:
                schema = json.loads(stored)
                wrapper = schema["wrapper"]
                extra = schema["extra"]
                if wrapper not in (None, "weight", "state_dict") or not isinstance(extra, dict):
                    raise ValueError("The checkpoint metadata is invalid.")
                if wrapper:
                    checkpoint = {**extra, wrapper: weights}
            elif "config" in metadata and "emb_g.weight" in weights:
                config = json.loads(metadata["config"])
                phone = weights.get("enc_p.emb_phone.weight")
                if not isinstance(config, list) or len(config) != 18 or phone is None or phone.ndim != 2:
                    raise ValueError("Invalid RVC configuration in the Safetensors metadata.")
                version = metadata.get("version", {256: "v1", 768: "v2"}.get(phone.shape[1]))
                if version not in ("v1", "v2"):
                    raise ValueError("Could not identify the RVC model version.")
                f0 = int("enc_p.emb_pitch.weight" in weights)
                if "f0" in metadata:
                    flag = metadata["f0"].strip().casefold()
                    if flag not in {"0", "1", "false", "true"}:
                        raise ValueError("Invalid pitch guidance in the Safetensors metadata.")
                    f0 = int(flag in {"1", "true"})
                checkpoint = {key: metadata[key] for key in ("sr", "info", "vocoder", "embedder_model") if key in metadata}
                checkpoint.update(config=config, weight=weights, version=version, f0=f0)
            target_suffix = ".pth"

        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(mkdtemp(prefix="converted-", dir=output_root))
        target = output_dir / (Path(original_name).stem + target_suffix)
        yield "65% | Writing the converted file...", None
        if suffix == ".pth":
            contiguous = {key: value.detach().contiguous().clone() for key, value in weights.items()}
            save_file(contiguous, str(target), metadata={_METADATA: metadata})
        else:
            torch.save(checkpoint, target)
        status = f"100% | Ready to download: {target.name}"
        if suffix == ".safetensors" and not (isinstance(checkpoint.get("config"), list)
                                              and isinstance(checkpoint.get("weight"), dict)):
            status += " | Tensor weights only. This file lacks the voice configuration required for RVC Inference."
        yield status, str(target)
    except ImportError:
        yield "Conversion failed. Install the safetensors dependency in the Ilaria RVC runtime.", None
    except Exception as error:
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)
        yield f"Conversion failed: {str(error).splitlines()[0] if str(error) else type(error).__name__}", None



def move_converted_model(result, output_root, models_root):
    """Install a converted voice without replacing an existing model."""
    if not result:
        return "Convert a voice to .pth first."
    try:
        source = Path(result).resolve(strict=True)
        if not source.is_relative_to(Path(output_root).resolve()) or source.suffix.lower() != ".pth":
            return "Only a converted .pth voice can be moved to models."
        import torch

        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if not (isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), list)
                and len(checkpoint["config"]) >= 3 and isinstance(checkpoint.get("weight"), dict)
                and "emb_g.weight" in checkpoint["weight"]):
            return "This .pth contains only weights or is missing required RVC data."
        destination = Path(models_root) / source.name
        try:
            with destination.open("xb") as output:
                try:
                    with source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output, length=1024 * 1024)
                except BaseException:
                    output.close()
                    destination.unlink()
                    raise
        except FileExistsError:
            return f"A model named {destination.name} already exists."
        source.unlink()
        return f"Moved to models: {destination.name}. Click Refresh in Inference."
    except FileNotFoundError:
        return "The converted file is no longer available. Convert it again."
    except Exception as error:
        return f"Could not move the model: {error}"
