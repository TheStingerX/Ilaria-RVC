"""Epoch-boundary training checkpoints. No inference exports or UI dependencies."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import uuid

import numpy as np
import torch
import torch.distributed as dist

STATE_DIR = "training-state"
STOP_FILE = "stop-request.json"
_GENERATION = re.compile(r"e\d+-s\d+-[a-f0-9]{32}\Z")


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def training_signature(hps, world_size):
    """Allow a larger target epoch count, but never resume a different experiment."""
    filelist = Path(hps.data.training_files).read_bytes()
    signature = {
        "sample_rate": hps.sample_rate, "version": hps.version,
        "if_f0": int(hps.if_f0), "model": _plain(hps.model),
        "data": {key: _plain(value) for key, value in hps.data.items()
                 if key != "training_files"},
        "batch_size": int(hps.train.batch_size), "world_size": int(world_size),
        "filelist_sha256": hashlib.sha256(filelist).hexdigest(),
    }
    # Preserve compatibility with old default-architecture checkpoints.
    for key, default in (("vocoder", "HiFi-GAN"), ("embedder_model", "contentvec")):
        value = getattr(hps.train, key, default)
        if value != default:
            signature[key] = value
    return signature


def has_training_state(model_dir):
    return (Path(model_dir) / STATE_DIR / "latest.json").is_file()


def _model(model):
    return model.module if hasattr(model, "module") else model


def capture_rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device())
    return result


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], torch.cuda.current_device())


def distributed():
    return dist.is_available() and dist.is_initialized()


def epoch_stop_requested(model_dir, rank):
    """One rank decides; every rank stops at the same completed epoch."""
    value = [(Path(model_dir) / STOP_FILE).is_file() if rank == 0 else None]
    if distributed():
        dist.broadcast_object_list(value, src=0)
    return bool(value[0])


def gather_rng_states():
    state = capture_rng_state()
    if not distributed():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def share_boundary_error(error, rank):
    """A disk/export failure must stop all ranks instead of stranding a DDP peer."""
    value = [str(error) if error else None]
    if distributed():
        dist.broadcast_object_list(value, src=0)
    if value[0]:
        raise RuntimeError("Training checkpoint/export failed: " + value[0])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_tensor_file(path, value):
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_generation(root, generation):
    if isinstance(generation, str) and _GENERATION.fullmatch(generation):
        target = root / generation
        # A generation is an application-created directory, never a supplied path.
        if target.exists() and target.resolve().parent == root.resolve() and not target.is_symlink():
            shutil.rmtree(str(target))


def save_training_state(model_dir, nets, optims, schedulers, scaler,
                        completed_epoch, global_step, rng_states, signature,
                        keep_latest=True, batch_offset=0):
    """Commit all three files together; a failed save leaves the old pair usable."""
    if int(completed_epoch) < 0 or int(global_step) < 0 or int(batch_offset) < 0:
        raise ValueError("Epoch and step counts must not be negative.")
    root = Path(model_dir) / STATE_DIR
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest.json"
    previous = None
    if latest.is_file():
        previous = json.loads(latest.read_text(encoding="utf-8")).get("generation")
    generation = "e%s-s%s-%s" % (completed_epoch, global_step, uuid.uuid4().hex)
    destination = root / generation
    destination.mkdir()
    temporary = root / ("latest-" + uuid.uuid4().hex + ".tmp")
    committed = False
    try:
        for name, net, optimizer in zip(("G.pth", "D.pth"), nets, optims):
            _write_tensor_file(destination / name, {
                "model": _model(net).state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": int(completed_epoch), "global_step": int(global_step),
            })
        _write_tensor_file(destination / "training.pt", {
            "schema": 1, "completed_epoch": int(completed_epoch),
            "global_step": int(global_step), "signature": signature,
            "batch_offset": int(batch_offset),
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "scaler": scaler.state_dict(), "rng_states": rng_states,
        })
        manifest = {"schema": 1, "generation": generation,
                    "completed_epoch": int(completed_epoch), "global_step": int(global_step),
                    "batch_offset": int(batch_offset),
                    "files": {name: _sha256(destination / name)
                              for name in ("G.pth", "D.pth", "training.pt")}}
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(latest))
        committed = True
    finally:
        if temporary.exists():
            temporary.unlink()
        if not committed:
            _remove_generation(root, generation)
    if keep_latest and previous != generation:
        try:
            _remove_generation(root, previous)
        except OSError:
            pass  # Retaining an old valid pair does not invalidate the new commit.
    return manifest


def load_training_state(model_dir, nets, optims, expected_signature):
    """Verify the complete pair before applying any restored model/optimizer state."""
    root = Path(model_dir) / STATE_DIR
    manifest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    generation = manifest.get("generation")
    if manifest.get("schema") != 1 or not isinstance(generation, str) or not _GENERATION.fullmatch(generation):
        raise ValueError("Invalid training-state manifest. Keep the experiment intact or use a new Model Name.")
    directory = root / generation
    if directory.resolve().parent != root.resolve() or directory.is_symlink():
        raise ValueError("Training checkpoint folder is outside this experiment.")
    names = ("G.pth", "D.pth", "training.pt")
    if set(manifest.get("files", {})) != set(names):
        raise ValueError("The training checkpoint pair is incomplete.")
    for name in names:
        path = directory / name
        if not path.is_file() or path.is_symlink() or _sha256(path) != manifest["files"][name]:
            raise ValueError("Training checkpoint is missing or damaged: " + name)
    state = torch.load(str(directory / "training.pt"), map_location="cpu")
    if state.get("schema") != 1 or state.get("signature") != expected_signature:
        raise ValueError("Saved training does not match these settings/data. Keep the original dataset, batch size, GPU count, sample rate and model architecture, or use a new Model Name.")
    epoch, steps = state.get("completed_epoch"), state.get("global_step")
    offset = state.get("batch_offset", 0)
    if type(offset) is not int or offset < 0 or offset != manifest.get("batch_offset", 0):
        raise ValueError("Saved partial-epoch batch cursor is inconsistent.")
    if (type(epoch) is not int or type(steps) is not int or epoch < 0 or steps < 0
            or epoch != manifest.get("completed_epoch") or steps != manifest.get("global_step")):
        raise ValueError("Saved epoch/step metadata is inconsistent.")
    checkpoints = [torch.load(str(directory / name), map_location="cpu") for name in names[:2]]
    for checkpoint, net in zip(checkpoints, nets):
        if checkpoint.get("iteration") != epoch or checkpoint.get("global_step") != steps:
            raise ValueError("Generator and discriminator checkpoint counts do not match.")
        current, stored = _model(net).state_dict(), checkpoint["model"]
        if set(current) != set(stored) or any(current[key].shape != stored[key].shape for key in current):
            raise ValueError("Saved checkpoint architecture does not match the selected model.")
    for checkpoint, net, optimizer in zip(checkpoints, nets, optims):
        _model(net).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
    return state


def restore_runtime_state(state, schedulers, scaler, rank):
    for scheduler, saved in zip(schedulers, state["schedulers"]):
        scheduler.load_state_dict(saved)
    # Precision can change on resume. A fresh FP16 scaler starts safely when the
    # previous run used FP32; disabling scaling naturally ignores FP16 scale data.
    if scaler.is_enabled() and state["scaler"]:
        scaler.load_state_dict(state["scaler"])
    states = state["rng_states"]
    if rank >= len(states):
        raise ValueError("Checkpoint GPU count does not match this training run.")
    restore_rng_state(states[rank])
