"""Optional epoch previews, isolated from the training models and CUDA context.

The trainer only exports a temporary snapshot. A separate CPU process converts
the fixed source recording, then exits and releases its models. No retrieval
index is used, so changes between previews reflect the generator being trained.
"""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


MAX_PREVIEW_SECONDS = 30.0
METHODS = ("swift", "beatrice", "fcpe", "parselmouth", "pm", "rmvpe", "harvest", "crepe")
MANIFEST = "previews.json"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent),
                                         suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _read_manifest(directory):
    try:
        data = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("previews"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"previews": [], "status": "No training previews have been generated yet."}


def prepare_preview(root, experiment_dir, enabled, source, every=10, pitch=0, method="pm"):
    """Validate UI options and freeze the source as a mono 16-kHz WAV.

    This runs before the trainer starts. The uploaded recording is copied rather
    than kept in Gradio's temporary upload folder, so later epochs and resumes
    use the same input even after that upload has expired.
    """
    if not enabled:
        return {"enabled": False}
    root, experiment_dir = Path(root).resolve(), Path(experiment_dir).resolve()
    if isinstance(every, bool) or not float(every).is_integer() or not 1 <= int(every) <= 100:
        raise ValueError("Preview frequency must be a whole number from 1 to 100 epochs.")
    if isinstance(pitch, bool) or not float(pitch).is_integer() or not -24 <= int(pitch) <= 24:
        raise ValueError("Preview Pitch must be a whole number from -24 to +24.")
    if method not in METHODS:
        raise ValueError("Choose a supported preview pitch extractor.")
    if not source:
        raise ValueError("Upload a short audio clip for Training previews (up to 30 seconds).")
    source_path = Path(os.path.expandvars(os.path.expanduser(str(source))))
    if not source_path.is_file():
        raise ValueError("The preview audio is missing. Upload the clip again.")
    # Decode with the installed PyAV runtime, but stop as soon as the limit is
    # exceeded instead of reading a possibly hours-long upload into memory.
    import av
    import numpy as np
    import soundfile as sf
    chunks, frames = [], 0
    with av.open(str(source_path.resolve()), mode="r") as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunk = converted.to_ndarray().reshape(-1)
                frames += len(chunk)
                if frames > MAX_PREVIEW_SECONDS * 16000:
                    raise ValueError("The preview clip is longer than 30 seconds. Upload a clip of 30 seconds or less.")
                chunks.append(chunk)
        for converted in resampler.resample(None):
            chunk = converted.to_ndarray().reshape(-1)
            frames += len(chunk)
            if frames > MAX_PREVIEW_SECONDS * 16000:
                raise ValueError("The preview clip is longer than 30 seconds. Upload a clip of 30 seconds or less.")
            chunks.append(chunk)
    audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    seconds = len(audio) / 16000.0
    if not math.isfinite(seconds) or seconds < 0.2:
        raise ValueError("The preview clip must contain at least 0.2 seconds of audio.")
    if seconds > MAX_PREVIEW_SECONDS:
        raise ValueError("The preview clip is %.1f seconds long. Upload a clip of 30 seconds or less." % seconds)
    if not np.isfinite(audio).all():
        raise ValueError("The preview clip contains invalid audio samples. Use another recording.")
    directory = experiment_dir / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ("source-%s.tmp.wav" % os.getpid())
    try:
        sf.write(str(temporary), audio, 16000, format="WAV", subtype="FLOAT")
        os.replace(str(temporary), str(directory / "source.wav"))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"enabled": True, "source": str(directory / "source.wav"),
            "every": int(every), "pitch": int(pitch), "method": method,
            "duration": seconds, "device": "cpu"}


def latest_preview(experiment_dir, epoch=None):
    """Return (plain-English status, audio path or None, dropdown choices).

    Gradio 3.14 uses plain-string dropdown choices. Only successful records
    whose WAV still exists are offered to the UI.
    """
    directory = Path(experiment_dir) / "previews"
    manifest = _read_manifest(directory)
    records = []
    for record in manifest["previews"]:
        if not isinstance(record, dict):
            continue
        name = record.get("file", "")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            continue
        try:
            valid_numbers = int(record["epoch"]) >= 1 and int(record.get("steps", 0)) >= 0
        except (KeyError, TypeError, ValueError):
            continue
        if valid_numbers and (directory / name).is_file():
            records.append(record)
    records.sort(key=lambda row: (int(row["epoch"]), int(row.get("steps", 0))))
    choices = ["Epoch %s | Total Steps: %s" % (row["epoch"], row.get("steps", 0))
               for row in records]
    selected = next((row for row in records if str(row["file"]) == str(epoch)
                     or str(row["epoch"]) == str(epoch)
                     or "Epoch %s | Total Steps: %s" % (row["epoch"], row.get("steps", 0)) == str(epoch)), None) if epoch is not None else None
    selected = selected or (records[-1] if records else None)
    status = manifest.get("status", "No training previews have been generated yet.")
    if selected is None:
        return status, None, choices
    if epoch is not None:
        status = "Preview: epoch %s | Total Steps: %s | Generated: %s" % (
            selected["epoch"], selected.get("steps", 0), selected.get("generated_at", "unknown"))
    return status, str(directory / selected["file"]), choices


def _export_snapshot(net_g, hps, path):
    """Build the normal RVC inference format without changing trainer tensors."""
    import torch
    model = net_g.module if hasattr(net_g, "module") else net_g
    weights = {key: value.detach().to(device="cpu", dtype=torch.float32).clone()
               for key, value in model.state_dict().items() if "enc_q" not in key}
    checkpoint = {
        "weight": weights,
        "config": [hps.data.filter_length // 2 + 1, 32, hps.model.inter_channels,
                   hps.model.hidden_channels, hps.model.filter_channels, hps.model.n_heads,
                   hps.model.n_layers, hps.model.kernel_size, hps.model.p_dropout,
                   hps.model.resblock, hps.model.resblock_kernel_sizes,
                   hps.model.resblock_dilation_sizes, hps.model.upsample_rates,
                   hps.model.upsample_initial_channel, hps.model.upsample_kernel_sizes,
                   hps.model.spk_embed_dim, hps.model.gin_channels, hps.data.sampling_rate],
        "sr": hps.sample_rate, "f0": int(hps.if_f0), "version": hps.version,
        "info": "Temporary training preview; not a resumable checkpoint.",
    }
    from ilaria_vocoder import architecture_metadata
    checkpoint.update(architecture_metadata(hps))
    torch.save(checkpoint, str(path))


class TrainingPreview:
    def __init__(self, hps, logger=None):
        self.hps, self.logger = hps, logger
        self.root = Path.cwd().resolve()
        self.directory = Path(hps.model_dir).resolve() / "previews"
        options = getattr(hps, "ilaria_preview", None)
        self.options = vars(options) if options is not None and not isinstance(options, dict) else (options or {})

    def _status(self, text, record=None):
        manifest = _read_manifest(self.directory)
        manifest["status"] = text
        if record:
            manifest["previews"] = [row for row in manifest["previews"]
                                    if isinstance(row, dict) and row.get("file") != record["file"]] + [record]
        _atomic_json(self.directory / MANIFEST, manifest)
        if self.logger:
            self.logger.info(text)

    def on_epoch(self, epoch, global_step, net_g):
        """Run at an epoch boundary on rank zero; optional failures are nonfatal."""
        if not self.options.get("enabled"):
            return None
        snapshot = job_path = None
        start = time.monotonic()
        try:
            every = int(self.options.get("every", 10))
            if every < 1 or (epoch % every and epoch != int(self.hps.total_epoch)):
                return None
            self.directory.mkdir(parents=True, exist_ok=True)
            self._status("Generating preview for epoch %s on CPU; training continues after conversion." % epoch)
            with tempfile.NamedTemporaryFile(dir=str(self.directory), suffix=".pth", delete=False) as stream:
                snapshot = Path(stream.name)
            _export_snapshot(net_g, self.hps, snapshot)
            name = "epoch_%06d_steps_%012d.wav" % (epoch, global_step)
            output = self.directory / name
            job_path = snapshot.with_suffix(".json")
            _atomic_json(job_path, {"root": str(self.root), "snapshot": str(snapshot),
                                   "source": self.options["source"], "output": str(output),
                                   "speaker": int(self.options.get("speaker", 0)),
                                   "pitch": int(self.options.get("pitch", 0)),
                                   "method": self.options.get("method", "pm")})
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--render", str(job_path)],
                                    cwd=str(self.root), env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=600, creationflags=flags)
            if result.returncode != 0 or not output.is_file():
                details = result.stdout.decode("utf-8", errors="replace").strip()
                raise RuntimeError(details[-1500:] or "The preview converter did not produce audio.")
            record = {"epoch": int(epoch), "steps": int(global_step), "file": name,
                      "generated_at": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                      "processing_seconds": time.monotonic() - start,
                      "pitch": int(self.options.get("pitch", 0)), "method": self.options.get("method", "pm")}
            self._status("Preview ready: epoch %s | Total Steps: %s | CPU conversion: %.1f s" % (
                epoch, global_step, record["processing_seconds"]), record)
            return str(output)
        except Exception as error:
            # Preview generation must not discard a completed training epoch.
            try:
                self._status("Preview failed at epoch %s; training continues. %s" % (epoch, str(error)))
            except Exception:
                if self.logger:
                    self.logger.warning("Preview failed at epoch %s: %s", epoch, error)
            return None
        finally:
            for path in (snapshot, job_path):
                try:
                    if path and path.exists():
                        path.unlink()
                except OSError:
                    pass


def render_preview(job):
    """Child-process conversion. Deliberately avoids Config() and all CUDA use."""
    root = Path(job["root"]).resolve()
    os.chdir(str(root))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "ilaria"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["weight_root"] = str(Path(job["snapshot"]).resolve().parent)
    os.environ["index_root"] = str(root / "models")
    os.environ["rmvpe_root"] = str(root / "assets" / "rmvpe")
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    # RVC inference samples latent/noise values. Repeating the same seed makes
    # successive epoch comparisons less affected by that random variation.
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    from infer.modules.vc.modules import VC
    config = SimpleNamespace(device="cpu", is_half=False, x_pad=1, x_query=6,
                             x_center=38, x_max=41, use_jit=False, dml=False)
    converter = VC(config)
    converter.get_vc(Path(job["snapshot"]).name)
    info, result = converter.vc_single(int(job.get("speaker", 0)), job["source"], int(job.get("pitch", 0)), None,
                                      job.get("method", "pm"), "", "", 0.0, 3, 0, 1.0, 0.33)
    if result is None:
        raise RuntimeError("Preview conversion failed: " + str(info))
    rate, audio = result
    if audio is None or len(audio) == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Preview conversion returned empty or invalid audio.")
    output = Path(job["output"])
    temporary = output.with_suffix(".tmp.wav")
    try:
        sf.write(str(temporary), audio, rate, format="WAV", subtype="PCM_16")
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an isolated CPU training preview.")
    parser.add_argument("--render", required=True)
    arguments = parser.parse_args()
    try:
        render_preview(json.loads(Path(arguments.render).read_text(encoding="utf-8")))
    except Exception as exception:
        print("Preview error: " + str(exception), file=sys.stderr)
        sys.exit(1)
