"""UI-side ownership of training jobs and cooperative stop requests."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import threading


def experiment_path(root, name):
    name = str(name or "").strip()
    if not name or name in (".", "..") or re.search(r'[\\/:*?"<>|\x00-\x1f]', name):
        raise ValueError("Enter a simple Model Name without a folder path or filename characters.")
    if name.endswith((".", " ")):
        raise ValueError("Model Name cannot end with a dot or a space.")
    base = Path(root).resolve() / "logs"
    result = (base / name).resolve()
    if result.parent != base.resolve():
        raise ValueError("Model Name must identify one folder inside logs.")
    return result


class TrainingSession:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.job = None

    def _prune(self):
        if self.job and self.job.get("process") is not None and self.job["process"].poll() is not None:
            self.job = None

    @contextmanager
    def start(self, name):
        folder = experiment_path(self.root, name)
        with self.lock:
            self._prune()
            if self.job is not None:
                raise ValueError("Training is already running for %s. Stop and save it before starting another job." % self.job["name"])
            folder.mkdir(parents=True, exist_ok=True)
            request = folder / "stop-request.json"
            if request.exists():
                request.unlink()
            job = {"name": folder.name, "folder": folder, "process": None}
            self.job = job
        try:
            yield job
        finally:
            with self.lock:
                # A browser disconnect must not allow a second live trainer.
                if self.job is job and (job["process"] is None or job["process"].poll() is not None):
                    self.job = None

    def attach(self, process, log_offset=0):
        with self.lock:
            if self.job is None:
                raise RuntimeError("The training session was not registered.")
            self.job["log_offset"] = log_offset
            self.job["process"] = process

    def active_training(self):
        with self.lock:
            self._prune()
            if self.job is None or self.job["process"] is None:
                return None
            return self.job["name"], self.job["log_offset"]

    def request_stop(self):
        with self.lock:
            self._prune()
            if self.job is None:
                return "No training job is running. Start Train Pth first."
            target = self.job["folder"] / "stop-request.json"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(target.parent), suffix=".tmp", delete=False) as stream:
                    temporary = stream.name
                    json.dump({"action": "stop_after_epoch"}, stream)
                os.replace(temporary, str(target))
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
            return "Stop requested for %s. Finishing the current epoch, then saving the training checkpoints. Keep the application open until Training Log says STOPPED! Training saved." % self.job["name"]
