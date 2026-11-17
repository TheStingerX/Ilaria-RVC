"""Prepare a separate CUDA 12.8 runtime for the Windows portable package."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


INDEX = "https://download.pytorch.org/whl/cu128"
PACKAGES = ("torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "torchvision==0.22.1+cu128")
SMOKE_TEST = """
import torch
import torchaudio
import torchvision
from fairseq import checkpoint_utils
from ilaria_embedders import load_fairseq_hubert
from pathlib import Path

assert torch.version.cuda == "12.8", torch.version.cuda
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Install a current NVIDIA driver and retry.")
if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError("The installed PyTorch build does not include sm_120.")
devices = [i for i in range(torch.cuda.device_count())
           if torch.cuda.get_device_capability(i) == (12, 0)]
if not devices:
    raise RuntimeError("No NVIDIA SM 12.0 GPU detected. The regular launcher is sufficient for this PC.")
for index in devices:
    device = torch.device(f"cuda:{index}")
    x = torch.randn((64, 64), device=device, dtype=torch.float16, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize(index)
    print("CUDA forward/backward passed:", torch.cuda.get_device_name(index))
hubert = Path("assets/hubert/hubert_base.pt")
if hubert.is_file():
    load_fairseq_hubert(hubert)
    print("ContentVec checkpoint loaded.")
"""


def blackwell_detected():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and any(
        row.partition(",")[0].strip() == "12.0" for row in probe.stdout.splitlines()
    )


def ensure_runtime(root):
    python = root / "runtime-cu128" / "python.exe"
    if python.is_file():
        return python
    if not blackwell_detected():
        return None
    print("Preparing CUDA 12.8 for your NVIDIA RTX 50 GPU. This can take a few minutes.", flush=True)
    log_path = root / "rtx50-runtime-install.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve())],
            cwd=root, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode or not python.is_file():
        message = f"RTX 50 setup failed. See {log_path} for details."
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "Ilaria RVC", 0x10)
        raise RuntimeError(message)
    return python


def install(root):
    source = root / "runtime"
    target = root / "runtime-cu128"
    if os.name != "nt" or sys.version_info[:2] != (3, 9):
        raise RuntimeError("RTX 50 setup requires the Windows Python 3.9 portable installation.")
    if target.exists():
        raise RuntimeError("runtime-cu128 already exists. Rename it before rebuilding.")
    if not (source / "python.exe").is_file():
        raise FileNotFoundError("The original portable runtime is missing.")
    with TemporaryDirectory(prefix=".runtime-cu128-build-", dir=root) as temporary:
        candidate = Path(temporary) / "runtime"
        print("Copying the portable runtime...", flush=True)
        shutil.copytree(source, candidate, ignore=shutil.ignore_patterns("__pycache__"))
        python = str(candidate / "python.exe")
        # The cloned DirectML wheel depends on the old Torch build.
        subprocess.run([python, "-m", "pip", "uninstall", "-y", "torch-directml"],
                       cwd=root, check=True)
        subprocess.run(
            [python, "-m", "pip", "--disable-pip-version-check", "--isolated", "install",
             "--index-url", INDEX, "--extra-index-url", "https://pypi.org/simple", *PACKAGES],
            cwd=root, check=True,
        )
        subprocess.run(
            [python, "-c", "import sys; sys.path.insert(0, 'ilaria'); " + SMOKE_TEST],
            cwd=root, check=True,
        )
        candidate.rename(target)
    print("RTX 50 runtime ready. Start ILARIA_RVC.bat or ILARIA_RVC_LOGS.bat.")


if __name__ == "__main__":
    install(Path(__file__).resolve().parent.parent)
