"""Build Windows CUDA runtimes and select the legacy runtime when needed."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


MODERN_CUDA = ("torch==2.10.0+cu128", "torchaudio==2.10.0+cu128", "torchvision==0.25.0+cu128")
PACKAGES = {
    "cu126": tuple(package.replace("cu128", "cu126") for package in MODERN_CUDA),
    "cu128": ("torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "torchvision==0.22.1+cu128"),
    "cu118": ("torch==2.0.0+cu118", "torchaudio==2.0.1+cu118", "torchvision==0.15.1+cu118"),
}
SMOKE_TEST = """
import torch
import torchaudio
import torchvision
from fairseq import checkpoint_utils
from ilaria_embedders import load_fairseq_hubert
from pathlib import Path

import sys
assert torch.version.cuda == sys.argv[1], torch.version.cuda
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Install a current NVIDIA driver and retry.")
if sys.argv[1] == "12.8" and "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError("The installed PyTorch build does not include sm_120.")
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    x = torch.randn((64, 64), device=device, dtype=torch.float32, requires_grad=True)
    (x @ x).float().mean().backward()
    torch.cuda.synchronize(index)
    print("CUDA forward/backward passed:", torch.cuda.get_device_name(index))
hubert = Path("assets/hubert/hubert_base.pt")
if hubert.is_file():
    load_fairseq_hubert(hubert)
    print("ContentVec checkpoint loaded.")
"""


def legacy_nvidia_detected():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    try:
        capabilities = [tuple(map(int, line.strip().split(".")))
                        for line in probe.stdout.splitlines() if line.strip()]
    except ValueError:
        return False
    return (probe.returncode == 0 and bool(capabilities)
            and all((5, 0) <= capability < (7, 5) for capability in capabilities))


def runtime_has_torch(runtime, backend):
    metadata = PACKAGES[backend][0].replace("==", "-") + ".dist-info"
    return (runtime / "python.exe").is_file() and (
        runtime / "Lib" / "site-packages" / metadata
    ).is_dir()


def select_windows_runtime(root):
    upgraded = root / "runtime-py310"
    updated_python = upgraded / "Scripts" / "python.exe"
    if updated_python.is_file() and (upgraded / ".cuda-verified").is_file():
        return str(updated_python)
    bundled = root / "runtime"
    legacy_gpu = legacy_nvidia_detected()
    modern = (bundled / "Lib" / "site-packages" / "torch-2.10.0+cu128.dist-info").is_dir()
    if modern and not legacy_gpu:
        return str(bundled / "python.exe")
    backend = ("cu126" if modern else "cu118") if legacy_gpu else "cu128"
    if runtime_has_torch(bundled, backend):
        return str(bundled / "python.exe")
    if backend == "cu128":
        raise RuntimeError("Extract the full Windows CUDA 12.8 package to a new folder.")

    legacy = root / f"runtime-{backend}"
    if not runtime_has_torch(legacy, backend):
        log_path = root / f"{backend}-runtime-install.log"
        print(f"Preparing {backend} for your NVIDIA GPU. See", log_path, flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--legacy", backend],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"{backend} setup failed. See {log_path}.") from error
    return str(legacy / "python.exe")


def portable_runtime_ignore(directory, names):
    ignored = {name for name in names if name == "__pycache__"}
    if Path(directory).name.lower() == "site-packages":
        replaced = {"torch", "torchgen", "functorch", "torchaudio", "torchvision",
                    "torch_directml", "torchvision.libs", "torchaudio.libs", "xformers"}
        ignored.update(name for name in names if name.lower() in replaced or (
            name.lower().endswith(".dist-info")
            and name.lower().partition("-")[0] in replaced
        ))
    return ignored


def build_runtime(root, target, backend="cu128"):
    cuda_version = {"cu128": "12.8", "cu126": "12.6", "cu118": "11.8"}[backend]
    source = root / "runtime"
    if os.name != "nt" or sys.version_info[:2] != ((3, 10) if backend == "cu126" else (3, 9)):
        raise RuntimeError("The CUDA runtime build requires the corresponding Windows portable Python runtime.")
    if target.exists():
        raise RuntimeError("The target runtime already exists. Choose an empty release folder.")
    if not (source / "python.exe").is_file():
        raise FileNotFoundError("The original portable runtime is missing.")
    with TemporaryDirectory(prefix=f".runtime-{backend}-build-", dir=target.parent) as temporary:
        candidate = Path(temporary) / "runtime"
        print("Copying the portable runtime...", flush=True)
        shutil.copytree(source, candidate, ignore=portable_runtime_ignore)
        python = str(candidate / "python.exe")
        print(f"Downloading and installing CUDA {cuda_version} packages...", flush=True)
        subprocess.run(
            [python, "-m", "pip", "--disable-pip-version-check", "--isolated", "install", "--no-cache-dir",
             "--progress-bar", "on",
             "--index-url", f"https://download.pytorch.org/whl/{backend}",
             "--extra-index-url", "https://pypi.org/simple", *PACKAGES[backend]],
            cwd=root, check=True,
        )
        subprocess.run(
            [python, "-c", "import sys; sys.path.insert(0, 'ilaria'); " + SMOKE_TEST, cuda_version],
            cwd=root, check=True,
        )
        candidate.rename(target)
    print(f"CUDA {cuda_version} runtime ready:", target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a CUDA 12.8 runtime for a clean Windows release.")
    parser.add_argument("release_folder", type=Path, nargs="?")
    parser.add_argument("--legacy", nargs="?", const="cu118", choices=("cu118", "cu126"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.legacy:
        build_runtime(root, root / f"runtime-{args.legacy}", args.legacy)
    else:
        if args.release_folder is None:
            parser.error("release_folder is required")
        destination = args.release_folder.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        build_runtime(root, destination / "runtime")
