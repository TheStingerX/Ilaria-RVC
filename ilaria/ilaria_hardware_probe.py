"""Read local hardware/runtime metadata without training, benchmarks or network access.

PyTorch may initialize its CUDA metadata interface when reading device properties.
No tensor is allocated and no CUDA memory query/context is requested by this probe.
Available VRAM is taken from NVIDIA's optional management tool and is a snapshot.
"""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import csv
import datetime
import importlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

GIB = float(1024 ** 3)


def _module(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _cpu_name():
    name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    return str(name).strip() or "Unknown CPU"


def _windows_memory():
    if os.name != "nt":
        return None, None
    try:
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended_available", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / GIB, status.available / GIB
    except Exception:
        pass
    return None, None


def _nvidia_status():
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            return []
        rows = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 6:
                continue
            index, uuid, name, total, free, driver = [value.strip() for value in row]
            try:
                rows.append({"physical_index": int(index), "uuid": uuid, "name": name,
                             "total_gib": float(total) / 1024.0,
                             "free_gib": float(free) / 1024.0, "driver": driver})
            except (ValueError, TypeError):
                continue
        return rows
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return []


def _runtime_support(major, minor, architectures, cuda_version):
    """Conservative compatibility screening, never an exact-name GPU allowlist."""
    try:
        cuda_parts = tuple(int(piece) for piece in str(cuda_version).split(".")[:2])
    except ValueError:
        cuda_parts = ()
    if major >= 10 and cuda_parts and cuda_parts < (12, 8):
        return False, "This newer GPU architecture needs a compatible newer PyTorch/CUDA build; the installed CUDA runtime predates 12.8."
    native = []
    ptx = []
    for entry in architectures:
        match = re.fullmatch(r"(sm|compute)_(\d+)", str(entry))
        if match:
            value = int(match.group(2))
            (native if match.group(1) == "sm" else ptx).append(divmod(value, 10))
    target = (major, minor)
    if any(a == major and b <= minor for a, b in native):
        return True, "The installed runtime lists compatible kernels for this GPU architecture."
    if any(entry <= target for entry in ptx):
        return True, "The runtime provides forward-compatible PTX; native kernel coverage and performance are not benchmarked."
    if not architectures or major <= 9:
        return True, "CUDA sees this GPU. Native architecture coverage could not be confirmed; no training benchmark was run."
    return False, "CUDA sees this GPU, but compatible kernels could not be confirmed for this newer architecture. Check the PyTorch/CUDA build."


def _matching_management_row(properties, rows):
    uuid = getattr(properties, "uuid", None)
    if uuid:
        if isinstance(uuid, bytes):
            uuid = uuid.decode("ascii", "replace")
        matches = [row for row in rows if row["uuid"].lower() == str(uuid).lower()]
        if len(matches) == 1:
            return matches[0]
    # Names and memory identify a unique GPU. Never attach another identical
    # device's free-memory reading when CUDA_VISIBLE_DEVICES remaps indices.
    matches = [row for row in rows if row["name"] == properties.name
               and abs(row["total_gib"] - properties.total_memory / GIB) < 0.25]
    return matches[0] if len(matches) == 1 else None


def collect_hardware(root=None):
    """Return detected facts; missing optional values are None, never invented."""
    result = {
        "checked_at": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "cpu": {"name": _cpu_name(), "logical": max(1, os.cpu_count() or 1), "physical": None},
        "ram": {"total_gib": None, "available_gib": None},
        "disk": {"path": str(Path(root or (Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "ilaria" else Path(__file__).resolve().parent)).resolve()), "free_gib": None},
        "os": ("Windows (NT %s)" % platform.version()) if os.name == "nt" else platform.platform(),
        "python": platform.python_version(),
        "torch_version": None, "cuda_version": None, "cuda_available": False,
        "devices": [], "warnings": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    psutil = _module("psutil")
    if psutil:
        try:
            result["cpu"]["physical"] = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            result["ram"] = {"total_gib": memory.total / GIB, "available_gib": memory.available / GIB}
        except Exception:
            result["warnings"].append("Some CPU or RAM information could not be read.")
    if result["ram"]["total_gib"] is None:
        total, available = _windows_memory()
        result["ram"] = {"total_gib": total, "available_gib": available}
    try:
        result["disk"]["free_gib"] = shutil.disk_usage(result["disk"]["path"]).free / GIB
    except OSError:
        result["warnings"].append("Free disk space could not be read for the application folder.")
    torch = _module("torch")
    if torch is None:
        result["warnings"].append("PyTorch could not be imported; CUDA training support is unavailable to this check.")
        return result
    result["torch_version"] = str(getattr(torch, "__version__", "Unknown"))
    result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
        if not result["cuda_available"]:
            result["warnings"].append("No usable CUDA GPU was detected by this PyTorch runtime.")
            return result
        if result["cuda_version"] is None:
            result["warnings"].append("PyTorch exposes an accelerator through its CUDA API but reports no NVIDIA CUDA runtime version; this may be another backend such as ROCm.")
        try:
            architectures = torch.cuda.get_arch_list()
        except Exception:
            architectures = []
        management = _nvidia_status()
        for index in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(index)
                major, minor = int(properties.major), int(properties.minor)
                supported, reason = _runtime_support(major, minor, architectures, result["cuda_version"])
                if result["cuda_version"] is None:
                    supported, reason = False, "A compatible NVIDIA CUDA runtime could not be confirmed for this training setup."
                row = _matching_management_row(properties, management)
                visibility = result["cuda_visible_devices"]
                training_index = index
                if visibility is not None:
                    mapped = [value.strip() for value in visibility.split(",")]
                    training_index = int(mapped[index]) if index < len(mapped) and mapped[index].isdigit() else None
                result["devices"].append({
                    "index": index, "name": str(properties.name),
                    "training_index": training_index,
                    "total_gib": properties.total_memory / GIB,
                    "free_gib": row["free_gib"] if row else None,
                    "compute_capability": "%s.%s" % (major, minor),
                    "compatible": supported, "reason": reason,
                    "driver": row["driver"] if row else None,
                })
            except Exception as error:
                result["warnings"].append("GPU %s could not be inspected: %s" % (index, error))
    except Exception as error:
        result["warnings"].append("CUDA inspection failed: " + str(error))
    return result
