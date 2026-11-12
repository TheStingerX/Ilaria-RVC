"""Manage HuBERT downloads and local custom embedder pairs."""

import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.request import Request, urlopen


REVISION = "3ff370b7023086f57e200ef86a9a251ddf4da472"
ASSETS_URL = f"https://huggingface.co/IAHispano/Applio/resolve/{REVISION}/Resources/embedders"
HUBERT_EMBEDDERS = {
    "spin": ("spin", 378356791, "057f12bfda54e2d486d86a52a3beb2a07c96a888bc6ac0c382c12ac18dbd500c"),
    "chinese-hubert-base": ("chinese_hubert_base", 377552987, "2fefccd26c2794a583b80f6f7210c721873cb7ebae2c1cde3baf9b27855e24d8"),
    "japanese-hubert-base": ("japanese_hubert_base", 377554841, "6c023ccb71e4c2b5a324c94fc5ebe12403d3081c5f370df229892419996fd113"),
    "korean-hubert-base": ("korean_hubert_base", 377554841, "931f6232879f8eadf7dbd9e00e1fa4cac61ad269af89d509b2ed75009b1a02c5"),
}
INFERENCE_EMBEDDERS = (
    "contentvec", "spin", "spin-v2", "chinese-hubert-base",
    "japanese-hubert-base", "korean-hubert-base",
)
MAX_CONFIG_BYTES = 256 * 1024
MAX_WEIGHTS_BYTES = 1024**3
_install_lock = Lock()


def _root(root):
    return Path(root).resolve() / "assets" / "hubert" / "embedders"


def _folder_name(value):
    name = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", name):
        raise ValueError("Use a folder name of up to 64 letters, numbers, spaces, underscores or dashes.")
    if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", name):
        raise ValueError("Choose a different folder name. Windows reserves this name.")
    return name


def _pair(directory):
    return (directory / "config.json").is_file() and (directory / "pytorch_model.bin").is_file()


def _validate_config(path):
    with path.open("rb") as config:
        raw = config.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("The embedder configuration is too large.")
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("model_type") != "hubert" or config.get("hidden_size") != 768:
        raise ValueError("Choose a HuBERT base config.json with 768 hidden features.")


def custom_embedder_names(root):
    directory = _root(root) / "custom"
    if not directory.is_dir():
        return []
    names = []
    for folder in directory.iterdir():
        if folder.is_dir() and folder.resolve().parent == directory.resolve() and _pair(folder):
            try:
                name = _folder_name(folder.name)
            except ValueError:
                continue
            names.append(name)
    return sorted(names, key=str.casefold)


def install_custom_embedder(root, name, weights_path, config_path):
    """Install both files in one rename after checking compatibility."""
    name = _folder_name(name)
    weights, config = Path(weights_path), Path(config_path)
    if weights.suffix.lower() != ".bin" or config.suffix.lower() != ".json":
        raise ValueError("Upload both a HuBERT .bin weights file and its .json configuration.")
    if not 0 < weights.stat().st_size <= MAX_WEIGHTS_BYTES:
        raise ValueError("Custom embedder weights must be nonempty and at most 1 GiB.")
    directory = _root(root) / "custom"
    with _install_lock:
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / name
        if destination.exists():
            raise ValueError("An embedder with this folder name already exists.")
        with TemporaryDirectory(prefix=".embedder-", dir=directory) as temporary:
            stage = Path(temporary) / "files"
            stage.mkdir()
            shutil.copyfile(config, stage / "config.json")
            _validate_config(stage / "config.json")
            shutil.copyfile(weights, stage / "pytorch_model.bin")
            from ilaria_hf_hubert import load_hf_hubert

            load_hf_hubert(stage, "cpu", False)
            stage.rename(destination)
    return name


_download_locks = {name: Lock() for name in HUBERT_EMBEDDERS}


def download_builtin_embedder(root, name):
    with _download_locks[name]:
        return _download_builtin_embedder(root, name)


def _download_builtin_embedder(root, name):
    source, size, expected = HUBERT_EMBEDDERS[name]
    directory = _root(root)
    destination = directory / name
    if _pair(destination):
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"The {name} folder is incomplete. Restore both embedder files.")
    with TemporaryDirectory(prefix=f".{name}-", dir=directory) as temporary:
        stage = Path(temporary) / "files"
        stage.mkdir()
        for filename, limit in (("config.json", MAX_CONFIG_BYTES), ("pytorch_model.bin", size)):
            url = f"{ASSETS_URL}/{source}/{filename}"
            checksum = hashlib.sha256()
            received = 0
            with urlopen(Request(url, headers={"User-Agent": "Ilaria-RVC"}), timeout=75) as response:
                with (stage / filename).open("xb") as output:
                    for chunk in iter(lambda: response.read(4 * 1024**2), b""):
                        received += len(chunk)
                        if received > limit:
                            raise ValueError(f"The {name} download exceeds the expected size.")
                        output.write(chunk)
                        checksum.update(chunk)
            if filename.endswith(".bin") and (received != size or checksum.hexdigest() != expected):
                raise ValueError(f"The {name} download did not pass the checksum check.")
        _validate_config(stage / "config.json")
        try:
            stage.rename(destination)
        except OSError:
            if not _pair(destination):
                raise
    return destination


def inference_embedder_directory(root, name):
    if name in HUBERT_EMBEDDERS:
        return download_builtin_embedder(root, name)
    if name.startswith("custom:"):
        custom_name = _folder_name(name.partition(":")[2])
        if custom_name in custom_embedder_names(root):
            return _root(root) / "custom" / custom_name
        raise ValueError(f"Install and select the custom embedder '{custom_name}' first.")
    raise ValueError("Choose a supported inference embedder.")


def inference_embedder_installed(root, name):
    if name in ("contentvec", "spin-v2"):
        filename = "hubert_base.pt" if name == "contentvec" else "spin-v2.pt"
        return (_root(root).parent / filename).is_file()
    if isinstance(name, str) and name in HUBERT_EMBEDDERS:
        return _pair(_root(root) / name)
    if isinstance(name, str) and name.startswith("custom:"):
        return name.partition(":")[2] in custom_embedder_names(root)
    return False


def start_missing_hubert_downloads(root, report):
    from threading import Thread

    def install(name):
        report(f"Downloading embedder: {name}")
        try:
            download_builtin_embedder(root, name)
        except (OSError, ValueError) as error:
            report(f"Embedder download failed for {name}: {error}. Select it in Inference to retry.")
        else:
            report(f"Embedder ready: {name}")

    for name in HUBERT_EMBEDDERS:
        if not inference_embedder_installed(root, name):
            Thread(target=install, args=(name,), name=f"Download {name}", daemon=True).start()
