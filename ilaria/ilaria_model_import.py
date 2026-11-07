"""Hugging Face download and selective RVC model import (Python 3.9+)."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener
from zipfile import ZipFile, BadZipFile

CHUNK = 4 * 1024 * 1024
MAX_BYTES = 16 * 1024 ** 3


class ImportProblem(ValueError):
    pass


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ImportProblem("The download redirected to an unsupported non-HTTPS address.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url):
    from time import sleep
    for attempt in range(3):
        try:
            return build_opener(HTTPSRedirects()).open(
                Request(url, headers={"User-Agent": "ILARIA-RVC/1.0"}), timeout=60)
        except HTTPError:
            raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            sleep(0.5 * (attempt + 1))


def _clean_name(name):
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in (".", "..") or re.search(r'[\x00-\x1f<>:"/\\|?*]', name):
        raise ImportProblem("A model filename contains unsupported characters.")
    if name.endswith((".", " ")) or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *("COM%d" % i for i in range(1, 10)), *("LPT%d" % i for i in range(1, 10))}:
        raise ImportProblem("A model filename is reserved by Windows: " + name)
    return name


def _accepted(name):
    low = str(name).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return low.endswith((".pth", ".index")) and not (low.endswith(".pth") and re.match(r"[gd]_\d+\.pth$", low))


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.digest()


def _zip_manifest(path):
    selected = []
    seen = set()
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 20000:
            raise ImportProblem("The ZIP contains too many entries.")
        for member in members:
            normalized = member.filename.replace("\\", "/")
            parts = PurePosixPath(normalized)
            if parts.is_absolute() or ".." in parts.parts or ":" in normalized or stat.S_ISLNK(member.external_attr >> 16):
                raise ImportProblem("The ZIP contains an unsafe path or symbolic link.")
            if member.is_dir() or not _accepted(normalized):
                continue
            if member.flag_bits & 1:
                raise ImportProblem("Password-protected ZIP archives are not supported.")
            if normalized in seen:
                raise ImportProblem("The ZIP contains duplicate file entries: " + normalized)
            seen.add(normalized)
            selected.append((member.filename, _clean_name(parts.name), member.file_size))
    return selected


def _input_file(value):
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        return path, path.name
    if isinstance(value, dict):
        path = Path(value.get("name") or value.get("path") or "")
        return path, value.get("orig_name") or path.name
    path = Path(value.name)
    return path, getattr(value, "orig_name", None) or path.name


def _index_group(name, model_names, fallback):
    low = Path(name).stem.casefold()
    matches = [stem for stem in model_names if stem.casefold() == low or "_" + stem.casefold() + "_" in "_" + low + "_"]
    if matches:
        return max(matches, key=len)
    if len(model_names) == 1:
        return model_names[0]
    # Standard added_IVF..._<experiment>_v2.index filenames can name their model.
    inferred = re.match(r"(?:added|trained)_IVF.*?_Flat_nprobe_\d+_(.+)_v[12]$", Path(name).stem, re.I)
    return inferred.group(1) if inferred else fallback


def _install_events(items, root, staged_roots=()):
    if not items:
        raise ImportProblem("No inference .pth or .index files found. Training G_/D_ checkpoints are skipped.")
    root = Path(root).resolve()
    models = list(dict.fromkeys(Path(name).stem for _, name, _ in items if name.lower().endswith(".pth")))
    owned_folders = {Path(path).resolve() for path in staged_roots}
    plan = {}
    for position, (source, name, fallback) in enumerate(items, 1):
        yield 45 + 15 * (position - 1) / len(items), "Checking filename conflicts: " + name, False
        destination = (root / "models" / Path(name).stem / name if name.lower().endswith(".pth")
                       else root / "models" / _clean_name(_index_group(name, models, fallback)) / name)
        destination = destination.resolve()
        if root not in destination.parents:
            raise ImportProblem("An installation path leaves the application folder.")
        key = str(destination).casefold()
        if key in plan:
            if _hash(plan[key][0]) != _hash(source):
                raise ImportProblem("Conflicting files share the filename " + name)
        else:
            if destination.exists() and (destination.stat().st_size != source.stat().st_size
                                         or _hash(destination) != _hash(source)):
                raise ImportProblem("A different file already exists: " + str(destination.relative_to(root)) + ". Rename the new file or remove the conflict manually.")
            plan[key] = (source, destination)
        yield 45 + 15 * position / len(items), "Checking filename conflicts: " + name, False
    imported, unchanged, created = 0, 0, []
    total_bytes = sum(source.stat().st_size for source, _ in plan.values())
    completed_bytes, previous = 0, -1
    try:
        for source, destination in plan.values():
            size = source.stat().st_size
            if destination.exists():
                unchanged += 1
                completed_bytes += size
                yield min(99.9, 60 + 39.9 * completed_bytes / max(1, total_bytes)), "Keeping identical file: " + destination.name, False
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Extracted files are already verified by ZIP CRC. On the same
            # volume, link them into place without reading/writing them again.
            # Only use owned staging files, never original uploaded files.
            if source.parent.resolve() in owned_folders:
                try:
                    os.link(source, destination)
                except FileExistsError:
                    raise
                except OSError:
                    pass  # Cross-volume/non-NTFS: retain the normal copy path.
                else:
                    created.append(destination)
                    imported += 1
                    completed_bytes += size
                    yield min(99.9, 60 + 39.9 * completed_bytes / max(1, total_bytes)), "Installing " + destination.name + " (already extracted)", False
                    continue
            with destination.open("xb") as output:
                created.append(destination)
                with source.open("rb") as input_stream:
                    while True:
                        chunk = input_stream.read(CHUNK)
                        if not chunk:
                            break
                        output.write(chunk)
                        completed_bytes += len(chunk)
                        progress = min(99.9, 60 + 39.9 * completed_bytes / max(1, total_bytes))
                        if int(progress) != previous or completed_bytes == total_bytes:
                            previous = int(progress)
                            yield progress, "Installing %s (%0.1f / %0.1f MiB)" % (destination.name, completed_bytes / (1024 * 1024), total_bytes / (1024 * 1024)), False
            imported += 1
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    yield 100.0, "Import complete! %d file(s) imported; %d identical file(s) already present. Click Refresh in Inference to view models." % (imported, unchanged), True


def _friendly(error):
    if isinstance(error, HTTPError):
        if error.code in (401, 403):
            return "The repository is private/gated or access was denied. Use a public model or import downloaded files."
        if error.code == 404:
            return "Hugging Face could not find that repository, revision, or file. Check the URL."
        return "Hugging Face returned HTTP %s. Try again later." % error.code
    if isinstance(error, URLError):
        if "getaddrinfo failed" in str(error.reason) or "11001" in str(error.reason):
            return "Could not resolve huggingface.co after 3 attempts (DNS/network error). Check your internet connection or VPN/proxy, then retry. Alternatively, download the ZIP in your browser and use Other > Import local models."
        return "Could not connect to Hugging Face after 3 attempts: " + str(error.reason)
    if isinstance(error, BadZipFile):
        return "The file is not a valid ZIP archive or is incomplete."
    return str(error)


def _import_events(files, root, owned_upload_root=None):
    if not files:
        raise ImportProblem("Drop one or more .zip, .pth, or .index files first.")
    if not isinstance(files, (list, tuple)):
        files = [files]
    yield 0.0, "Preparing model import...", False
    inputs = []
    total_bytes = 0
    for number, value in enumerate(files, 1):
        path, original_name = _input_file(value)
        name = _clean_name(original_name)
        if not path.is_file():
            raise ImportProblem("An uploaded file is missing. Upload it again.")
        if name.lower().endswith(".zip"):
            manifest = _zip_manifest(path)
            total_bytes += sum(size for _, _, size in manifest)
        elif _accepted(name):
            manifest = None
            total_bytes += path.stat().st_size
        else:
            raise ImportProblem("Unsupported file: " + name + ". Use an inference .pth, .index, or ZIP.")
        if total_bytes > MAX_BYTES:
            raise ImportProblem("The selected model files exceed the 16 GiB import limit.")
        inputs.append((path, name, manifest))
        yield 5 * number / len(files), "Inspecting files: " + name, False
    with tempfile.TemporaryDirectory(prefix="ilaria-import-", dir=owned_upload_root) as temporary:
        stage = Path(temporary)
        items, completed_bytes, previous = [], 0, -1
        last_report = 0.0
        for path, name, manifest in inputs:
            if manifest is None:
                items.append((path, name, Path(name).stem))
                completed_bytes += path.stat().st_size
                yield 5 + 40 * completed_bytes / max(1, total_bytes), "Prepared file: " + name, False
                continue
            with ZipFile(path) as archive:
                for internal, filename, size in manifest:
                    target = stage / ("model-%s" % len(items))
                    with archive.open(internal) as src, target.open("wb") as dest:
                        while True:
                            chunk = src.read(CHUNK)
                            if not chunk:
                                break
                            dest.write(chunk)
                            completed_bytes += len(chunk)
                            progress = 5 + 40 * completed_bytes / max(1, total_bytes)
                            if (int(progress) != previous and time.monotonic() - last_report >= 0.2) or completed_bytes == total_bytes:
                                last_report = time.monotonic()
                                previous = int(progress)
                                yield progress, "Extracting %s (%0.1f / %0.1f MiB)" % (filename, completed_bytes / (1024 * 1024), total_bytes / (1024 * 1024)), False
                    items.append((target, filename, Path(name).stem))
        owned = (stage,) if owned_upload_root is None else (stage, owned_upload_root)
        yield from _install_events(items, root, staged_roots=owned)


def _status(percent, detail, done=False):
    return (detail + " | 100.0%") if done else ("%.1f%% | %s" % (min(99.9, percent), detail))


def import_model_files(files, root):
    """Import files with phase/byte progress. Nothing is loaded or executed."""
    progress = 0.0
    try:
        for progress, detail, done in _import_events(files, root):
            yield _status(progress, detail, done)
    except Exception as error:
        yield _status(progress, "Import failed: " + _friendly(error))


def _resolve_huggingface_zip(url):
    parsed = urlparse(str(url).strip())
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ImportProblem("Use an https://huggingface.co model repository or ZIP download URL.")
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 2 or parts[0] in ("datasets", "spaces", "api"):
        raise ImportProblem("Use a Hugging Face model repository URL.")
    repo = "/".join(parts[:2])
    if len(parts) >= 5 and parts[2] in ("resolve", "blob"):
        filename = "/".join(parts[4:])
        if not filename.lower().endswith(".zip"):
            raise ImportProblem("The download URL must point to a .zip file.")
        return "https://huggingface.co/%s/resolve/%s/%s?download=true" % (repo, quote(parts[3], safe=""), quote(filename, safe="/")), _clean_name(Path(filename).name)
    revision = parts[3] if len(parts) >= 4 and parts[2] == "tree" else "main"
    prefix = "/".join(parts[4:]).rstrip("/") if len(parts) > 4 and parts[2] == "tree" else ""
    if len(parts) > 2 and parts[2] != "tree":
        raise ImportProblem("Use the repository page, its tree page, or a direct ZIP URL.")
    api = "https://huggingface.co/api/models/%s/revision/%s" % (repo, quote(revision, safe=""))
    with _open(api) as response:
        payload = json.load(response)
    candidates = sorted(item["rfilename"] for item in payload.get("siblings", []) if item.get("rfilename", "").lower().endswith(".zip") and (not prefix or item["rfilename"].startswith(prefix + "/")))
    if not candidates:
        raise ImportProblem("No ZIP archive found in this repository. Paste a direct ZIP link or use the file importer.")
    if len(candidates) > 1:
        preferred = [name for name in candidates if Path(name).stem.casefold() == parts[1].casefold()]
        if len(preferred) != 1:
            links = ["https://huggingface.co/%s/resolve/%s/%s?download=true" % (repo, quote(revision, safe=""), quote(name, safe="/")) for name in candidates[:10]]
            raise ImportProblem("Multiple ZIPs found. Paste the direct URL for the one you want: " + " | ".join(links))
        candidates = preferred
    filename = candidates[0]
    return "https://huggingface.co/%s/resolve/%s/%s?download=true" % (repo, quote(revision, safe=""), quote(filename, safe="/")), _clean_name(Path(filename).name)


def download_huggingface_models(url, root):
    """Download uses 0-70% of workflow, local import the remaining 30%."""
    progress = 0.0
    try:
        yield _status(progress, "Finding the model ZIP on Hugging Face...")
        resolved, name = _resolve_huggingface_zip(url)
        progress = 5.0
        yield _status(progress, "ZIP found: " + name + "; connecting...")
        with tempfile.TemporaryDirectory(prefix="ilaria-download-") as temporary:
            target = Path(temporary) / name
            with _open(resolved) as response, target.open("wb") as output:
                total = int(response.headers.get("Content-Length") or 0)
                received, previous = 0, -1
                if total > MAX_BYTES:
                    raise ImportProblem("The download exceeds the 16 GiB limit.")
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > MAX_BYTES:
                        raise ImportProblem("The download exceeds the 16 GiB limit.")
                    output.write(chunk)
                    progress = min(70.0, 5 + 65 * received / total) if total else 5.0
                    tick = int(progress) if total else received // (4 * CHUNK)
                    if tick != previous:
                        previous = tick
                        detail = ("Downloading %s (%0.1f / %0.1f MiB)" % (name, received / (1024 * 1024), total / (1024 * 1024))) if total else ("Downloading %s: %.1f MiB; total size unknown (download phase)" % (name, received / (1024 * 1024)))
                        yield _status(progress, detail)
                if total and received != total:
                    raise ImportProblem("The download is incomplete. Try again.")
            progress = 70.0
            yield _status(progress, "Download finished; inspecting and installing model files...")
            for local_progress, detail, done in _import_events([str(target)], root):
                progress = 70 + 0.3 * local_progress
                yield _status(progress, detail, done)
    except Exception as error:
        yield _status(progress, "Download failed: " + _friendly(error))
