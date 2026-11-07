"""Download public Hugging Face G/D pretrained pairs without loading weights."""
import sys
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / 'ilaria'), str(_ROOT)]

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote, urlparse
from zipfile import ZipFile, BadZipFile

from ilaria_model_import import _open, _clean_name, _friendly, ImportProblem
from ilaria_pretrain_selector import _pair_role

CHUNK = 16 * 1024**2
MAX_BYTES = 16 * 1024**3
MAX_JSON = 16 * 1024**2
_INSTALL_LOCK = threading.RLock()


def _relative(name):
    name = str(name).replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or not path.parts:
        raise ImportProblem('An archive or repository path is unsafe.')
    for part in path.parts:
        _clean_name(part)
    return path


def _url_info(url):
    parsed = urlparse(str(url or '').strip())
    if (parsed.scheme != 'https' or parsed.hostname != 'huggingface.co'
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ImportProblem('Paste an https://huggingface.co public model repository, ZIP or .pth link.')
    parts = [unquote(part) for part in parsed.path.strip('/').split('/')]
    if len(parts) < 2 or not all(parts[:2]) or parts[0] in ('spaces', 'datasets', 'api'):
        raise ImportProblem('Use a Hugging Face model repository link.')
    repo = '/'.join(parts[:2])
    revision, prefix, filename = 'main', '', None
    if len(parts) > 2:
        if parts[2] in ('resolve', 'blob') and len(parts) >= 5:
            revision, filename = parts[3], '/'.join(parts[4:])
            _relative(filename)
            if not filename.lower().endswith(('.zip', '.pth')):
                raise ImportProblem('A direct download link must end in .zip or .pth.')
        elif parts[2] == 'tree' and len(parts) >= 4:
            revision, prefix = parts[3], '/'.join(parts[4:]).rstrip('/')
            if prefix:
                _relative(prefix)
        else:
            raise ImportProblem('Use a repository page, a tree page, or a direct ZIP/.pth download link.')
    return repo, revision, prefix, filename


def _direct(repo, revision, filename):
    return 'https://huggingface.co/%s/resolve/%s/%s?download=true' % (
        repo, quote(revision, safe=''), quote(filename, safe='/'))


def _pairs(names):
    """Only pair recognizable G/D names in the same original folder."""
    groups = {}
    for name in names:
        path = _relative(name)
        if path.suffix.lower() != '.pth':
            continue
        role = _pair_role(path.stem)
        if role is None:
            continue
        letter, f0, key = role
        group = (str(path.parent).casefold(), f0, key)
        slots = groups.setdefault(group, {})
        if letter in slots:
            raise ImportProblem('Several %s files match the same pretrained pair. Use an unambiguous ZIP or direct pair.' % letter)
        slots[letter] = str(path)
    return [(slots['G'], slots['D']) for slots in groups.values() if 'G' in slots and 'D' in slots]


def _repo_names(repo, revision, prefix=''):
    api = 'https://huggingface.co/api/models/%s/revision/%s' % (repo, quote(revision, safe=''))
    with _open(api) as response:
        raw = response.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ImportProblem('Repository metadata is too large. Use direct ZIP or G/D links.')
    payload = json.loads(raw)
    names = [item.get('rfilename', '') for item in payload.get('siblings', [])]
    return sorted(name for name in names if name and (not prefix or name.startswith(prefix + '/')))


def _resolve(url, discriminator_url=''):
    repo, revision, prefix, filename = _url_info(url)
    if discriminator_url:
        other_repo, other_revision, _, other_file = _url_info(discriminator_url)
        if not filename or not other_file or not filename.lower().endswith('.pth') or not other_file.lower().endswith('.pth'):
            raise ImportProblem('When using two links, paste a direct .pth link for each member of the G/D pair.')
        left = _pair_role(PurePosixPath(filename).stem)
        right = _pair_role(PurePosixPath(other_file).stem)
        if not left or not right or left[0] == right[0] or left[1:] != right[1:]:
            raise ImportProblem('Those filenames do not form one matching generator/discriminator pair.')
        items = [(_direct(repo, revision, filename), PurePosixPath(filename).name),
                 (_direct(other_repo, other_revision, other_file), PurePosixPath(other_file).name)]
        package = '_'.join(part for part in (repo.split('/')[-1], PurePosixPath(filename).parent.name, left[2]) if part)
        return 'pair', items, package
    if filename and filename.lower().endswith('.zip'):
        return 'zip', [(_direct(repo, revision, filename), PurePosixPath(filename).name)], PurePosixPath(filename).stem
    names = _repo_names(repo, revision, prefix)
    pairs = _pairs(names)
    if filename:
        pairs = [pair for pair in pairs if filename in pair]
        if not pairs:
            raise ImportProblem('No matching G/D partner was found beside this .pth. Paste both direct links or a ZIP containing the complete pair.')
    if len(pairs) == 1:
        pair = pairs[0]
        key = _pair_role(PurePosixPath(pair[0]).stem)[2]
        package = '_'.join(part for part in (repo.split('/')[-1], PurePosixPath(pair[0]).parent.name, key) if part)
        return 'pair', [(_direct(repo, revision, name), PurePosixPath(name).name) for name in pair], package
    if len(pairs) > 1:
        links = ['G: %s | D: %s' % (_direct(repo, revision, pair[0]), _direct(repo, revision, pair[1])) for pair in pairs[:8]]
        raise ImportProblem('Several pretrained pairs were found. Choose one pair by pasting its direct G link (and optionally D link):\n' + '\n'.join(links))
    zips = [name for name in names if name.lower().endswith('.zip')]
    if len(zips) == 1:
        return 'zip', [(_direct(repo, revision, zips[0]), PurePosixPath(zips[0]).name)], PurePosixPath(zips[0]).stem
    if len(zips) > 1:
        raise ImportProblem('Several ZIPs were found. Paste the direct link for the pretrained ZIP you want:\n' + '\n'.join(_direct(repo, revision, name) for name in zips[:8]))
    raise ImportProblem('No complete G/D pair or ZIP was found. Voice inference .pth files are not pretrained pairs.')


def _download(url, path, start, span, budget):
    received, previous = 0, -1
    with _open(url) as response, path.open('xb') as stream:
        total = int(response.headers.get('Content-Length') or 0)
        if total > budget:
            raise ImportProblem('The download exceeds the 16 GiB limit.')
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            received += len(chunk)
            if received > budget:
                raise ImportProblem('The download exceeds the 16 GiB limit.')
            stream.write(chunk)
            percent = start + span * received / total if total else start
            tick = int(percent) if total else received // CHUNK
            if tick != previous:
                previous = tick
                yield min(start + span, percent), 'Downloading %s: %.1f MiB%s' % (
                    path.name, received / 1024**2, ' / %.1f MiB' % (total / 1024**2) if total else ' (total size unknown)')
        if not received:
            raise ImportProblem('The download is empty.')
        if total and received != total:
            raise ImportProblem('The download is incomplete. Retry it.')


def _download_files(downloads, stage):
    """Transfer independently of Gradio iteration; G and D share one byte limit."""
    names = [_clean_name(name) for _, name in downloads]
    if len(set(name.casefold() for name in names)) != len(names):
        raise ImportProblem('The selected files have the same filename.')
    lock, stop = threading.Lock(), threading.Event()
    received, totals = [0] * len(downloads), [0] * len(downloads)
    started = time.monotonic()

    def transfer(number):
        url, _ = downloads[number]
        with _open(url) as response, (stage / names[number]).open('xb') as output:
            total = int(response.headers.get('Content-Length') or 0)
            with lock:
                totals[number] = total
                if sum(totals) > MAX_BYTES:
                    raise ImportProblem('The download exceeds the 16 GiB limit.')
            while not stop.is_set():
                block = response.read(CHUNK)
                if not block:
                    break
                with lock:
                    received[number] += len(block)
                    if sum(received) > MAX_BYTES:
                        raise ImportProblem('The download exceeds the 16 GiB limit.')
                output.write(block)
            if stop.is_set():
                raise ImportProblem('Download cancelled.')
            if not received[number]:
                raise ImportProblem('The download is empty.')
            if total and received[number] != total:
                raise ImportProblem('The download is incomplete. Retry it.')

    executor = ThreadPoolExecutor(max_workers=min(2, len(downloads)))
    futures = [executor.submit(transfer, number) for number in range(len(downloads))]
    try:
        while True:
            for future in futures:
                if future.done():
                    future.result()
            with lock:
                copied, total = sum(received), sum(totals)
                known = all(totals)
            elapsed = max(.001, time.monotonic() - started)
            progress = min(65.0, 5 + 60 * copied / total) if known else 5.0
            detail = 'Downloading G + D' if len(downloads) == 2 else 'Downloading pretrained ZIP'
            if all(future.done() for future in futures):
                yield 65.0, '%s complete: %.1f MiB | %.1f MiB/s | 16 MiB blocks' % (detail, copied / 1024**2, copied / elapsed / 1024**2)
                break
            yield progress, '%s: %.1f MiB%s | %.1f MiB/s | 16 MiB blocks' % (
                detail, copied / 1024**2, ' / %.1f MiB' % (total / 1024**2) if known else '', copied / elapsed / 1024**2)
            stop.wait(.5)
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True)


def _zip_items(path):
    files, seen = {}, set()
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 20000:
            raise ImportProblem('The ZIP contains too many entries.')
        for member in members:
            relative = _relative(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ImportProblem('ZIP symbolic links are not supported.')
            if member.is_dir():
                continue
            key = str(relative).casefold()
            if key in seen:
                raise ImportProblem('The ZIP contains duplicate filenames.')
            seen.add(key)
            if member.flag_bits & 1:
                raise ImportProblem('Password-protected ZIP files are not supported.')
            if relative.suffix.lower() == '.pth':
                files[str(relative)] = (member.filename, member.file_size)
    pairs = _pairs(files)
    selected = list(dict.fromkeys(name for pair in pairs for name in pair))
    if not selected:
        raise ImportProblem('The ZIP has no complete G/D pretrained pair. It may contain an inference voice model instead.')
    if sum(files[name][1] for name in selected) > MAX_BYTES:
        raise ImportProblem('Extracted pretrained files exceed the 16 GiB limit.')
    if any(files[name][1] == 0 for name in selected):
        raise ImportProblem('A pretrained file is empty.')
    return [(name, files[name][0], files[name][1]) for name in selected], len(pairs)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.digest()


def _folder_name(value):
    value = re.sub(r'[^\w .()-]+', '_', str(value), flags=re.UNICODE).strip(' ._')[:100]
    return _clean_name(value or 'pretrained')


def _inside(root, path):
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ImportProblem('The pretrained destination leaves the application folder.')
    return resolved


def _fixed_folder(root, relative):
    expected = root / relative
    resolved = _inside(root, expected)
    if resolved != expected:
        raise ImportProblem('The pretrained download folders must not be symbolic links or junctions.')
    return resolved


def download_pretrained(url, root, discriminator_url=''):
    """Yield percentage logs. Install only complete pairs, atomically by folder."""
    progress = 0.0
    try:
        root = Path(root).resolve()
        yield '0.0% | Finding a pretrained G/D pair on Hugging Face...'
        kind, downloads, package = _resolve(url, discriminator_url)
        destination_base = _fixed_folder(root, 'assets/pretrained_custom')
        destination_base.mkdir(parents=True, exist_ok=True)
        stage_base = _fixed_folder(root, 'TEMP/pretrained-downloads')
        stage_base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='download-', dir=str(stage_base)) as temporary:
            stage = Path(temporary)
            for progress, detail in _download_files(downloads, stage):
                yield '%.1f%% | %s' % (progress, detail)
            ready = stage / 'ready'
            ready.mkdir()
            if kind == 'zip':
                archive_path = stage / downloads[0][1]
                items, count = _zip_items(archive_path)
                total = sum(size for _, _, size in items)
                if shutil.disk_usage(stage).free < total + 64 * 1024**2:
                    raise ImportProblem('Not enough free disk space to extract the pretrained pair.')
                copied, last_report = 0, 0.0
                with ZipFile(archive_path) as archive:
                    for relative, internal, expected in items:
                        target = ready / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        written = 0
                        with archive.open(internal) as source, target.open('xb') as output:
                            while True:
                                block = source.read(CHUNK)
                                if not block:
                                    break
                                written += len(block); copied += len(block)
                                if written > expected or copied > MAX_BYTES:
                                    raise ImportProblem('The ZIP exceeds its declared file sizes.')
                                output.write(block)
                                progress = 65 + 25 * copied / total
                                if time.monotonic() - last_report >= .2:
                                    yield '%.1f%% | Extracting %s...' % (progress, Path(relative).name)
                                    last_report = time.monotonic()
                        if written != expected:
                            raise ImportProblem('A ZIP entry is incomplete.')
            else:
                count = 1
                for _, name in downloads:
                    os.rename(str(stage / name), str(ready / name))
            progress = 92.0
            yield '92.0% | Checking paired files and destination conflicts...'
            destination = _inside(root, destination_base / _folder_name(package))
            if destination != destination_base / _folder_name(package):
                raise ImportProblem('The destination package must not be a symbolic link or junction.')
            with _INSTALL_LOCK:
                if destination.exists():
                    if not destination.is_dir():
                        raise ImportProblem('A file already uses the destination folder name.')
                    for source in ready.rglob('*.pth'):
                        target = destination / source.relative_to(ready)
                        _inside(root, target)
                        if not target.is_file() or _hash(source) != _hash(target):
                            raise ImportProblem('Different or incomplete pretrained files already exist in %s. They were kept. Move that folder aside before downloading a different package with the same name.' % destination.relative_to(root).as_posix())
                    yield '100.0% | This complete pretrained pair is already installed; identical files were kept. In Training, click Refresh pretrained list and select the pair.'
                    return
                # One same-volume folder rename exposes both G/D together.
                os.rename(str(ready), str(destination))
            yield '100.0%% | Installed %s complete pretrained pair(s) in %s. In Training, click Refresh pretrained list and select the pair. File contents and architecture are checked by the trainer, not inferred from filenames.' % (count, destination.relative_to(root).as_posix())
    except Exception as error:
        detail = _friendly(error)
        if 'Alternatively, download the ZIP in your browser' in detail:
            detail = detail.split('Alternatively, download the ZIP in your browser', 1)[0] + 'You can also download the G/D files in your browser and place both in assets/pretrained_custom/<name>.'
        yield '%.1f%% | Pretrained download failed: %s' % (min(99.9, progress), detail)
