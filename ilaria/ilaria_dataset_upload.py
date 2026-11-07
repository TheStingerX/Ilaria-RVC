"""Import uploaded audio directly into datasets, keeping original sources."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path, PurePosixPath
import os
import re
import shutil
import stat
import threading
import time
import uuid
import zipfile

IMPORT_LOCK = threading.Lock()
AUDIO = {'.wav', '.wave', '.flac', '.mp3', '.ogg', '.oga', '.opus', '.m4a', '.m4b', '.aac', '.wma', '.aif', '.aiff', '.aifc', '.webm', '.mp4', '.mkv', '.wv', '.amr', '.mpga'}
MAX_BYTES = 16 * 1024**3
MAX_FILES = 5000


def list_datasets(root):
    folder = Path(root).resolve() / 'datasets'
    if not folder.is_dir():
        return []
    return sorted([p.name for p in folder.iterdir() if not p.name.startswith('.')
                   and (p.is_dir() or p.suffix.lower() in AUDIO)], key=str.casefold)


def remove_dataset(root, selected):
    """Delete only the explicit selected direct child of datasets."""
    if not selected or Path(str(selected)).name != selected or selected in ('.', '..'):
        raise ValueError('Select one dataset to remove.')
    folder = (Path(root).resolve() / 'datasets').resolve()
    target = folder / selected
    if target.resolve().parent != folder or target.is_symlink() or getattr(target.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('Linked or external dataset paths cannot be removed.')
    if selected not in list_datasets(root):
        raise ValueError('The selected dataset was not found.')
    with IMPORT_LOCK:
        if target.is_dir():
            for parent, dirs, files in os.walk(target, followlinks=False):
                for name in dirs + files:
                    child = Path(parent) / name
                    if child.is_symlink() or getattr(child.lstat(), 'st_file_attributes', 0) & 0x400:
                        raise ValueError('The dataset contains a linked path; remove that link manually first.')
            # Resolved absolute target is verified above to stay under datasets.
            shutil.rmtree(target)
        else:
            target.unlink()
    return 'Removed dataset: ' + selected


def import_dataset(files, root, model_name=''):
    """Yield (progress text, installed path or None); commit only complete imports."""
    if not files:
        yield 'Drop dataset audio files or a ZIP first.', None
        return
    if not isinstance(files, (list, tuple)):
        files = [files]
    with IMPORT_LOCK:
        folder = Path(root).resolve() / 'datasets'
        stage = None
        committed = []
        try:
            folder.mkdir(exist_ok=True)
            stage = folder / ('.upload-' + uuid.uuid4().hex)
            stage.mkdir()
            entries, archives, total = [], [], 0
            try:
                for upload in files:
                    source = Path(getattr(upload, 'name', upload))
                    original = str(getattr(upload, 'orig_name', '') or source.name).replace('\\', '/')
                    if source.suffix.lower() == '.zip' or Path(original).suffix.lower() == '.zip':
                        archive = zipfile.ZipFile(source)
                        archives.append(archive)
                        for info in archive.infolist():
                            name = info.filename.replace('\\', '/')
                            path = PurePosixPath(name)
                            if path.is_absolute() or '..' in path.parts or ':' in name or stat.S_ISLNK(info.external_attr >> 16):
                                raise ValueError('The ZIP contains an unsafe path or link.')
                            if info.is_dir() or path.suffix.lower() not in AUDIO:
                                continue
                            if info.flag_bits & 1:
                                raise ValueError('Password-protected ZIPs are not supported.')
                            entries.append((path.name, info.file_size, archive, info))
                            total += info.file_size
                    elif Path(original).suffix.lower() in AUDIO:
                        entries.append((PurePosixPath(original).name, source.stat().st_size, None, source))
                        total += source.stat().st_size
                    else:
                        raise ValueError('Drop audio files or a ZIP; unsupported file: ' + PurePosixPath(original).name)
                    if len(entries) > MAX_FILES or total > MAX_BYTES:
                        raise ValueError('This import exceeds 5,000 audio files or 16 GiB. Split the dataset into smaller imports.')
                if not entries or total <= 0:
                    raise ValueError('No nonempty supported audio files were found.')
                if shutil.disk_usage(folder).free < total + 64 * 1024**2:
                    raise ValueError('Not enough free space to import this dataset.')
                copied, used = 0, {p.name.casefold() for p in folder.iterdir()}
                last_update = time.monotonic()
                yield '0.0%% | Importing %s audio file(s)...' % len(entries), None
                for name, size, archive, source in entries:
                    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).rstrip('. ')
                    stem, suffix = Path(name).stem, Path(name).suffix
                    name = (stem[:80] or 'audio') + suffix
                    if re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', Path(name).stem):
                        name = 'audio_' + name
                    candidate, number = name, 2
                    while candidate.casefold() in used:
                        candidate = '%s_%s%s' % (Path(name).stem, number, suffix)
                        number += 1
                    used.add(candidate.casefold())
                    reader = archive.open(source) if archive else source.open('rb')
                    actual = 0
                    with reader, (stage / candidate).open('xb') as output:
                        while True:
                            block = reader.read(4 * 1024**2)
                            if not block:
                                break
                            actual += len(block)
                            if actual > size or copied + actual > MAX_BYTES:
                                raise ValueError('Audio size changed while importing. Retry with the original files.')
                            output.write(block)
                            if time.monotonic() - last_update >= 0.25:
                                yield '%.1f%% | Importing %s' % (min(99.9, (copied + actual) / total * 100), candidate), None
                                last_update = time.monotonic()
                        if actual != size:
                            raise ValueError('An audio file is incomplete. Retry the import.')
                    copied += actual
                    yield '%.1f%% | Imported %s' % (min(99.9, copied / total * 100), candidate), None
            finally:
                for archive in archives:
                    archive.close()
            for source in stage.iterdir():
                target = folder / source.name
                if os.name == "nt":
                    source.rename(target)  # Windows refuses to overwrite an existing target.
                else:
                    os.link(source, target)  # Exclusive publication on other platforms too.
                    source.unlink()
                committed.append(target)
            stage.rmdir()
            stage = None
            yield '100.0%% | Dataset import done! %s audio file(s). Ready for Process Data.' % len(entries), str(folder) + os.sep
        except Exception as error:
            for target in committed:
                if target.parent.resolve() == folder.resolve():
                    target.unlink(missing_ok=True)
            yield 'Dataset import failed: ' + str(error), None
        finally:
            if stage and stage.exists() and stage.resolve().parent == folder.resolve() and not stage.is_symlink():
                shutil.rmtree(stage)


def validate_dataset_path(dataset_path, root):
    """Require source audio directly in the folder before starting preprocessing."""
    value = str(dataset_path or "").strip().strip('"')
    if not value:
        raise ValueError("No dataset found. Upload audio first.")
    if IMPORT_LOCK.locked():
        raise ValueError("Dataset import in progress. Please wait.")
    folder = Path(os.path.expandvars(value)).expanduser()
    if not folder.is_absolute():
        folder = Path(root) / folder
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError("No dataset found. Upload audio first.")
    audio = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO and p.stat().st_size > 0]
    if not audio:
        raise ValueError("No dataset found. Upload audio first.")
    return str(folder)
