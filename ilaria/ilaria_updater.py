"""ILARIA RVC release updater. Python 3.9+, standard library, managed files only."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from contextlib import contextmanager
try:
    import ilaria_release_migration
except ImportError:
    payload_path = _ilaria_install_root / 'ilaria_release_payload.py'
    if payload_path.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location('ilaria_release_migration', payload_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ilaria_sys.modules['ilaria_release_migration'] = module

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, unquote
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

CURRENT_VERSION = '1.9.0'
MANIFEST_URL = 'https://huggingface.co/TheStinger/Ilaria_RVC/resolve/main/version.json'
LEGACY_MANAGED_FILES = (
    'ILARIA_RVC.py', 'ILARIA_RVC.bat',
    'ilaria_model_import.py', 'ilaria_model_merge.py', 'ilaria_model_info.py',
    'infer/modules/train/train.py', 'infer/lib/train/epoch_eta.py',
    'infer/lib/train/losses.py', 'infer/lib/train/mel_processing.py',
    'infer/modules/vc/modules.py', 'infer/lib/train/training_performance.py',
    'infer/modules/train/extract_feature_print.py',
    'infer/modules/train/extract/extract_f0_rmvpe.py',
    'assets/ilaria_rvc_icon.png', 'assets/ilaria_rvc_icon_transparent.png',
    'ilaria_updater.py', 'ilaria_launcher.py', 'ilaria_version.json',
)
MANAGED_FILES = tuple('ilaria/' + name if '/' not in name and name.startswith('ilaria_') else name for name in LEGACY_MANAGED_FILES)
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
CHUNK = 1024 * 1024
_HASH = re.compile(r'^[0-9a-f]{64}$')
_ID = re.compile(r'^[0-9a-f]{32}$')


class UpdaterError(ValueError):
    """A human-readable updater failure."""


def version_tuple(value):
    if not isinstance(value, str) or not re.fullmatch(r'v?[0-9]{1,6}\.[0-9]{1,6}\.[0-9]{1,6}', value):
        raise UpdaterError('Versions must use three numbers, for example 1.2.0.')
    result = tuple(int(part) for part in value.lstrip('v').split('.'))
    if any(part > 999999 for part in result):
        raise UpdaterError('The release version is invalid.')
    return result


def _version(value):
    return '.'.join(str(part) for part in version_tuple(value))


def _release_names(files):
    """Accept clean-folder releases and legacy records without arbitrary paths."""
    if not isinstance(files, dict) or len(files) > 100:
        raise UpdaterError('The release must include all supported program files (at most 100 files).')
    if set(MANAGED_FILES).issubset(files):
        base, pattern, version = MANAGED_FILES, r'ilaria/ilaria_[a-z0-9_]{1,60}\.py', 'ilaria/ilaria_version.json'
    elif set(LEGACY_MANAGED_FILES).issubset(files):
        base, pattern, version = LEGACY_MANAGED_FILES, r'ilaria_[a-z0-9_]{1,60}\.py', 'ilaria_version.json'
    else:
        raise UpdaterError('The release is missing required program files. Older updaters need the layout migration first.')
    extras = set(files) - set(base)
    if any(not isinstance(name, str) or not re.fullmatch(pattern, name) for name in extras):
        raise UpdaterError('A release file is outside the supported ILARIA RVC program paths.')
    return tuple(name for name in base if name != version) + tuple(sorted(extras)) + (version,)


def _root(root):
    result = Path(root).resolve()
    if not result.is_dir():
        raise UpdaterError('The ILARIA RVC installation folder does not exist.')
    return result


def _safe_path(root, relative):
    """Do not follow links or Windows directory junctions below the root."""
    relative = str(relative)
    if not relative or '\\' in relative or relative.startswith('/'):
        raise UpdaterError('An update path is invalid.')
    parts = relative.split('/')
    if any(part in ('', '.', '..') or ':' in part for part in parts):
        raise UpdaterError('An update path is invalid.')
    current = root
    for number, part in enumerate(parts):
        current = current / part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or getattr(details, 'st_file_attributes', 0) & 0x400:
            raise UpdaterError('Update paths cannot contain symbolic links or directory junctions.')
        if number < len(parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise UpdaterError('An update folder is occupied by a file.')
    if current.exists() and current.is_dir():
        raise UpdaterError('An update file path is occupied by a folder.')
    return current


def _state(root):
    # A trailing child is checked so the state directory itself must be a directory.
    _safe_path(root, '.ilaria-updates/check')
    result = root / '.ilaria-updates'
    result.mkdir(exist_ok=True)
    return result


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            value.update(block)
    return value.hexdigest()


def _json(path, max_bytes=MAX_MANIFEST_BYTES):
    if path.stat().st_size > max_bytes:
        raise UpdaterError('An updater metadata file is too large.')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.ilaria-json-', dir=str(path.parent))
    try:
        with os.fdopen(handle, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _exclusive_lock(root):
    state = _state(root)
    path = _safe_path(root, '.ilaria-updates/update.lock')
    stream = path.open('a+b')
    locked = False
    try:
        if path.stat().st_size == 0:
            stream.write(b' ')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise UpdaterError('Another updater is already running. Wait for it to finish.')
        locked = True
        stream.seek(0)
        stream.write(('Updater PID: %s\n' % os.getpid()).encode('ascii'))
        stream.truncate()
        stream.flush()
        yield state
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def read_local_version(root):
    root = _root(root)
    path = _safe_path(root, 'ilaria/ilaria_version.json')
    if not path.exists():
        path = _safe_path(root, 'ilaria_version.json')
    if not path.exists():
        return CURRENT_VERSION
    try:
        value = _json(path)
        return _version(value['version'])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise UpdaterError('The installed version file is invalid: %s' % error)


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme.lower() != 'https':
            raise UpdaterError('The update server attempted an insecure redirect.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url):
    request = Request(url, headers={'User-Agent': 'ILARIA-RVC-Updater/' + CURRENT_VERSION,
                                    'Cache-Control': 'no-cache'})
    for attempt in range(3):
        try:
            return build_opener(_HTTPSRedirect()).open(request, timeout=20)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 2:
                raise
        time.sleep(0.5 * (attempt + 1))


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise UpdaterError('The update manifest must be a JSON object.')
    try:
        version = _version(manifest['version'])
        minimum = _version(manifest['minimum_version'])
        url = manifest['download_url']
        checksum = manifest['sha256']
        files = manifest['files']
        changelog = manifest.get('changelog', [])
    except (KeyError, TypeError) as error:
        raise UpdaterError('The update manifest is missing a required field: %s' % error)
    if version_tuple(minimum) > version_tuple(version):
        raise UpdaterError('The update minimum version is invalid.')
    if not isinstance(url, str) or len(url) > 4000:
        raise UpdaterError('The package download URL is invalid.')
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise UpdaterError('The package download URL is invalid.')
    decoded = unquote(parsed.path)
    if (parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or port not in (None, 443)
            or parsed.username or parsed.password
            or not decoded.startswith('/TheStinger/Ilaria_RVC/resolve/')
            or not decoded.lower().endswith('.zip')
            or any(part in ('.', '..') for part in decoded.split('/'))
            or '\\' in decoded or parsed.fragment):
        raise UpdaterError('Update packages must be ZIP files in TheStinger/Ilaria_RVC on Hugging Face.')
    if not isinstance(checksum, str) or not _HASH.fullmatch(checksum.lower()):
        raise UpdaterError('The package SHA-256 checksum is invalid.')
    names = _release_names(files)
    if any(not isinstance(value, str) or not _HASH.fullmatch(value.lower()) for value in files.values()):
        raise UpdaterError('A program file checksum is invalid.')
    if (not isinstance(changelog, list) or len(changelog) > 100
            or any(not isinstance(line, str) or len(line) > 2000 for line in changelog)):
        raise UpdaterError('The release changelog is invalid.')
    return {'version': version, 'minimum_version': minimum, 'download_url': url,
            'sha256': checksum.lower(), 'files': {name: files[name].lower() for name in names},
            'changelog': list(changelog)}


def check_update(root):
    installed = read_local_version(root)
    try:
        with _open(MANIFEST_URL) as response:
            raw = response.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise UpdaterError('The published update manifest is too large.')
        manifest = validate_manifest(json.loads(raw.decode('utf-8-sig')))
    except HTTPError as error:
        if error.code == 404:
            return {'available': False, 'installed': installed, 'version': installed,
                    'changelog': [], 'manifest': None,
                    'message': 'Check complete: no updates have been published yet.\nYou can keep using your installed version of Ilaria RVC.'}
        raise UpdaterError('Could not check updates: Hugging Face returned HTTP %s.' % error.code)
    except (URLError, TimeoutError, ConnectionError, OSError) as error:
        raise UpdaterError('Could not connect to Hugging Face. Check your internet connection and try again: %s' % error)
    except (UnicodeError, ValueError) as error:
        if isinstance(error, UpdaterError):
            raise
        raise UpdaterError('Could not read the published update manifest: %s' % error)
    available = version_tuple(manifest['version']) > version_tuple(installed)
    compatible = version_tuple(installed) >= version_tuple(manifest['minimum_version'])
    message = ('Version %s is available.' % manifest['version'] if available
               else 'Check complete: no newer version is available. You are using version %s.' % installed)
    if available and not compatible:
        message = 'This update requires ILARIA RVC %s or newer. Install a compatible base first.' % manifest['minimum_version']
    return {'available': available and compatible, 'installed': installed,
            'version': manifest['version'], 'changelog': manifest['changelog'],
            'manifest': manifest, 'message': message}


def _pending_path(root):
    return _safe_path(root, '.ilaria-updates/pending.json')


def pending_update(root):
    """Read validated pending metadata without installing anything."""
    root = _root(root)
    path = _pending_path(root)
    if not path.exists():
        return None
    try:
        pending = _json(path)
        if not isinstance(pending, dict) or not _ID.fullmatch(str(pending.get('id', ''))):
            raise UpdaterError('The pending update identifier is invalid.')
        manifest = validate_manifest(pending['manifest'])
        return {'id': pending['id'], 'manifest': manifest}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UpdaterError('The pending update is invalid: %s' % error)


def _verify_staged(root, transaction, manifest):
    files = manifest['files']
    for name in _release_names(files):
        target = _safe_path(root, '.ilaria-updates/%s/staged/%s' % (transaction, name))
        if not target.is_file() or _digest(target) != files[name]:
            raise UpdaterError('A staged update file failed verification: ' + name)
        if name.endswith('.py'):
            try:
                compile(target.read_bytes(), name, 'exec')
            except (SyntaxError, UnicodeError, ValueError) as error:
                raise UpdaterError('A Python file in the update has invalid syntax or encoding: %s: %s' % (name, error))
    version_name = 'ilaria/ilaria_version.json' if 'ilaria/ilaria_version.json' in files else 'ilaria_version.json'
    version_file = _safe_path(root, '.ilaria-updates/%s/staged/%s' % (transaction, version_name))
    version = _json(version_file)
    if not isinstance(version, dict) or _version(version.get('version')) != manifest['version']:
        raise UpdaterError('The package version does not match the release manifest.')


def _validate_archive(archive, names):
    entries = archive.infolist()
    if len(entries) != len(names):
        raise UpdaterError('The ZIP must contain exactly the supported program files, with no outer folder.')
    seen = set()
    total = 0
    for entry in entries:
        name = entry.filename
        if (name not in names or name.casefold() in seen or entry.is_dir()
                or entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16)
                or '\\' in name or '\x00' in name):
            raise UpdaterError('The ZIP contains an unexpected, duplicate or unsafe entry: ' + name[:150])
        mode = entry.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise UpdaterError('The ZIP contains an unsupported file type.')
        if entry.file_size > MAX_ARCHIVE_BYTES or entry.file_size < 0:
            raise UpdaterError('An extracted program file is too large.')
        total += entry.file_size
        if total > MAX_EXTRACTED_BYTES:
            raise UpdaterError('The extracted update is too large.')
        seen.add(name.casefold())
    return entries, total


def prepare_update(root, manifest):
    """Download and verify; yield progress. Does not replace installed files."""
    root = _root(root)
    manifest = validate_manifest(manifest)
    installed = read_local_version(root)
    if version_tuple(installed) < version_tuple(manifest['minimum_version']):
        raise UpdaterError('Your installed version is too old for this update.')
    if version_tuple(manifest['version']) <= version_tuple(installed):
        raise UpdaterError('This release is not newer than the installed version.')
    with _exclusive_lock(root) as state:
        if _safe_path(root, '.ilaria-updates/journal.json').exists():
            raise UpdaterError('An interrupted update needs recovery. Restart ILARIA_RVC.bat first.')
        transaction = uuid.uuid4().hex
        folder = state / transaction
        folder.mkdir()
        package = _safe_path(root, '.ilaria-updates/%s/package.zip' % transaction)
        yield '0.0% | Connecting to Hugging Face...'
        digest = hashlib.sha256()
        received = 0
        last_emit = 0.0
        try:
            with _open(manifest['download_url']) as response, package.open('xb') as output:
                try:
                    size = int(response.headers.get('Content-Length', '0'))
                except (TypeError, ValueError):
                    size = 0
                if size > MAX_ARCHIVE_BYTES:
                    raise UpdaterError('The update package is too large.')
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    received += len(block)
                    if received > MAX_ARCHIVE_BYTES:
                        raise UpdaterError('The update package exceeds the download size limit.')
                    digest.update(block)
                    output.write(block)
                    now = time.monotonic()
                    if now - last_emit >= 0.2:
                        percent = min(64.9, 5 + 60 * received / size) if size > 0 else 5.0
                        yield '%.1f%% | Downloading update: %.1f MiB%s' % (percent, received / 1024**2,
                            ' / %.1f MiB' % (size / 1024**2) if size > 0 else ' (total size not supplied)')
                        last_emit = now
                output.flush()
                os.fsync(output.fileno())
            if not received or digest.hexdigest() != manifest['sha256']:
                raise UpdaterError('The downloaded ZIP checksum does not match. No installed files were changed.')
            yield '65.0% | Download verified. Checking ZIP contents...'
            with zipfile.ZipFile(package) as archive:
                entries, total = _validate_archive(archive, _release_names(manifest['files']))
                extracted = 0
                for entry in entries:
                    target = _safe_path(root, '.ilaria-updates/%s/staged/%s' % (transaction, entry.filename))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    check = hashlib.sha256()
                    with archive.open(entry) as source, target.open('xb') as output:
                        written = 0
                        while True:
                            block = source.read(CHUNK)
                            if not block:
                                break
                            written += len(block)
                            if written > entry.file_size:
                                raise UpdaterError('A ZIP entry exceeds its declared size.')
                            output.write(block)
                            check.update(block)
                        if written != entry.file_size or check.hexdigest() != manifest['files'][entry.filename]:
                            raise UpdaterError('A program file failed checksum verification: ' + entry.filename)
                    extracted += written
                    yield '%.1f%% | Verified %s' % (70 + 25 * extracted / max(total, 1), entry.filename)
            _verify_staged(root, transaction, manifest)
            _atomic_json(_pending_path(root), {'id': transaction, 'manifest': manifest})
            yield '100.0%% | Version %s is downloaded and verified. Ready to install and restart.' % manifest['version']
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise UpdaterError('Could not prepare the update: %s' % error)


def _atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.ilaria-install-', dir=str(target.parent))
    try:
        with os.fdopen(handle, 'wb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output, CHUNK)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _journal(root, relative='.ilaria-updates/journal.json'):
    path = _safe_path(root, relative)
    if not path.exists():
        return None
    try:
        data = _json(path)
        if not isinstance(data, dict) or not _ID.fullmatch(str(data.get('id', ''))):
            raise UpdaterError('The recovery journal identifier is invalid.')
        if data.get('phase') not in ('applying', 'committed'):
            raise UpdaterError('The recovery journal phase is invalid.')
        old = data.get('original')
        new = data.get('replacement')
        _release_names(old)
        _release_names(new)
        if set(old) != set(new):
            raise UpdaterError('The recovery journal file list is invalid.')
        if any(v is not None and (not isinstance(v, str) or not _HASH.fullmatch(v)) for v in old.values()):
            raise UpdaterError('The recovery backup checksums are invalid.')
        if any(not isinstance(v, str) or not _HASH.fullmatch(v) for v in new.values()):
            raise UpdaterError('The recovery replacement checksums are invalid.')
        return data
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UpdaterError('Cannot recover the interrupted update: %s' % error)


def _clear_transaction(root, journal):
    path = _pending_path(root)
    if path.exists():
        try:
            pending = _json(path)
        except (ValueError, OSError):
            pending = None
        if isinstance(pending, dict) and pending.get('id') == journal['id']:
            path.unlink()
    _safe_path(root, '.ilaria-updates/journal.json').unlink(missing_ok=True)


def _clear_matching_last_update(root, journal):
    path = _safe_path(root, '.ilaria-updates/last-update.json')
    if path.exists():
        try:
            record = _json(path)
        except (ValueError, OSError):
            record = None
        if isinstance(record, dict) and record.get('id') == journal['id']:
            path.unlink()


def _rollback(root, journal):
    transaction = journal['id']
    # Verify every old file before changing any installed file.
    for name, checksum in journal['original'].items():
        _safe_path(root, name)
        if checksum is not None:
            backup = _safe_path(root, '.ilaria-updates/backups/%s/%s' % (transaction, name))
            if not backup.is_file() or _digest(backup) != checksum:
                raise UpdaterError('A recovery backup is missing or damaged: ' + name)
    nested_payload = journal['replacement'].get('ilaria/ilaria_release_payload.py')
    if nested_payload and _safe_path(root, '.ilaria-updates/model-storage-1.8.json').is_file():
        from ilaria_model_storage_migration import restore_models
        restore_models(root, nested_payload)
    for payload_name in ('ilaria_release_payload.py', 'ilaria/ilaria_release_payload.py'):
        if payload_name in journal['replacement']:
            from ilaria_release_migration import rollback_layout
            rollback_layout(root, journal['replacement'][payload_name])
    order = _release_names(journal['original'])
    for name in order:
        destination = _safe_path(root, name)
        checksum = journal['original'][name]
        if checksum is None:
            if destination.exists():
                if _digest(destination) != journal['replacement'][name]:
                    raise UpdaterError('Recovery found an unexpected new file and kept it: ' + name)
                destination.unlink()
        else:
            _atomic_copy(_safe_path(root, '.ilaria-updates/backups/%s/%s' % (transaction, name)), destination)
            if _digest(destination) != checksum:
                raise UpdaterError('Recovery verification failed: ' + name)



def _recover_locked(root):
    journal = _journal(root)
    if not journal:
        return
    if journal['phase'] == 'committed':
        _clear_transaction(root, journal)
        yield '100.0% | Previous update completed successfully. Startup can continue.'
        return
    yield '0.0% | Recovering an interrupted update from its backup...'
    _rollback(root, journal)
    _safe_path(root, '.ilaria-updates/journal.json').unlink(missing_ok=True)
    _clear_matching_last_update(root, journal)
    yield '100.0% | Previous files restored. The update can be retried.'


def recover_interrupted(root):
    root = _root(root)
    with _exclusive_lock(root):
        yield from _recover_locked(root)


def rollback_latest(root):
    """Restore the last committed release only while its installed files are intact."""
    root = _root(root)
    with _exclusive_lock(root):
        yield from _recover_locked(root)
        record = _journal(root, '.ilaria-updates/last-update.json')
        if record is None:
            raise UpdaterError('There is no completed update available to restore.')
        if record['phase'] != 'committed':
            raise UpdaterError('The last update record is not a completed release.')
        yield '0.0% | Checking whether the last update can be restored...'
        for name, checksum in record['replacement'].items():
            current = _safe_path(root, name)
            if not current.is_file() or _digest(current) != checksum:
                from ilaria_release_migration import installed_file_matches
                if not installed_file_matches(root, name, checksum):
                    raise UpdaterError('Automatic restore refused because an installed program file changed: ' + name)
        journal = dict(record, phase='applying')
        journal_path = _safe_path(root, '.ilaria-updates/journal.json')
        _atomic_json(journal_path, journal)
        yield '25.0% | Restoring the previous program files...'
        try:
            _rollback(root, journal)
        except Exception as error:
            raise UpdaterError('Restore did not finish: %s. Restart ILARIA_RVC.bat to recover.' % error)
        _clear_matching_last_update(root, journal)
        journal_path.unlink(missing_ok=True)
        yield '100.0% | Previous version restored. The backup is retained.'


def apply_pending(root):
    """Install a verified pending release. The UI must already be stopped."""
    root = _root(root)
    with _exclusive_lock(root):
        yield from _recover_locked(root)
        pending = pending_update(root)
        if pending is None:
            yield 'No pending update.'
            return
        transaction, manifest = pending['id'], pending['manifest']
        installed = read_local_version(root)
        if version_tuple(installed) < version_tuple(manifest['minimum_version']) or version_tuple(manifest['version']) <= version_tuple(installed):
            raise UpdaterError('The pending release is not compatible with the installed version.')
        yield '0.0% | Rechecking downloaded program files...'
        _verify_staged(root, transaction, manifest)
        original = {}
        names = _release_names(manifest['files'])
        for number, name in enumerate(names, 1):
            target = _safe_path(root, name)
            original[name] = _digest(target) if target.is_file() else None
            if target.exists():
                backup = _safe_path(root, '.ilaria-updates/backups/%s/%s' % (transaction, name))
                _atomic_copy(target, backup)
                if _digest(backup) != original[name]:
                    raise UpdaterError('Could not verify a backup: ' + name)
            yield '%.1f%% | Backing up %s' % (20 * number / len(names), name)
        journal = {'id': transaction, 'phase': 'applying', 'original': original,
                   'replacement': manifest['files']}
        journal_path = _safe_path(root, '.ilaria-updates/journal.json')
        _atomic_json(journal_path, journal)
        try:
            order = names
            for number, name in enumerate(order, 1):
                source = _safe_path(root, '.ilaria-updates/%s/staged/%s' % (transaction, name))
                target = _safe_path(root, name)
                # Detect changed staging before each replacement as well.
                if _digest(source) != manifest['files'][name]:
                    raise UpdaterError('A staged file changed during installation: ' + name)
                _atomic_copy(source, target)
                if _digest(target) != manifest['files'][name]:
                    raise UpdaterError('Installed file verification failed: ' + name)
                yield '%.1f%% | Installing %s' % (20 + 75 * number / len(order), name)
            for name in names:
                if _digest(_safe_path(root, name)) != manifest['files'][name]:
                    raise UpdaterError('Final release verification failed: ' + name)
            journal['phase'] = 'committed'
            _atomic_json(journal_path, journal)
            _atomic_json(_safe_path(root, '.ilaria-updates/last-update.json'), journal)
        except Exception as error:
            try:
                _rollback(root, journal)
                journal_path.unlink(missing_ok=True)
            except Exception as rollback_error:
                raise UpdaterError('Update failed: %s. Recovery also needs attention: %s. Keep .ilaria-updates/backups.' % (error, rollback_error))
            raise UpdaterError('Update failed and the previous files were restored: %s' % error)
        _clear_transaction(root, journal)
        yield '100.0%% | Updated to %s. Backup kept in .ilaria-updates/backups/%s.' % (manifest['version'], transaction)


def main(argv=None):
    parser = argparse.ArgumentParser(description='ILARIA RVC updater')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'ilaria' else Path(__file__).resolve().parent))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--apply-pending', action='store_true')
    mode.add_argument('--recover', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--rollback-latest', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = check_update(args.root)
            print(result['message'], flush=True)
        else:
            iterator = (apply_pending(args.root) if args.apply_pending else
                        rollback_latest(args.root) if args.rollback_latest else recover_interrupted(args.root))
            for line in iterator:
                print(line, flush=True)
        return 0
    except Exception as error:
        print('Update error: %s' % error, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
