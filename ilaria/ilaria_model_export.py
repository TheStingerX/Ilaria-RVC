"""Stream a shareable ZIP containing only an installed voice and optional index."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path
import os
import shutil
import time
import uuid
import zipfile


from ilaria_model_paths import voice_files, voice_path


def export_choices(root, model=None):
    root = Path(root).resolve()
    paths = voice_files(root / "models")
    models = sorted(paths, key=str.casefold)
    library = root / 'models'
    indexes = sorted((p.relative_to(root).as_posix() for p in library.rglob('*.index')
                      if p.is_file() and library.resolve() in p.resolve().parents
                      and not p.name.casefold().startswith('trained_')), key=str.casefold)
    selected = model if model in models else None
    stem = Path(selected).stem.casefold() if selected else ''
    group = paths[selected].parent if selected else None
    folder_matches = [name for name in indexes if group and group != library and (root / name).parent == group]
    matches = folder_matches or [
        name for name in indexes if stem and "_" + stem + "_" in "_" + Path(name).stem.casefold() + "_"
    ]
    # Never silently choose between several indexes with the same apparent name.
    index = matches[0] if len(matches) == 1 else None
    return models, selected, indexes, index


def export_model(model, index, root):
    """Yield (status, download_path); never deserialize weights or include logs/data."""
    folder = None
    complete = False
    try:
        root = Path(root).resolve()
        models, _, indexes, _ = export_choices(root, model)
        if not model or model not in models:
            raise ValueError('Select an installed Voice Model first.')
        files = [voice_path(root / 'models', model)]
        if index:
            if index not in indexes:
                raise ValueError('Refresh the list and select an available index.')
            files.append(root / index)
        before = [p.stat() for p in files]
        if any(s.st_size == 0 for s in before):
            raise ValueError('An input file is empty. Select a complete model and index.')
        total = sum(s.st_size for s in before)
        exports = root / 'TEMP' / 'model-exports'
        exports.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(exports).free < total + 64 * 1024**2:
            raise ValueError('Not enough free disk space to create this ZIP.')
        folder = exports / uuid.uuid4().hex
        folder.mkdir()
        target = folder / (Path(model).stem + '.zip')
        temporary = folder / 'export.partial'
        done, last_update = 0, 0.0
        yield '0.0% | Preparing model ZIP...', None
        # Stored entries avoid wasting CPU recompressing large neural weights.
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                yield '%.1f%% | Packing %s...' % (done / total * 100, path.name), None
                with path.open('rb') as reader, archive.open(path.name, 'w', force_zip64=True) as writer:
                    while True:
                        chunk = reader.read(4 * 1024**2)
                        if not chunk:
                            break
                        writer.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last_update >= 0.25:
                            yield '%.1f%% | Packing %s...' % (min(99.9, done / total * 100), path.name), None
                            last_update = time.monotonic()
        for path, original in zip(files, before):
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                raise ValueError('A model file changed during export. Wait for saving to finish, then retry.')
        os.replace(str(temporary), str(target))
        complete = True
        yield '100.0%% | Model ZIP ready: %s. Click the file below to download.%s' % (
            ' + '.join(p.name for p in files), ' No index was selected.' if len(files) == 1 else ''), str(target)
    except Exception as error:
        if folder is not None and folder.exists():
            shutil.rmtree(folder)
        yield 'Export failed: ' + str(error), None
    finally:
        if not complete and folder is not None and folder.exists():
            shutil.rmtree(folder)
