"""Move user-owned inference files without rewriting their contents."""

from pathlib import Path
import json
import stat

from ilaria_release_migration import _write

RECORD = ".ilaria-updates/model-storage-1.8.json"


def _safe(root, name):
    rel = Path(name)
    if rel.is_absolute() or not rel.parts or any(part in (".", "..") or ":" in part for part in rel.parts):
        raise ValueError("Invalid model migration path.")
    current = root
    for part in rel.parts:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Model migration cannot follow a link: {name}")
    return current


def _target(relative):
    path = Path(relative)
    parts = path.parts
    if ((len(parts) == 3 and parts[:2] == ("assets", "weights"))
            or (len(parts) == 2 and parts[0] == "models")) and path.suffix.casefold() == ".pth":
        return (Path("models") / path.stem / path.name).as_posix()
    if (parts[0] == "logs" and path.suffix.casefold() == ".index"
            and not path.name.casefold().startswith("trained_")):
        return (Path("models") / path.relative_to("logs")).as_posix()
    raise ValueError("Unexpected model migration entry.")


def _matches(path, row):
    if not path.is_file():
        return False
    details = path.stat()
    return details.st_size == row["size"] and details.st_mtime_ns == row["modified"]


def _paths(root, row):
    if _target(row["source"]) != row["target"]:
        raise ValueError("Invalid model migration destination.")
    return _safe(root, row["source"]), _safe(root, row["target"])


def _save(path, record):
    _write(path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def migrate_models(root, payload_hash):
    root = Path(root).resolve()
    record_path = _safe(root, RECORD)
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["complete"]:
            return
        if record["payload_hash"] != payload_hash:
            raise ValueError("A different model migration is already recorded.")
    else:
        sources = []
        for folder in ("assets/weights", "models"):
            sources.extend(p for p in _safe(root, folder).glob("*.pth") if p.is_file())
        sources.extend(p for p in _safe(root, "logs").rglob("*.index")
                       if p.is_file() and not p.name.casefold().startswith("trained_"))
        rows, targets = [], set()
        for source in sources:
            name = source.relative_to(root).as_posix()
            details = _safe(root, name).stat()
            row = {"source": name, "target": _target(name),
                   "size": details.st_size, "modified": details.st_mtime_ns}
            _, target = _paths(root, row)
            key = row["target"].casefold()
            if key in targets or target.exists():
                raise ValueError(f"Model migration found an existing file at {row['target']}. Both copies were kept.")
            targets.add(key)
            rows.append(row)
        record = {"payload_hash": payload_hash, "complete": False, "moves": rows}
        _save(record_path, record)
    for row in record["moves"]:
        source, target = _paths(root, row)
        if not source.exists() and _matches(target, row):
            continue
        if target.exists() or not _matches(source, row):
            raise ValueError(f"Model migration cannot safely move {row['source']}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    record["complete"] = True
    _save(record_path, record)


def restore_models(root, payload_hash):
    root = Path(root).resolve()
    record_path = _safe(root, RECORD)
    if not record_path.is_file():
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["payload_hash"] != payload_hash:
        return
    pending = []
    for row in record["moves"]:
        source, target = _paths(root, row)
        if _matches(source, row) and not target.exists():
            continue
        if source.exists() or not _matches(target, row):
            raise ValueError(f"Cannot restore {row['source']} because the model files changed after the update.")
        pending.append((source, target))
    for source, target in reversed(pending):
        source.parent.mkdir(parents=True, exist_ok=True)
        target.rename(source)
    record_path.unlink()
