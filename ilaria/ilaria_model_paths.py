"""Locate voice weights in model folders and legacy flat libraries."""
from pathlib import Path


def voice_files(base):
    base = Path(base).resolve()
    if not base.is_dir():
        return {}
    paths = sorted((*base.glob("*.pth"), *base.glob("*/*.pth")),
                   key=lambda p: (p.parent.name.casefold() != p.stem.casefold(),
                                  p.parent != base, str(p).casefold()))
    result = {}
    seen = set()
    for path in paths:
        if (path.is_file() and base in path.resolve().parents
                and path.name.casefold() not in seen):
            result[path.name] = path
            seen.add(path.name.casefold())
    return result


def voice_path(base, name):
    requested = str(name or "")
    if not requested or Path(requested).name != requested:
        raise ValueError("Select a Voice Model from the model list.")
    base = Path(base).resolve()
    for path in (base / Path(requested).stem / requested, base / requested):
        if path.suffix.casefold() == ".pth" and path.is_file() and base in path.resolve().parents:
            return path
    for display, path in voice_files(base).items():
        if display.casefold() == requested.casefold():
            return path
    raise ValueError("Model not found. Refresh the model list and select it again.")
