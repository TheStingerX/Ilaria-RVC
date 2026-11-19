"""Resolve prepared training paths from before app/ was introduced."""
from pathlib import Path


def rebase_training_paths(fields, app_root):
    app_root = Path(app_root).resolve()
    old_logs = app_root.parent / "logs"
    result = []
    for field in fields[:-1]:
        path = Path(field)
        if path.is_absolute() and path.is_relative_to(old_logs):
            path = app_root / "logs" / path.relative_to(old_logs)
            field = str(path)
        result.append(field)
    return [*result, *fields[-1:]]


def training_row_key(row, app_root):
    fields = rebase_training_paths(row.split("|"), app_root)
    return (*[str(Path(field)) for field in fields[:-1]], *fields[-1:])
