"""Stream binary uploads directly into the local RVC model importer."""

import json
from pathlib import Path
import shutil
import tempfile
import threading
from urllib.parse import unquote

import anyio
from fastapi import Request
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, StreamingResponse

from ilaria_model_import import MAX_BYTES, ImportProblem, _accepted, _clean_name, _friendly, _import_events


MAX_FILES = 64
UPLOAD_ROUTE = "/ilaria/model-import"
UPLOAD_LABEL = "model import"


def _manifest(request):
    encoded = request.headers.get("x-ilaria-files", "")
    if not encoded or len(encoded) > 12000:
        raise ImportProblem("Missing or oversized file list.")
    try:
        files = json.loads(unquote(encoded, encoding="utf-8", errors="strict"))
        length = int(request.headers.get("content-length", ""))
    except (UnicodeError, ValueError) as error:
        raise ImportProblem("Invalid file list or upload length.") from error
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ImportProblem(f"Choose 1–{MAX_FILES} files.")
    manifest = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or type(item.get("size")) is not int:
            raise ImportProblem("Invalid file information.")
        name, size = item["name"], item["size"]
        if _clean_name(name) != name or not (_accepted(name) or name.casefold().endswith(".zip")):
            raise ImportProblem("Choose a ZIP, inference PTH or INDEX file.")
        if size <= 0:
            raise ImportProblem(f"The file is empty: {name}")
        manifest.append((name, size))
    total = sum(size for _, size in manifest)
    if total != length:
        raise ImportProblem("The upload length does not match the file list.")
    if total > MAX_BYTES:
        raise ImportProblem("The upload exceeds the 16 GiB limit.")
    return manifest, total


class _UploadSink:
    def __init__(self, stage, manifest):
        self.files = [
            (stage / str(i), name, size) for i, (name, size) in enumerate(manifest)
        ]
        self.index = 0
        self.written = 0
        self.stream = None

    def write(self, chunk):
        data = memoryview(chunk)
        while data:
            if self.index == len(self.files):
                raise ImportProblem("Received more data than declared.")
            path, _, size = self.files[self.index]
            if self.stream is None:
                self.stream = path.open("xb")
            length = min(size - self.written, len(data))
            self.stream.write(data[:length])
            self.written += length
            data = data[length:]
            if self.written == size:
                self.close()
                self.index += 1
                self.written = 0

    def close(self):
        if self.stream is not None:
            stream, self.stream = self.stream, None
            stream.close()

    def finish(self):
        if self.index != len(self.files):
            raise ImportProblem("The upload was interrupted before all bytes arrived.")
        return [{"name": str(path), "orig_name": name} for path, name, _ in self.files]


def _line(percent, message, done=False):
    return json.dumps({"percent": percent, "message": message, "done": done},
                      ensure_ascii=False) + "\n"


def register_binary_model_import(server_app, root, gate):
    """Register on the FastAPI instance created by Gradio's launch method."""
    root = Path(root).resolve()
    slot = threading.BoundedSemaphore(1)

    async def upload(request: Request):
        def failure(message, code):
            return JSONResponse({"message": message, "percent": 0, "done": False}, status_code=code)

        origin = request.headers.get("origin", "")
        if not origin or origin.casefold() != f"{request.url.scheme}://{request.url.netloc}".casefold():
            return failure("Upload requests must come from this Ilaria RVC page.", 403)
        if (getattr(request.app, "auth", None) is not None
                and request.cookies.get("access-token") not in request.app.tokens):
            return failure("Sign in again to import model files.", 401)
        if request.headers.get("content-type") != "application/octet-stream":
            return failure("Expected a binary upload.", 415)
        try:
            manifest, total = _manifest(request)
        except ImportProblem as error:
            return failure(str(error), 400)
        if not slot.acquire(blocking=False):
            return failure("A model upload is already running. Wait for it to finish.", 409)
        try:
            gate.enter(UPLOAD_LABEL)
        except RuntimeError as error:
            slot.release()
            return failure(str(error), 409)

        temporary = None
        stream_owns_cleanup = False
        try:
            temp_root = root / "TEMP"
            temp_root.mkdir(exist_ok=True)
            if shutil.disk_usage(temp_root).free < total:
                return failure("There is not enough disk space for these files.", 507)
            temporary = tempfile.TemporaryDirectory(prefix="model-upload-", dir=temp_root)
            stage = Path(temporary.name)
            sink = _UploadSink(stage, manifest)
            try:
                async for chunk in request.stream():
                    if chunk:
                        await anyio.to_thread.run_sync(sink.write, chunk)
                files = sink.finish()
            finally:
                sink.close()
            events = _import_events(files, root, owned_upload_root=stage)

            async def report():
                try:
                    while True:
                        event = await anyio.to_thread.run_sync(next, events, None)
                        if event is None:
                            break
                        percent, message, done = event
                        yield _line(percent, message, done)
                except Exception as error:
                    yield _line(0, "Import failed: " + _friendly(error))
                finally:
                    with anyio.CancelScope(shield=True):
                        try:
                            await anyio.to_thread.run_sync(events.close)
                        finally:
                            cleanup()

            # An already disconnected client can cancel before entering report().
            from starlette.background import BackgroundTask

            cleaned = False

            def cleanup():
                nonlocal cleaned
                if cleaned:
                    return
                cleaned = True
                try:
                    temporary.cleanup()
                finally:
                    gate.leave(UPLOAD_LABEL)
                    slot.release()

            response = StreamingResponse(report(), media_type="application/x-ndjson",
                                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
                                         background=BackgroundTask(cleanup))
            stream_owns_cleanup = True
            return response
        except ClientDisconnect:
            return failure("Upload cancelled.", 400)
        except (OSError, ImportProblem) as error:
            return failure("Upload failed: " + _friendly(error), 400)
        finally:
            if not stream_owns_cleanup:
                try:
                    if temporary is not None:
                        temporary.cleanup()
                finally:
                    gate.leave(UPLOAD_LABEL)
                    slot.release()

    server_app.add_api_route(UPLOAD_ROUTE, upload, methods=["POST"], include_in_schema=False)
