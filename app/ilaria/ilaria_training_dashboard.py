"""Read training metrics and manage an optional local TensorBoard server."""
import atexit
import json
from collections import deque
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
from threading import Lock
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ilaria_training_tools import experiment_path

EPOCH = re.compile(r"Epoch: (\d+) \|\| Total Steps: (\d+) \|\| Time taken: (\d+)h\s+(\d+)m\s+(\d+(?:\.\d+)?)s")
PRIVATE_PATH = re.compile(r'(?:[A-Za-z]:[\\/]|\\\\)[^\r\n]*')


def training_metrics(root, name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from matplotlib.figure import Figure

    folder = experiment_path(root, name)
    rows = []
    previous = None
    log = folder / "train.log"
    if log.is_file():
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = EPOCH.search(line)
                if not match:
                    continue
                epoch, steps, hours, minutes, seconds = map(float, match.groups())
                duration = hours * 3600 + minutes * 60 + seconds
                rate = (steps - previous[1]) / duration if previous and epoch == previous[0] + 1 and steps > previous[1] and duration else None
                rows.append([int(epoch), int(steps), duration, round(rate, 2) if rate else None])
                previous = epoch, steps
    event_files = sorted(folder.glob("events.out.tfevents.*"), key=lambda path: path.name) if folder.is_dir() else []
    if not rows and not event_files:
        return "No training metrics found for this model yet.", None, []
    figure = Figure(figsize=(9, 5), layout="tight")
    epochs, losses = figure.subplots(2, 1)
    epochs.set(xlabel="Epoch", ylabel="Seconds", title="Epoch duration")
    losses.set(xlabel="Step", ylabel="Loss", title="Training loss")
    if rows:
        epochs.plot([row[0] for row in rows], [row[2] for row in rows])
    if event_files:
        events = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 1000, "images": 1, "histograms": 1, "tensors": 1})
        events.Reload()
        tags = events.Tags()["scalars"]
        for tag in ("loss/g/total", "loss/d/total", "loss/g/mel"):
            if tag in tags:
                values = events.Scalars(tag)
                losses.plot([item.step for item in values], [item.value for item in values], label=tag)
        if losses.lines:
            losses.legend()
    return f"{len(rows)} completed epoch records found.", figure, rows[-20:][::-1]


def diagnostic_report(root, name, version):
    import torch

    os_label = f"{platform.system()} {platform.release()}"
    if sys.platform == "win32":
        from winreg import HKEY_LOCAL_MACHINE, OpenKey, QueryValueEx

        build = sys.getwindowsversion().build
        try:
            with OpenKey(HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                build = int(QueryValueEx(key, "CurrentBuildNumber")[0])
                installation_type = QueryValueEx(key, "InstallationType")[0]
            if installation_type == "Client":
                if build >= 22000:
                    os_label = "Windows 11"
                elif build >= 10240:
                    os_label = "Windows 10"
        except (OSError, ValueError):
            pass
        os_label += f" (build {build})"

    lines = [
        f"Ilaria RVC {version}", f"OS: {os_label}",
        f"Python: {platform.python_version()}", f"PyTorch: {torch.__version__}",
        f"CUDA runtime: {torch.version.cuda or 'Unavailable'}",
        f"Free disk: {shutil.disk_usage(root).free / 1024**3:.1f} GiB",
    ]
    for index in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(index)
        lines.append(f"GPU {index}: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)")
    if name:
        folder = experiment_path(root, name)
        errors = deque(maxlen=10)
        for filename in ("train.log", "training_console.log"):
            path = folder / filename
            if path.is_file():
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        lower = line.lower()
                        if any(token in lower for token in ("runtimeerror:", "valueerror:", "out of memory", "error:", "failed:")):
                            message = re.split(r"(?:RuntimeError|ValueError|OSError|Error):", line, maxsplit=1, flags=re.I)[-1]
                            message = PRIVATE_PATH.sub("[path hidden]", message)
                            message = message.replace(name, "[model]")
                            errors.append(message.strip()[:300])
        lines.extend(["", "Recent training errors (review before sharing):", *errors] if errors else ["", "No matching training errors found."])
    return "\n".join(lines)


class LocalTensorBoard:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = Lock()
        self.process = None
        self.url = None
        atexit.register(self.stop)

    def start(self):
        from importlib.util import find_spec

        with self.lock:
            if self.process and self.process.poll() is None:
                return self.url
            if find_spec("tensorboard") is None:
                raise ValueError("TensorBoard is missing from the local Python runtime.")
            logs = self.root / "logs"
            logs.mkdir(exist_ok=True)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("ilaria_tensorboard_entry.py")), "--logdir", str(logs), "--load_fast=false", "--reload_interval", "2", "--host", "127.0.0.1", "--port", str(port)],
                cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            url = f"http://127.0.0.1:{port}/"
            opener = build_opener(ProxyHandler({}))
            deadline = monotonic() + 15
            while monotonic() < deadline and self.process.poll() is None:
                try:
                    with opener.open(url, timeout=0.5):
                        self.url = url
                        return url
                except OSError:
                    sleep(0.2)
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None
            self.url = None
            raise RuntimeError("TensorBoard did not respond on the local port. Try starting it again.")

    def charts_url(self, model=None):
        url = self.start()
        opener = build_opener(ProxyHandler({}))
        deadline = monotonic() + 25
        while monotonic() < deadline:
            try:
                with opener.open(url + "data/plugins_listing", timeout=1) as response:
                    enabled = json.load(response).get("scalars", {}).get("enabled")
                if enabled:
                    with opener.open(url + "data/plugin/scalars/tags", timeout=2) as response:
                        tags = json.load(response)
                    if any(series for run, series in tags.items() if not model or run.casefold() == model.casefold()):
                        return url + "#scalars"
            except (OSError, ValueError):
                pass
            sleep(0.5)
        return None

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            self.url = None
        return "TensorBoard stopped."



def refresh_training_metrics(root, name):
    import json

    try:
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), str(root), str(name or "")],
            cwd=root, capture_output=True, encoding="utf-8", timeout=12,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return "Reading training metrics timed out. Try again after the current epoch.", "", []
    if result.returncode:
        return "Could not read the selected model's training metrics.", "", []
    return json.loads(result.stdout)



def live_training_metrics(root, name, log_offset=None):
    folder = experiment_path(root, name)
    active = log_offset is not None
    result = {"model": name, "active": active, "values": []}
    try:
        with (folder / "train.log").open("rb") as log:
            end = log.seek(0, os.SEEK_END)
            start = min(log_offset or 0, end)
            tail = max(start, end - 128 * 1024)
            log.seek(tail)
            if tail > start:
                log.readline()
            records = log.read(end - log.tell()).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return result
    records = records.split("\n")[:-1]
    completed = None
    progress = None
    losses = None
    batch = None
    for line in records:
        message = line.rstrip().rsplit("\t", 1)[-1]
        if EPOCH.search(message):
            progress = None
            completed = message
        if message.startswith(("Train Epoch:", "STEP_PROGRESS!", "RESUMED!")):
            progress = message
        if message.startswith("[") and re.match(r"^\[\d+,", message):
            batch = message.strip("[]").split(", ", 1)
        if message.startswith("loss_disc="):
            losses = (message, batch)
    if completed:
        match = EPOCH.search(completed)
        values = result["values"]
        values.extend([["Last recorded epoch", match[1]], ["Steps at epoch record", match[2]],
                       ["Epoch duration", f"{match[3]}h {match[4]}m {match[5]}s"]])
        for label, pattern in (("Remaining", r"ETA\(est\): ([^|]+)"),
                               ("Estimated finish", r"End\(est\): ([^|]+)")):
            found = re.search(pattern, completed)
            if found:
                values.append([label, found[1].strip()])
    if progress:
        result["values"].append(["Latest progress", progress.removeprefix("STEP_PROGRESS! ")])
    if losses:
        message, batch = losses
        if batch and len(batch) == 2:
            result["values"].extend([["Loss recorded at step", batch[0]], ["Learning rate", batch[1]]])
        labels = {"loss_disc": "Discriminator loss", "loss_gen": "Generator adversarial loss", "loss_fm": "Feature matching loss",
                  "loss_mel": "Mel loss", "loss_kl": "KL loss"}
        result["values"].extend([labels[key], value] for key, value in re.findall(r"(loss_\w+)=([^,\s]+)", message)
                                if key in labels)
    return result


def register_training_metrics(server, root, training_session=None):
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
    from threading import BoundedSemaphore

    slot = BoundedSemaphore(1)

    async def read_live_metrics(request: Request):
        if (getattr(request.app, "auth", None) is not None
                and request.cookies.get("access-token") not in request.app.tokens):
            return JSONResponse({"status": "Sign in again to view training metrics."}, status_code=401)
        if request.headers.get("origin", "").casefold() != f"{request.url.scheme}://{request.url.netloc}".casefold():
            return JSONResponse({"status": "Open the dashboard from the Ilaria RVC page."}, status_code=403)
        active = training_session.active_training() if training_session is not None else None
        name, offset = active or (request.query_params.get("model", "").strip(), None)
        if not name:
            return {"status": "Start training or enter a Model Name to see its last recorded values."}
        try:
            return await run_in_threadpool(live_training_metrics, root, name, offset)
        except (OSError, ValueError):
            return {"status": "Could not read metrics for the selected model."}

    server.add_api_route("/ilaria/live-training", read_live_metrics, methods=["POST"], include_in_schema=False)

    async def read_metrics(request: Request):
        if getattr(request.app, "auth", None) is not None and request.cookies.get("access-token") not in request.app.tokens:
            return JSONResponse({"status": "Sign in again to refresh training metrics."}, status_code=401)
        if request.headers.get("origin", "").casefold() != f"{request.url.scheme}://{request.url.netloc}".casefold():
            return JSONResponse({"status": "Open the dashboard from the Ilaria RVC page."}, status_code=403)
        name = request.query_params.get("model", "").strip()
        if not name:
            return {"status": "Enter a Model Name in Training, then refresh."}
        try:
            experiment_path(root, name)
        except ValueError as error:
            return {"status": str(error)}
        if not slot.acquire(blocking=False):
            return {"status": "Training metrics are already being read. Try again in a moment."}
        try:
            status, image, rows = await run_in_threadpool(refresh_training_metrics, root, name)
            return {"status": status, "image": image, "rows": rows}
        except (OSError, ValueError):
            return {"status": "Could not read training metrics for this model."}
        finally:
            slot.release()

    server.add_api_route("/ilaria/training-metrics", read_metrics, methods=["POST"], include_in_schema=False)


if __name__ == "__main__":
    import base64
    import io
    import json

    status, figure, rows = training_metrics(sys.argv[1], sys.argv[2])
    image = ""
    if figure is not None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        image = base64.b64encode(buffer.getvalue()).decode("ascii")
    print(json.dumps([status, image, rows]))
