# -*- coding: utf-8 -*-

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[0]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))


print("Ilaria RVC is starting...", flush=True)
# Legacy updater packages carry an automatic support-folder payload.
_release_payload_path = _IlariaBootstrapPath(__file__).resolve().parent / "ilaria_release_payload.py"

if not _release_payload_path.is_file():
    _release_payload_path = _ilaria_install_root / "ilaria" / "ilaria_release_payload.py"
if _release_payload_path.is_file():
    import importlib.util as _release_import
    _release_spec = _release_import.spec_from_file_location("_ilaria_release_payload", _release_payload_path)
    _release_payload = _release_import.module_from_spec(_release_spec)
    _release_spec.loader.exec_module(_release_payload)
    _release_payload.ensure_layout(_ilaria_install_root)
from pathlib import Path as _StartupPath
import json as _startup_json
try:
    _startup_version = _startup_json.loads((_StartupPath(__file__).resolve().parent / "ilaria" / "ilaria_version.json").read_text(encoding="utf-8"))["version"]
except (OSError, ValueError, KeyError, TypeError):
    _startup_version = "Unknown"
print("Version: " + str(_startup_version), flush=True)
import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
if os.name == "nt" and __name__ == "__main__":
    from ilaria_hubert_assets import start_missing_hubert_downloads

    start_missing_hubert_downloads(now_dir, lambda message: print(message, flush=True))

from infer.modules.vc.modules import VC
from ilaria_inference_pipeline import preload_rmvpe, release_rmvpe
from i18n.i18n import I18nAuto
from configs.config import Config
import torch
import numpy as np
import gradio as gr
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

logging.getLogger("numba").setLevel(logging.WARNING)

class QuietModelLoading(logging.Filter):
    def filter(self, record):
        return not (record.levelno == logging.INFO and
                    record.getMessage().startswith(("Get sid:", "Loading:", "Select index:")))


logging.getLogger("infer.modules.vc.modules").addFilter(QuietModelLoading())

class QuietWindowsTransportReset(logging.Filter):
    """Keep unrelated connection and application errors visible."""
    def filter(self, record):
        if not record.exc_info or not record.getMessage().startswith(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost("
        ):
            return True
        error = record.exc_info[1]
        return not (isinstance(error, ConnectionResetError)
                    and getattr(error, "winerror", None) == 10054)


if os.name == "nt":
    for logger_name in ("", "asyncio"):
        logging.getLogger(logger_name).addFilter(QuietWindowsTransportReset())



logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % now_dir, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "models"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

CPU_THREADS = 8
MAX_CPU_THREADS = 12

# Keep startup diagnostics readable without hiding warnings or errors.
from contextlib import redirect_stdout
from io import StringIO
startup_output = StringIO()
config_logger = logging.getLogger("configs.config")
previous_config_level = config_logger.level
try:
    config_logger.setLevel(logging.WARNING)
    with redirect_stdout(startup_output):
        config = Config()
finally:
    config_logger.setLevel(previous_config_level)
for startup_line in startup_output.getvalue().splitlines():
    if not startup_line.startswith(("is_half:", "overwrite preprocess and configs.json")):
        print(startup_line, flush=True)
config.n_cpu = CPU_THREADS
vc = VC(config)
voice_lock = threading.RLock()
loaded_voice = None
voice_selection_started = False
threading.Thread(target=preload_rmvpe, args=(config,), name="RMVPE startup", daemon=True).start()

if config.dml:
    import fairseq

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
from datetime import datetime
print("PC time: " + datetime.now().strftime("%d-%m-%Y %H:%M:%S"), flush=True)
print("GPU: " + str(config.gpu_name or "No NVIDIA GPU detected"), flush=True)
if str(config.device).startswith("cuda") and torch.cuda.is_available():
    startup_gpu = torch.cuda.get_device_properties(torch.device(config.device))
    print("GPU memory: %.1f GiB" % (startup_gpu.total_memory / 1024**3), flush=True)
print("Device: " + str(config.device), flush=True)
print("Language: " + str(i18n.language), flush=True)
print("PyTorch: " + str(torch.__version__), flush=True)
# Gradio prints this only after the local server has started successfully.
gr.strings.en["RUNNING_LOCALLY_SEPARATED"] = "Status: Ready\nWeb UI: {0}://127.0.0.1:{2}"
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
else:
    gpu_info = i18n("Your GPU doesn't work for training")

default_batch_size = 8
gpus = "-".join([i[0] for i in gpu_infos])

class ToolButton(gr.Button, gr.components.FormComponent):

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

weight_root = index_root = "models"
os.environ["weight_root"] = weight_root
os.environ["index_root"] = index_root

from ilaria_model_paths import voice_files, voice_path


def inference_choices():
    models = sorted(voice_files(weight_root), key=str.casefold)
    indexes = sorted((path.as_posix() for path in pathlib.Path(index_root).rglob("*")
                      if path.is_file() and path.suffix.casefold() == ".index"
                      and not path.name.casefold().startswith("trained_")), key=str.casefold)
    return models, indexes


names, index_paths = inference_choices()


def change_choices():
    models, indexes = inference_choices()
    return (
        {"choices": models, "value": models[0] if models else None, "__type__": "update"},
        {"choices": indexes, "__type__": "update"},
    )


# This fragment is inserted into ILARIA_RVC.py before its existing callbacks.
import functools
import inspect


class UpdateOperationGate:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = {}
        self.children = []
        self.updating = False

    def enter(self, label):
        with self.lock:
            if self.updating:
                raise RuntimeError('An update is being prepared. Wait for Ilaria RVC to restart.')
            self.active[label] = self.active.get(label, 0) + 1

    def leave(self, label):
        with self.lock:
            self.active[label] -= 1
            if not self.active[label]:
                del self.active[label]

    def track(self, function, label):
        if inspect.isgeneratorfunction(function):
            @functools.wraps(function)
            def generator(*args, **kwargs):
                self.enter(label)
                try:
                    yield from function(*args, **kwargs)
                finally:
                    self.leave(label)
            return generator
        @functools.wraps(function)
        def synchronous(*args, **kwargs):
            self.enter(label)
            try:
                return function(*args, **kwargs)
            finally:
                self.leave(label)
        return synchronous

    def process(self, *args, **kwargs):
        with self.lock:
            if self.updating:
                raise RuntimeError('An update is being prepared. New processes cannot start yet.')
            child = _original_popen(*args, **kwargs)
            self.children.append(child)
            return child

    def begin_update(self, ignored_operations=()):
        with self.lock:
            self.children = [child for child in self.children if child.poll() is None]
            if self.updating:
                raise RuntimeError('An update is already being prepared.')
            blocking = self.active.keys() - set(ignored_operations)
            if blocking or self.children:
                labels = ', '.join(sorted(blocking)) or 'a background process'
                raise RuntimeError('Wait for the current operation to finish: ' + labels + '.')
            self.updating = True

    def cancel_update(self):
        with self.lock:
            self.updating = False

    def idle_for_restart(self, ignored_operations=()):
        with self.lock:
            return (self.updating and not (self.active.keys() - set(ignored_operations))
                    and not any(child.poll() is None for child in self.children))


_original_popen = Popen
update_operation_gate = UpdateOperationGate()
Popen = update_operation_gate.process
program_restart_requested = threading.Event()
program_restart_code = 42



def scan_training_advice(sample_rate="32k", precision="FP32", selected="Auto (largest compatible GPU)"):
    from ilaria_hardware_probe import collect_hardware
    from ilaria_training_advisor import hardware_report, recommendation_report, gpu_choices, AUTO_GPU
    yield "Checking local CPU, RAM, GPU, runtime and free memory...", "", "", gr.update(), None
    try:
        hardware = collect_hardware(now_dir)
        choices = gpu_choices(hardware)
        selected = selected if selected in choices else AUTO_GPU
        advice = recommendation_report(hardware, sample_rate, precision, selected)
        yield ("PC check complete | Checked at: " + hardware["checked_at"] + " | Training settings unchanged",
               hardware_report(hardware), advice, gr.update(choices=choices, value=selected), hardware)
    except Exception as error:
        yield "PC check could not finish: " + str(error), "No training settings have been changed.", "", gr.update(), None


def refresh_training_advice(hardware, sample_rate, precision, selected):
    from ilaria_training_advisor import recommendation_report
    if not hardware:
        return "Click Scan PC and suggest settings first.", ""
    try:
        return ("Suggestions updated from the scan at " + hardware["checked_at"] + "; scan again to refresh free memory.",
                recommendation_report(hardware, sample_rate, precision, selected))
    except (ValueError, KeyError, TypeError) as error:
        return "Cannot suggest settings: " + str(error), ""


def installed_program_version():
    from ilaria_updater import read_local_version
    try:
        return read_local_version(now_dir)
    except Exception:
        return 'Unknown'


def check_program_updates():
    from ilaria_updater import check_update
    hidden_notice = {'visible': False, 'value': '', '__type__': 'update'}
    yield 'Checking for updates...', '', {'visible': False, '__type__': 'update'}, hidden_notice
    try:
        result = check_update(now_dir)
        notes = '\n'.join('- ' + str(item) for item in result.get('changelog', []))
        available = bool(result.get('available'))
        message = ('Update available: ' + result['version']) if available else result.get('message', 'You are up to date.')
        notice = {'visible': available, 'value': '**Ilaria RVC %s is available.** Open **Other > Update Ilaria RVC** to see the changes and install it.' % result['version'] if available else '', '__type__': 'update'}
        yield message, notes, {'visible': available, '__type__': 'update'}, notice
    except Exception as error:
        yield 'Could not check for updates: ' + str(error), '', {'visible': False, '__type__': 'update'}, hidden_notice


def download_program_update():
    from ilaria_updater import check_update, prepare_update
    acquired = False
    restarting = False
    try:
        if os.environ.get('ILARIA_RVC_MANAGED_LAUNCHER') != '1':
            yield '0.0% | Close this UI and start ILARIA_RVC.bat to enable automatic installation and restart.'
            return
        if config.iscolab:
            yield '0.0% | Automatic restart is available in the local Windows installation.'
            return
        update_operation_gate.begin_update()
        acquired = True
        yield '0.0% | Checking the latest release...'
        result = check_update(now_dir)
        if not result.get('available'):
            yield '0.0% | ' + result.get('message', 'No newer update is available.')
            return
        yield from prepare_update(now_dir, result['manifest'])
        restarting = True
        timer = threading.Timer(1.0, program_restart_requested.set)
        timer.daemon = True
        timer.start()
        yield '100.0%% | Restarting to install version %s. This page will reload when ready.' % result['version']
    except Exception as error:
        yield 'Update stopped: ' + str(error)
    finally:
        if acquired and not restarting:
            update_operation_gate.cancel_update()


RELOAD_INTERRUPTIBLE_OPERATIONS = frozenset({'inference warmup', 'model loading'})


def restart_local_ui():
    global program_restart_code
    if config.iscolab or os.environ.get("ILARIA_RVC_LIFECYCLE_V2") != "1":
        return "Close this instance and start the new ILARIA_RVC.bat to enable restart."
    try:
        update_operation_gate.begin_update(RELOAD_INTERRUPTIBLE_OPERATIONS)
    except RuntimeError as error:
        return str(error)
    program_restart_code = 43
    timer = threading.Timer(0.5, program_restart_requested.set)
    timer.daemon = True
    timer.start()
    return "Restarting Ilaria RVC. This page will reload when ready."


RELOAD_LOCAL_UI_JS = """(status, previous) => {
    if (!String(status).startsWith("Restarting Ilaria RVC.")) return [];
    let attempts = 0;
    async function check() {
        try {
            const response = await fetch("config", {cache: "no-store"});
            if (response.ok) {
                const config = await response.json();
                const instance = config.components.find(item => item.props?.elem_id === "reload-ui-instance");
                if (instance && String(instance.props.value) !== String(previous)) {
                    window.location.reload();
                    return;
                }
            }
        } catch (_) {}
        if (++attempts < 120) window.setTimeout(check, 1000);
    }
    window.setTimeout(check, 1000);
    return [];
}"""


class BrowserPresence:
    def __init__(self):
        self.lock = threading.Lock()
        self.tabs = {}
        self.connected = False
        self.changed = threading.Event()

    def record(self, tab, opened, sequence):
        from time import monotonic

        if not isinstance(tab, str) or not 12 <= len(tab) <= 80:
            return
        with self.lock:
            previous = self.tabs.get(tab)
            if previous and sequence < previous[0]:
                return
            if opened:
                self.connected = True
            elif previous is None:
                return
            self.tabs[tab] = (sequence, monotonic() + (135 if opened else 0))
            self.changed.set()

    def shutdown_on_close(self):
        from time import monotonic

        while not program_restart_requested.is_set():
            self.changed.wait(1)
            self.changed.clear()
            with self.lock:
                now = monotonic()
                self.tabs = {key: value for key, value in self.tabs.items() if value[1] > now}
                if self.connected and not self.tabs and not program_restart_requested.is_set():
                    # The launcher closes its Windows job and terminates this server's descendants.
                    os._exit(0)


browser_presence = BrowserPresence()
BROWSER_PRESENCE_JS = """() => {
    if (window.ilariaPresenceStarted) return [];
    window.ilariaPresenceStarted = true;
    const tab = Date.now().toString(36) + Math.random().toString(36).slice(2);
    let sequence = 0;
    let closed = false;
    function send(opened) {
        const data = JSON.stringify({data: [tab, opened, ++sequence]});
        fetch("api/ui_presence", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: data, keepalive: true, cache: "no-store"
        }).catch(() => {});
    }
    window.addEventListener("pagehide", () => { closed = true; send(false); });
    window.addEventListener("pageshow", () => { closed = false; send(true); });
    document.addEventListener("visibilitychange", () => { if (!closed) send(true); });
    window.setInterval(() => { if (!closed) send(true); }, 15000);
    send(true);
    return [];
}"""


def wait_for_program_restart(application):
    try:
        while not program_restart_requested.wait(0.5):
            pass
        ignored = RELOAD_INTERRUPTIBLE_OPERATIONS if program_restart_code in (0, 43) else ()
        if not update_operation_gate.idle_for_restart(ignored):
            raise RuntimeError('Restart refused because an operation is still running.')
        # The package is already verified and durable. Give the final UI message
        # time to arrive, then close the server in the main thread.
        sleep(1.0)
        watchdog = threading.Timer(8.0, lambda: os._exit(program_restart_code))
        watchdog.daemon = True
        watchdog.start()
        application.close(verbose=False)
        raise SystemExit(program_restart_code)
    except KeyboardInterrupt:
        application.close(verbose=False)


def notify_launcher_ready():
    from ilaria_release_migration import mark_ready
    mark_ready(now_dir, keep_updater=os.environ.get("ILARIA_RVC_MANAGED_LAUNCHER") == "1" and os.environ.get("ILARIA_RVC_NESTED_LAUNCHER") != "1")
    token = os.environ.get('ILARIA_RVC_READY_TOKEN', '')
    if not token:
        return
    if len(token) != 32 or any(letter not in '0123456789abcdef' for letter in token):
        raise RuntimeError('Invalid launcher readiness token.')
    folder = pathlib.Path(now_dir) / '.ilaria-updates'
    if folder.is_symlink() or getattr(folder.stat(), 'st_file_attributes', 0) & 0x400:
        raise RuntimeError('The updater state folder must be a normal directory.')
    with (folder / ('ready-' + token)).open('x', encoding='ascii') as stream:
        stream.write(token)
        stream.flush()
        os.fsync(stream.fileno())


UPDATER_RELOAD_JS = """(status) => {
    const match = String(status || '').match(/Restarting to install version ([0-9]+\\.[0-9]+\\.[0-9]+)/);
    if (!match) return [];
    const expected = match[1];
    let attempts = 0;
    const timer = window.setInterval(async () => {
        if (++attempts > 180) { window.clearInterval(timer); return; }
        try {
            const response = await fetch('config', {cache: 'no-store'});
            if (!response.ok) return;
            const data = await response.json();
            const ready = (data.components || []).some(component => component.props &&
                component.props.elem_id === 'updater-installed-version' && component.props.value === expected);
            if (ready) { window.clearInterval(timer); window.location.reload(); }
        } catch (error) { /* The server is restarting. */ }
    }, 1000);
    return [];
}"""


def select_voice_model(sid, protect_single, protect_batch):
    global loaded_voice, voice_selection_started
    with voice_lock:
        voice_selection_started = True
        loaded_voice = None
        if not sid:
            release_rmvpe()
            from ilaria_pitch import release_fcpe
            release_fcpe()
        result = list(vc.get_vc(sid or "", protect_single, protect_batch, find_index=False, reuse_loaded=True))
        loaded_voice = sid or None
    _, choices = inference_choices()
    model = pathlib.Path(str(sid or "").replace("\\", "/")).stem.casefold()
    matches = []
    if model:
        group = voice_path(weight_root, sid).parent.name.casefold()
        for path in choices:
            parts = path.replace("\\", "/").casefold().split("/")
            index_name = pathlib.Path(parts[-1]).stem
            if group in parts[1:-1]:
                matches.append((0, path))
            elif index_name == model:
                matches.append((1, path))
            elif "_" + model + "_" in "_" + index_name + "_":
                matches.append((2, path))
    selected = min(matches, key=lambda item: (item[0], item[1].casefold()))[1] if matches else ""
    update = {"choices": choices, "value": selected or None, "__type__": "update"}
    result[3:5] = [dict(update), dict(update)]
    return tuple(result) + (selected, selected, dict(result[0]) if isinstance(result[0], dict) else result[0])


def clean():
    return {"value": "", "__type__": "update"}

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

def preprocess_percentage(log, dataset):
    total = len(os.listdir(dataset))
    completed = len({line.split("->", 1)[0] for line in log.splitlines() if "->Suc." in line})
    return min(99.9, completed / total * 100) if total else 0.0


def feature_percentage(exp_dir, version, with_pitch):
    root = os.path.join(now_dir, "logs", exp_dir)
    wav_dir = os.path.join(root, "1_16k_wavs")
    if not os.path.isdir(wav_dir):
        return 0.0
    wavs = [name for name in os.listdir(wav_dir) if name.endswith(".wav")]
    if not wavs:
        return 0.0
    def names(folder):
        path = os.path.join(root, folder)
        return set(os.listdir(path)) if os.path.isdir(path) else set()
    features = names("3_feature256" if version == "v1" else "3_feature768")
    done = sum(name[:-4] + ".npy" in features for name in wavs)
    total = len(wavs)
    if with_pitch:
        coarse, fine = names("2a_f0"), names("2b-f0nsf")
        done += sum(name + ".npy" in coarse and name + ".npy" in fine for name in wavs)
        total *= 2
    return min(99.9, done / total * 100)


def compact_workflow_log(log, stage, finished=False, successful=True, percentage=0.0):
    """Present a single useful status, keeping verbose diagnostics on disk."""
    import re
    title = "Pre-Process" if stage == "preprocess" else "Feature Extraction"
    if "Traceback (most recent call last)" in log or not successful:
        return title + " failed; check " + ("preprocess.log" if stage == "preprocess" else "extract_f0_feature.log")
    if finished:
        return title + " done! | 100.0%"
    status = "start preprocess" if stage == "preprocess" else ("F0: preparing pitch extraction..." if stage == "f0" else "Extracting audio features...")
    for line in log.splitlines():
        line = line.strip()
        if not line or line.startswith("['"):
            continue
        if stage == "preprocess":
            if "->Suc." in line:
                status = "Pre-Process: " + line.split("->", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
            elif line == "start preprocess":
                status = line
        elif stage == "f0":
            match = re.search(r"f0ing,now-(\d+),all-(\d+),-(.+)", line)
            if match:
                filename = match.group(3).replace("\\", "/").rsplit("/", 1)[-1]
                status = "F0 pitch extraction: " + filename
        else:
            match = re.search(r"now-(\d+),all-(\d+),([^,]+)", line)
            if match:
                status = "Extracting audio features: " + match.group(3)
            elif line.startswith("load model(s)"):
                status = "Loading HuBERT feature model..."
            elif line.startswith("move model"):
                status = "Preparing HuBERT feature model..."
    return f"{status} | {percentage:.1f}%"


def check_saved_training_sample_rate(exp_dir, sample_rate):
    config_path = os.path.join(exp_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            saved_rate = int(json.load(stream)["data"]["sampling_rate"])
    except FileNotFoundError:
        return
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "Cannot read the saved training configuration. Restore it or use a new Model Name."
        ) from error
    selected_rate = sr_dict[sample_rate]
    if saved_rate != selected_rate:
        raise ValueError(
            f"This Model Name uses {saved_rate / 1000:g} kHz, but you selected "
            f"{selected_rate / 1000:g} kHz. Keep the original Sample Rate or use "
            "a new Model Name and run Process Data and Feature Extraction again."
        )


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    if globals().get("training_queue") is not None and training_queue.blocks_manual():
        yield "Error: The training queue is running. Pause it before starting a manual step."
        return
    from ilaria_dataset_upload import validate_dataset_path
    try:
        trainset_dir = validate_dataset_path(trainset_dir, now_dir)
    except (OSError, ValueError) as error:
        yield "Error: " + (str(error) if isinstance(error, ValueError) else "Cannot read dataset folder.")
        return
    if not str(exp_dir or "").strip():
        yield "Enter a Model Name in Training before starting this step."
        return
    try:
        check_saved_training_sample_rate(os.path.join(now_dir, "logs", exp_dir), sr)
    except ValueError as error:
        yield "Pre-Process error: " + str(error)
        return
    n_p = max(1, min(MAX_CPU_THREADS, int(n_p)))
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield compact_workflow_log(log := f.read(), "preprocess", percentage=preprocess_percentage(log, trainset_dir))
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield compact_workflow_log(log, "preprocess", finished=True, successful=p.returncode == 0 and "end preprocess" in log)

def training_dataset_error(dataset_path):
    from ilaria_dataset_upload import validate_dataset_path
    try:
        validate_dataset_path(dataset_path, now_dir)
    except ValueError as error:
        return "Error: " + str(error)
    except OSError:
        return "Error: Cannot read dataset folder."
    return None


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe, precision="FP32", dataset_path="datasets", embedder_model="contentvec"):
    if globals().get("training_queue") is not None and training_queue.blocks_manual():
        yield "Error: The training queue is running. Pause it before starting a manual step."
        return
    error = training_dataset_error(dataset_path)
    if error:
        yield error
        return
    if not str(exp_dir or "").strip():
        yield "Enter a Model Name in Training before starting this step."
        return
    audio_dir = pathlib.Path(now_dir) / "logs" / str(exp_dir) / "1_16k_wavs"
    if not audio_dir.is_dir() or not any(p.is_file() and p.suffix.lower() == ".wav" and p.stat().st_size > 0 for p in audio_dir.iterdir()):
        yield "Error: Run 1. Process Data first."
        return
    if precision not in ("FP16", "FP32"):
        yield "Choose FP16 or FP32 in Training Settings & Other."
        return
    if precision == "FP16" and not torch.cuda.is_available():
        yield "FP16 requires a CUDA GPU. Select FP32 in Training Settings & Other."
        return
    from ilaria_embedders import ensure_embedder, check_feature_embedder, save_feature_embedder
    try:
        check_feature_embedder(pathlib.Path(now_dir)/"logs"/str(exp_dir), embedder_model, version19)
        yield from ensure_embedder(now_dir, embedder_model)
        save_feature_embedder(pathlib.Path(now_dir)/"logs"/str(exp_dir), embedder_model)
    except (OSError, ValueError) as error:
        yield "Feature Extraction error: " + str(error)
        return
    f0method = "pm" if f0method == "parselmouth" else f0method
    n_p = max(1, min(MAX_CPU_THREADS, int(n_p)))
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            if f0method in ("swift", "beatrice", "fcpe"):
                cmd = __import__("subprocess").list2cmdline([config.python_cmd,
                    str(pathlib.Path(now_dir)/"ilaria"/"ilaria_pitch.py"), "--extract",
                    str(pathlib.Path(now_dir)/"logs"/str(exp_dir)), f0method])
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            precision == "FP16",
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )
                    ps.append(p)
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield compact_workflow_log(f.read(), "f0", percentage=feature_percentage(exp_dir, version19, if_f0))
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield compact_workflow_log(log, "f0", percentage=feature_percentage(exp_dir, version19, if_f0))
        pitch_processes = ps if f0method == "rmvpe_gpu" and gpus_rmvpe != "-" else [p]
        if any(process.returncode != 0 for process in pitch_processes) or "f0fail-" in log or "Traceback (most recent call last)" in log:
            yield "Feature Extraction failed; check extract_f0_feature.log."
            return

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        cmd += " --precision " + precision + " --embedder " + embedder_model
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )
        ps.append(p)
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield compact_workflow_log(f.read(), "feature", percentage=feature_percentage(exp_dir, version19, if_f0))
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    success = all(p.returncode == 0 for p in ps) and "all-feature-done" in log and "Traceback (most recent call last)" not in log
    if success:
        save_feature_embedder(pathlib.Path(now_dir)/"logs"/str(exp_dir), embedder_model)
    yield compact_workflow_log(log, "feature", finished=True, successful=success)


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["32k", "40k", "48k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )

def change_f0(if_f0_3, sr2, version19, f0method8="rmvpe"):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        change_f0_method(f0method8, if_f0_3),
        *get_pretrained_models(path_str, "f0" if if_f0_3 is True else "", sr2),
    )

def stream_training_progress(process, log_path, total_epochs, offset=0, total_steps=None):
    """Publish each confirmed epoch together with its matching ETA."""
    import re
    target_steps = int(total_steps) if total_steps is not None else None
    if target_steps is not None and target_steps < 1:
        raise ValueError("Total Steps must be positive.")
    current_epoch = 0
    step_completion_marker = None
    completed = 0
    completed_steps = 0
    pending = ""
    epoch_line = ""
    remaining = "measuring first epoch"
    estimated_finish = "calculating..."
    completion_marker = None
    stopped_marker = None

    def status(state="Training Info", completed_at=None, inactive=False):
        if target_steps is not None:
            return ("%s: %.1f%% | Epochs: %s completed | Current Epoch: %s | Total Steps: %s/%s"
                    % (state, min(100.0, completed_steps / target_steps * 100.0), completed, current_epoch, completed_steps, target_steps)
                    + (" | Completed at: " + completed_at if completed_at else "" if inactive else
                       " | Remaining: %s | Estimated finish: %s" % (remaining, estimated_finish))
                    + ("\n" + epoch_line if epoch_line else ""))
        return ("%s: %.1f%% | Epochs: %s/%s | Total Steps: %s"
                % (state, completed / total_epochs * 100, completed, total_epochs, completed_steps)
                + (" | Completed at: " + completed_at if completed_at else "" if inactive else
                   " | Remaining: %s | Estimated finish: %s" % (remaining, estimated_finish))
                + ("\n" + epoch_line if epoch_line else ""))

    preview_manifest = pathlib.Path(log_path).parent / "previews" / "previews.json"
    preview_modified = None
    yield status()
    while True:
        running = process.poll() is None
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log:
                log.seek(0, 2)
                if log.tell() < offset:
                    offset = 0
                    pending = ""
                log.seek(offset)
                chunk = log.read()
                offset = log.tell()
        except FileNotFoundError:
            chunk = ""
        pending += chunk
        lines = pending.split("\n")
        pending = lines.pop()
        if not running and pending:
            lines.append(pending)
            pending = ""
        for line in lines:
            resumed = re.search(r"RESUMED! Epochs: (\d+) \| Total Steps: (\d+)", line)
            if resumed:
                completed = int(resumed.group(1)) if target_steps is not None else min(total_epochs, int(resumed.group(1)))
                completed_steps = int(resumed.group(2))
                yield status()
            stopped = re.search(r"STOPPED! Training saved \| Epoch: (\d+) \| Total Steps: (\d+) \| Saved at: (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if stopped:
                completed, completed_steps = int(stopped.group(1)), int(stopped.group(2))
                stopped_marker = stopped.group(3)
            console = re.search(r"Training Model: .* ====> Epoch: (\d+) \|\| Total Steps: (\d+) \|\| Time taken: \d+h \d{2}m  \d{2}s", line)
            if console:
                current_epoch = int(console.group(1))
                if target_steps is None:
                    completed = min(total_epochs, current_epoch)
                completed_steps = int(console.group(2))
                epoch_line = console.group(0)
                print(epoch_line, flush=True)
            step_progress = re.search(r"STEP_PROGRESS! Total Steps: (\d+)/(\d+) \| Epoch: (\d+) \| Completed Epochs: (\d+) \| Remaining: ([^|]+) \| Estimated finish: (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if step_progress and target_steps is not None and int(step_progress.group(2)) == target_steps:
                completed_steps = int(step_progress.group(1))
                current_epoch, completed = int(step_progress.group(3)), int(step_progress.group(4))
                remaining, estimated_finish = step_progress.group(5).strip(), step_progress.group(6)
                yield status("Finalizing model..." if completed_steps == target_steps else "Training Info")
            step_done = re.search(r"DONE! Step target reached \| Total Steps: (\d+)/(\d+)", line)
            if step_done:
                step_completion_marker = (int(step_done.group(1)), int(step_done.group(2)))
            eta = re.search(r"ETA\(est\): ([^|]+)", line)
            finish = re.search(r"End\(est\): (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if eta and finish and target_steps is None:
                remaining = eta.group(1).strip()
                estimated_finish = finish.group(1)
                # One yield per epoch, even when several records arrive together.
                yield status("Finalizing model..." if completed == total_epochs else "Training Info")
            done = re.search(r"DONE! Training completed \| Completed at: (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if done:
                completion_marker = done.group(1)
        if not running:
            reached_target = (step_completion_marker == (target_steps, target_steps) and completed_steps == target_steps) if target_steps is not None else completed == total_epochs
            success_code = process.returncode == 0 if target_steps is not None else process.returncode in (0, 2333333)
            if success_code and reached_target and completion_marker:
                print("DONE! Training completed | Completed at: " + completion_marker, flush=True)
                yield status("DONE! Training completed", completion_marker)
            elif process.returncode == 0 and stopped_marker:
                message = "STOPPED! Training saved | Epoch: %s | Total Steps: %s | Saved at: %s" % (completed, completed_steps, stopped_marker)
                print(message, flush=True)
                yield message + "\nResume with the same Model Name and settings, then click 4. Train Pth."
            else:
                yield status("Training stopped; final model was not confirmed (exit code %s); check train.log/training_console.log" % process.returncode, inactive=True)
            break
        try:
            modified = preview_manifest.stat().st_mtime_ns
        except OSError:
            modified = None
        if modified != preview_modified:
            preview_modified = modified
            yield status()
        sleep(0.25)


def batch_inference_audio(sid, folder, output_folder, audio_path, pitch, method,
                          index_path, detected_index, index_ratio, filter_radius,
                          resample, volume_mix, protect, output_format, *audio_controls):
    from types import SimpleNamespace

    try:
        options = inference_options_from_controls(*audio_controls)
    except ValueError as error:
        yield str(error), gr.update(value=None)
        return
    paths = [SimpleNamespace(name=audio_path)] if audio_path else []
    results = vc.vc_multi(sid, folder, output_folder, paths, pitch, method,
                          index_path, detected_index, index_ratio, filter_radius,
                          resample, volume_mix, protect, output_format, return_audio=True,
                          inference_options=options, return_export=True)
    last_audio = None
    while True:
        with voice_lock:
            try:
                message, audio = next(results)
            except StopIteration:
                return
            if audio is not last_audio:
                try:
                    output = named_inference_audio(audio, loaded_voice)
                except (ValueError, OSError, RuntimeError) as error:
                    message += "\nPreview failed. " + str(error)
                    output = gr.update(value=None)
                last_audio = audio
            else:
                output = gr.update()
        yield message, output



def refresh_merge_model_choices(model_a=None, model_b=None):
    choices = sorted(voice_files(weight_root), key=str.casefold)
    return tuple({"choices": choices, "value": value if value in choices else None, "__type__": "update"} for value in (model_a, model_b))


def describe_merge_ratio(percent):
    return "**Blend:** Model A **%g%%** + Model B **%g%%**. A value of 100 keeps A; 0 keeps B." % (float(percent), 100 - float(percent))


def inspect_merge_models(model_a, model_b):
    from ilaria_model_merge import inspect_model_pair
    yield from inspect_model_pair(model_a, model_b, now_dir)


def merge_voice_models(model_a, model_b, percent, sample_rate, output_name, info, precision):
    from ilaria_model_merge import merge_models
    selected_rate = "auto" if sample_rate == "Auto (from models)" else sample_rate
    yield from merge_models(model_a, model_b, float(percent) / 100, selected_rate, output_name, info, precision, now_dir)


def download_voice_models(url):
    from ilaria_model_import import download_huggingface_models
    yield from download_huggingface_models(url, now_dir)


def import_voice_models(files):
    from ilaria_model_import import import_model_files
    yield from import_model_files(files, now_dir)



def refresh_export_models(selected=None):
    from ilaria_model_export import export_choices
    models, selected, indexes, index = export_choices(now_dir, selected)
    return gr.update(choices=models, value=selected), gr.update(choices=indexes, value=index)


def select_export_index(model):
    return refresh_export_models(model)[1]


def export_voice_model(model, index):
    from ilaria_model_export import export_model
    yield from export_model(model, index, now_dir)


def update_training_controls(version, vocoder, rate, pitch, embedder, target, cache,
                             precision, pretrain, generator, discriminator, method, preview):
    from ilaria_training_controls import control_updates
    return control_updates(now_dir, version, vocoder, rate, pitch, embedder, target,
        cache, precision, pretrain, generator, discriminator, method, preview,
        cuda=torch.cuda.is_available(), no_value=i18n("否"), gpu_pitch=F0GPUVisible)






def recommended_pretrained_info(label):
    from ilaria_pretrain_catalog import describe
    return describe(label)


def download_recommended_pretrained(label):
    from ilaria_pretrain_catalog import download
    yield from download(label, now_dir)


def download_training_pretrained(url, discriminator_url=""):
    from ilaria_pretrain_download import download_pretrained
    yield from download_pretrained(url, now_dir, discriminator_url)


def refresh_model_info_choices(selected=None):
    from ilaria_model_info import list_model_choices
    choices = list_model_choices(now_dir)
    return {"choices": choices, "value": selected if selected in choices else None, "__type__": "update"}


def inspect_voice_model(name):
    from ilaria_model_info import inspect_model
    yield from inspect_model(name, now_dir)


def persist_training_precision(config_path, template, precision, preview_config=None, training_options=None):
    """Set this experiment's precision without changing shared templates."""
    import copy
    import tempfile
    if precision not in ("FP16", "FP32"):
        raise ValueError("Choose FP16 or FP32.")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = copy.deepcopy(template)
    data["train"]["fp16_run"] = precision == "FP16"
    if training_options is not None:
        data["train"].update(training_options)
    if preview_config is not None:
        data["ilaria_preview"] = preview_config
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(config_path), suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=4, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, config_path)
        return data
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def _click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    precision="FP32",
    preview_config=None,
    notification_options=None,
    training_options=None,
):
    if not str(exp_dir1 or "").strip():
        yield "Enter a Model Name in Training before starting this step."
        return
    if precision not in ("FP16", "FP32"):
        yield "Choose FP16 or FP32 in Training Settings & Other."
        return
    if precision == "FP16" and not torch.cuda.is_available():
        yield "FP16 requires a CUDA GPU. Select FP32 in Training Settings & Other."
        return
    options = training_options or {"vocoder":"HiFi-GAN", "embedder_model":"contentvec", "silent_training_files":2, "training_target":"Epochs", "total_steps":10000, "index_algorithm":"Auto"}
    global f0_dir, f0nsf_dir
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    try:
        check_saved_training_sample_rate(exp_dir, sr2)
    except ValueError as error:
        yield "Training settings error: " + str(error)
        return
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    required = [gt_wavs_dir, feature_dir]
    if if_f0_3:
        required.extend([os.path.join(exp_dir, "2a_f0"), os.path.join(exp_dir, "2b-f0nsf")])
    if any(not os.path.isdir(folder) for folder in required):
        yield "Error: Run Process Data and Feature Extraction first."
        return
    if if_f0_3:
        f0_dir = "%s/2a_f0" % exp_dir
        f0nsf_dir = "%s/2b-f0nsf" % exp_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    if not names:
        yield "Error: Run Process Data and Feature Extraction first."
        return
    opt = []
    for name in sorted(names):
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(int(options["silent_training_files"])):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy"
                "|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(int(options["silent_training_files"])):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    if options["embedder_model"] == "spin-v2":
        opt = [line.replace("/logs/mute/3_feature", "/logs/mute_spin-v2/3_feature") for line in opt]
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isfile(os.path.join(exp_dir, "training-state", "latest.json")):
        try:
            with open(filelist_path, "r", encoding="utf-8") as stream:
                previous_files = stream.read().splitlines()
        except OSError:
            yield "The resumable checkpoint is missing its filelist.txt. Restore the original experiment or use a new Model Name."
            return
        if sorted(previous_files) != sorted(opt):
            yield "The prepared dataset or Speaker ID changed since this checkpoint. Restore the original settings and data, or use a new Model Name for fresh training."
            return
    else:
        shuffle(opt)
        with open(filelist_path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(opt))
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    try:
        persist_training_precision(config_save_path, config.json_config[config_path], precision, preview_config, options)
    except (OSError, ValueError, KeyError, TypeError) as error:
        yield "Cannot update training precision: " + str(error)
        return
    cmd = [config.python_cmd, "infer/modules/train/train.py", "-e", str(exp_dir1),
           "-sr", str(sr2), "-f0", str(int(bool(if_f0_3))), "-bs", str(int(batch_size12)),
           "-te", str(int(total_epoch11)), "-se", str(int(save_epoch10)),
           "-l", str(int(if_save_latest13 == i18n("是"))),
           "-c", str(int(if_cache_gpu17 == i18n("是"))),
           "-sw", str(int(if_save_every_weights18 == i18n("是"))), "-v", str(version19)]
    if gpus16:
        cmd.extend(["-g", str(gpus16)])
    if pretrained_G14:
        cmd.extend(["-pg", str(pretrained_G14)])
    if pretrained_D15:
        cmd.extend(["-pd", str(pretrained_D15)])
    from subprocess import STDOUT
    log_path = os.path.join(exp_dir, "train.log")
    log_offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    # Keep verbose diagnostics on disk; the monitor prints only epoch summaries.
    with open(os.path.join(exp_dir, "training_console.log"), "ab", buffering=0) as console_log:
        try:
            p = Popen(cmd, shell=False, cwd=now_dir, stdout=console_log, stderr=STDOUT)
        except OSError:
            if notification_options:
                try:
                    from ilaria_training_notifications import notify_training_event
                    register_notification_worker(notify_training_event(
                        now_dir, exp_dir1, event="error", **notification_options))
                except Exception:
                    pass
            raise
        training_session.attach(p)
        if notification_options:
            try:
                from ilaria_training_notification_monitor import start_notification_monitor
                start_notification_monitor(p, log_path, int(total_epoch11), log_offset,
                    now_dir, exp_dir1, gate=update_operation_gate, total_steps=options["total_steps"] if options["training_target"] == "Steps" else None, **notification_options)
            except Exception:
                logger.debug("Optional training notification monitor could not start", exc_info=True)
        yield from stream_training_progress(p, log_path, int(total_epoch11), log_offset, total_steps=options["total_steps"] if options["training_target"] == "Steps" else None)


from ilaria_training_tools import TrainingSession
training_session = TrainingSession(now_dir)



def register_notification_worker(result):
    worker = result.get("_process") if isinstance(result, dict) else None
    if worker is not None:
        with update_operation_gate.lock:
            update_operation_gate.children.append(worker)


def test_training_alerts(desktop=True, sound=True):
    from ilaria_training_notifications import test_notifications
    result = test_notifications(now_dir, desktop=bool(desktop), sound=bool(sound))
    register_notification_worker(result)
    return result.get("status", "Notification test requested.")


def refresh_dataset_choices():
    from ilaria_dataset_upload import list_datasets
    return gr.update(choices=list_datasets(now_dir), value=None)


def delete_training_dataset(selected, current_path):
    from ilaria_dataset_upload import remove_dataset, list_datasets
    from pathlib import Path
    with update_operation_gate.lock:
        if update_operation_gate.updating or update_operation_gate.active or any(child.poll() is None for child in update_operation_gate.children):
            return "Wait for current operations to finish before deleting a dataset.", gr.update(), gr.update()
        try:
            target = (Path(now_dir) / "datasets" / str(selected or "")).resolve()
            message = remove_dataset(now_dir, selected)
            current = Path(str(current_path or "datasets"))
            if not current.is_absolute():
                current = Path(now_dir) / current
            current = current.resolve()
            clear_path = current == target or target in current.parents
            return message, gr.update(choices=list_datasets(now_dir), value=None), gr.update(value="datasets") if clear_path else gr.update()
        except (OSError, ValueError) as error:
            return "Dataset removal failed: " + str(error), gr.update(), gr.update()


def import_training_dataset(files, model_name):
    from ilaria_dataset_upload import import_dataset
    for status, path in import_dataset(files, now_dir, model_name):
        yield status, gr.update(value=path) if path else gr.update()


def check_training_dataset(dataset, silence_db=-45, silence_seconds=2, volume_tolerance=6):
    from ilaria_dataset_checker import check_dataset
    yield from check_dataset(dataset, silence_db=float(silence_db),
                             min_silence_seconds=float(silence_seconds),
                             volume_tolerance_db=float(volume_tolerance), root=now_dir)


def stop_and_save_training():
    try:
        return training_session.request_stop()
    except (OSError, ValueError) as error:
        return "Cannot request a stop: " + str(error)


def read_training_preview(model_name, selected=None):
    from ilaria_training_tools import experiment_path
    from ilaria_training_preview import latest_preview
    try:
        folder = experiment_path(now_dir, model_name)
        status, audio, choices = latest_preview(folder, selected)
        return status, audio, gr.update(choices=choices, value=selected if selected in choices else (choices[-1] if choices else None))
    except (OSError, ValueError, TypeError) as error:
        return "Preview unavailable: " + str(error), None, gr.update(choices=[], value=None)


def select_training_preview(model_name, selected):
    status, audio, _ = read_training_preview(model_name, selected)
    return status, audio


def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11,
                batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
                gpus16, if_cache_gpu17, if_save_every_weights18, version19,
                precision="FP32", preview_enabled=False, preview_source=None,
                preview_every=10, preview_pitch=0, preview_method="rmvpe",
                notification_desktop=True, notification_sound=True, dataset_path="datasets",
                vocoder="HiFi-GAN", embedder_model="contentvec", silent_files=2,
                training_target="Epochs", total_steps=10000, index_algorithm="Auto",
                pretrain_selection="Auto (official pretrained for current settings)"):
    if globals().get("training_queue") is not None and training_queue.blocks_manual():
        yield "Error: The training queue is running. Pause it before starting a manual step.", gr.update(), gr.update(), gr.update()
        return
    error = training_dataset_error(dataset_path)
    if error:
        yield error, gr.update(), gr.update(), gr.update()
        return
    from ilaria_training_preview import prepare_preview
    import uuid
    from ilaria_training_options import validate_options
    try:
        from ilaria_pretrain_selector import resolve_training_pretrain
        pretrained_G14, pretrained_D15 = resolve_training_pretrain(pretrain_selection,
            sr2, if_f0_3, version19, now_dir, pretrained_G14, pretrained_D15)
        options = validate_options(now_dir, exp_dir1, sr2, if_f0_3, version19, vocoder,
            embedder_model, silent_files, training_target, total_steps, index_algorithm,
            pretrain_selection, pretrained_G14, pretrained_D15, if_cache_gpu17 == i18n("是"))
    except (OSError, ValueError, TypeError) as error:
        yield "Training settings error: " + str(error), gr.update(), gr.update(), gr.update()
        return
    notification_options = {"desktop": bool(notification_desktop), "sound": bool(notification_sound),
                            "run_id": uuid.uuid4().hex}
    try:
        with training_session.start(exp_dir1) as job:
            try:
                settings = prepare_preview(now_dir, job["folder"], preview_enabled,
                                           preview_source, preview_every, preview_pitch, preview_method)
            except Exception as error:
                raise ValueError("Check Training previews: " + str(error)) from error
            settings["speaker"] = int(spk_id5)
            last_audio = object()
            for text in _click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10,
                    total_epoch11, batch_size12, if_save_latest13, pretrained_G14,
                    pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18,
                    version19, precision, settings, notification_options, training_options=options):
                status, audio, history = read_training_preview(job["name"])
                if audio == last_audio:
                    audio = gr.update()
                    history = gr.update()
                else:
                    last_audio = audio
                yield text, status, audio, history
    except (OSError, ValueError, RuntimeError) as error:
        yield "Training could not start or continue: " + str(error), gr.update(), gr.update(), gr.update()


def train_index(exp_dir1, version19, dataset_path="datasets", index_algorithm="Auto"):
    if globals().get("training_queue") is not None and training_queue.blocks_manual():
        yield "Error: The training queue is running. Pause it before starting a manual step."
        return
    error = training_dataset_error(dataset_path)
    if error:
        yield error
        return
    if not str(exp_dir1 or "").strip():
        yield "Enter a Model Name in Training before starting this step."
        return
    if exp_dir1 in (".", "..") or any(char in exp_dir1 for char in '/\\:*?"<>|'):
        yield "Enter a Model Name without path separators or special filename characters."
        return
    exp_dir = "logs/%s" % exp_dir1
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % exp_dir
        if version19 == "v1"
        else "%s/3_feature768" % exp_dir
    )
    if not os.path.exists(feature_dir):
        yield "Error: Run 2. Feature Extraction first."
        return
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        yield "Error: Run 2. Feature Extraction first."
        return
    yield "Train Index: loading audio features... | 0.0%"
    npys = []
    for file_number, name in enumerate(sorted(listdir_res), 1):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
        if file_number % 20 == 0 or file_number == len(listdir_res):
            yield f"Train Index: loading {name} | {20 * file_number / len(listdir_res):.1f}%"
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    from ilaria_index_algorithm import prepare_index_features, ivf_list_count
    try:
        big_npy = yield from prepare_index_features(big_npy, index_algorithm, config.n_cpu)
    except (ValueError, MemoryError) as error:
        yield "Train Index error: " + str(error)
        return

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = ivf_list_count(big_npy.shape[0])
    yield "Train Index: preparing index... | 40.0%"
    import faiss

    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    yield "Train Index: training index... | 40.0%"
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    yield "Train Index: adding audio features... | 60.0%"
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
        yield f"Train Index: adding audio features... | {60 + 35 * min(i + batch_size_add, len(big_npy)) / len(big_npy):.1f}%"
    import tempfile

    # Keep inference-ready indexes alongside the voice library, grouped by model.
    destination = pathlib.Path(now_dir) / "models" / exp_dir1
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / (
        f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    )
    fd, temporary = tempfile.mkstemp(prefix=".index-", suffix=".partial", dir=destination)
    os.close(fd)
    try:
        faiss.write_index(index, temporary)
        os.replace(temporary, index_path)
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)
    yield "Train Index done! | 100.0%"


def run_queued_training(job):
    args = job["args"]
    prep = job["preparation"]
    model, rate, pitch_guidance, version, precision, dataset = args[0], args[1], args[2], args[13], args[14], args[22]
    model_folder = pathlib.Path(now_dir) / "logs" / model
    resume = ((model_folder / "training-state" / "latest.json").is_file()
              or (any(model_folder.glob("G_*.pth")) and any(model_folder.glob("D_*.pth"))))
    if job["auto_prepare"] and not resume:
        stages = [
            ("1. Process Data", preprocess_dataset(dataset, model, rate, prep["cpu"]), "Pre-Process done!"),
            ("2. Feature Extraction", extract_f0_feature(prep["feature_gpus"], prep["cpu"], prep["pitch_method"], pitch_guidance, model, version, prep["pitch_gpus"], precision, dataset, args[24]), "Feature Extraction done!"),
            ("3. Train Index", train_index(model, version, dataset, args[28]), "Train Index done!"),
        ]
        for title, operation, success in stages:
            last = ""
            try:
                for status in operation:
                    last = str(status)
                    yield model + " | " + title + "\n" + last
            finally:
                operation.close()
            if not last.startswith(success):
                raise ValueError(title + ": " + (last or "No successful completion was reported."))
    elif resume:
        yield model + " | Resuming saved training; prepared data is kept."
    last = ""
    operation = click_train(*args)
    try:
        for result in operation:
            last = str(result[0])
            yield model + " | 4. Train Pth\n" + last
    finally:
        operation.close()
    if last.startswith("DONE! Training completed"):
        return "Completed"
    if last.startswith("STOPPED! Training saved"):
        return "Stopped"
    raise ValueError(last or "Training did not confirm completion.")


from ilaria_training_queue import TrainingQueue
training_queue = TrainingQueue(now_dir, run_queued_training, update_operation_gate)


def refresh_training_queue(selected=None):
    table, log, choices, selection = training_queue.display(selected)
    return table, log, gr.update(choices=choices, value=selection)


def add_training_to_queue(*values):
    args = list(values[:30])
    cpu, feature_gpus, pitch_gpus, pitch_method, automatic = values[30:]
    with training_session.lock:
        training_session._prune()
        if training_session.job and training_session.job["name"].casefold() == str(args[0] or "").strip().casefold():
            yield "This Model Name is already training. Use a different name for the next job."
            return
    yield "Saving this job's dataset and settings. Please wait..."
    try:
        from ilaria_pretrain_selector import resolve_training_pretrain, FROZEN, NONE, AUTO
        if args[23] == "RefineGAN" and args[29] == AUTO:
            raise ValueError("Select a compatible RefineGAN pretrained pair or None before adding the job.")
        args[8], args[9] = resolve_training_pretrain(args[29], args[1], args[2], args[13], now_dir, args[8], args[9])
        args[29] = FROZEN if args[29] != NONE else NONE
        training_queue.add(args, {"cpu": int(cpu), "feature_gpus": feature_gpus,
                                 "pitch_gpus": pitch_gpus, "pitch_method": pitch_method}, automatic)
        yield training_queue.message
    except (OSError, ValueError, TypeError) as error:
        yield "Cannot add training job: " + str(error)


def show_selected_queue_job(selected):
    table, log, _ = refresh_training_queue(selected)
    return table, log


def start_training_queue():
    with update_operation_gate.lock:
        update_operation_gate.children = [p for p in update_operation_gate.children if p.poll() is None]
        if training_queue.is_running():
            return "The queue is already running."
        if update_operation_gate.updating or update_operation_gate.active or update_operation_gate.children:
            return "Wait for current operations to finish before starting the queue."
        try:
            return training_queue.start()
        except (OSError, ValueError, RuntimeError) as error:
            return "Queue could not start: " + str(error)


def pause_training_queue():
    return training_queue.pause()


def remove_queued_training(selected):
    try:
        return training_queue.change(str(selected or "").split(" | ", 1)[0], "remove")
    except (OSError, ValueError) as error:
        return str(error)


def retry_queued_training(selected):
    try:
        return training_queue.change(str(selected or "").split(" | ", 1)[0], "retry")
    except (OSError, ValueError) as error:
        return str(error)


F0GPUVisible = config.dml is False

def change_f0_method(f0method8, if_f0_3=True):
    return {
        "visible": bool(if_f0_3 and f0method8 == "rmvpe_gpu" and F0GPUVisible),
        "__type__": "update",
    }

UI_ICON_PATH = pathlib.Path(__file__).resolve().parent / "assets" / "ilaria_rvc_icon_transparent.png"

ILARIA_HEADER = """
<section class="ilaria-hero" aria-label="ILARIA RVC">
  <div class="ilaria-hero-copy">
    <h1>♡ ILARIA RVC ♡</h1>
    <p class="ilaria-description">RVC V2 Training &amp; Inference provided by Ilaria!</p>
    <div class="ilaria-support">
      <a class="ilaria-shop" href="https://ko-fi.com/ilariaowo" target="_blank" rel="noopener noreferrer">
        💖 Ilaria's Ko-Fi Shop <span aria-hidden="true">&#8599;&#xFE0E;</span>
      </a>
    </div>
  </div>
  <div class="ilaria-portrait">
    <img src="/file=assets/ilaria_rvc_icon_transparent.png" alt="Ilaria's character illustration" width="955" height="955">
  </div>
</section>
"""

UI_FAVICON_JS = """
() => {
    let icon = document.querySelector("link[rel~='icon']");
    if (!icon) {
        icon = document.createElement("link");
        icon.rel = "icon";
        document.head.appendChild(icon);
    }
    icon.type = "image/png";
    icon.href = "favicon.ico?v=ilaria-rvc-ashi-transparent-2";
    return [];
}
"""

WAVEFORM_JS = r"""
() => {
    const host = document.querySelector("gradio-app");
    const root = host?.shadowRoot || host || document;
    if (root.ilariaWaveformObserver) return [];

    const players = new Map();
    const maxBytes = 64 * 1024 * 1024;
    const maxSeconds = 30 * 60;
    const barCount = 150;
    const width = 600;
    let nextId = 0;
    let scheduled = false;

    function element(tag, className) {
        const node = document.createElement(tag);
        node.className = className;
        return node;
    }

    function svg(tag, attrs) {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
        return node;
    }

    function timestamp(seconds) {
        const total = Math.max(0, Math.floor(seconds));
        return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
    }

    async function readAudio(url, signal) {
        const response = await fetch(url, {signal});
        if (!response.ok || Number(response.headers.get("content-length")) > maxBytes || !response.body)
            return null;
        const reader = response.body.getReader();
        const parts = [];
        let size = 0;
        try {
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                size += value.byteLength;
                if (size > maxBytes) {
                    await reader.cancel();
                    return null;
                }
                parts.push(value);
            }
        } finally {
            reader.releaseLock();
        }
        const data = new Uint8Array(size);
        let offset = 0;
        for (const part of parts) {
            data.set(part, offset);
            offset += part.byteLength;
        }
        return data.buffer;
    }

    function makePeaks(buffer) {
        const peaks = new Float32Array(barCount);
        for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
            const samples = buffer.getChannelData(channel);
            for (let i = 0; i < barCount; i++) {
                const start = Math.floor(i * samples.length / barCount);
                const end = Math.floor((i + 1) * samples.length / barCount);
                const stride = Math.max(1, Math.floor((end - start) / 768));
                for (let j = start; j < end; j += stride)
                    peaks[i] = Math.max(peaks[i], Math.abs(samples[j]));
            }
        }
        return peaks;
    }

    function draw(audio, peaks) {
        const panel = element("div", "ilaria-waveform");
        const track = element("div", "ilaria-waveform-track");
        const time = element("div", "ilaria-waveform-time");
        const elapsed = document.createElement("span");
        const duration = document.createElement("span");
        const play = element("button", "ilaria-waveform-play");
        play.type = "button";
        const save = element("a", "ilaria-waveform-save");
        save.href = audio.currentSrc;
        save.download = "";
        if (audio.closest("#single-inference-output, #batch-inference-output")) {
            try {
                save.download = decodeURIComponent(new URL(audio.currentSrc).pathname).split(/[/\\]/).pop();
            } catch (_) {}
        }
        save.textContent = "Download";
        const seek = element("input", "ilaria-waveform-seek");
        seek.type = "range";
        seek.min = "0";
        seek.max = String(audio.duration);
        seek.step = "any";
        seek.setAttribute("aria-label", "Seek in audio waveform");
        seek.title = "Seek in audio";

        const plot = svg("svg", {viewBox: "0 0 600 64", preserveAspectRatio: "none", "aria-hidden": "true"});
        const id = "ilaria-waveform-progress-" + ++nextId;
        const mask = svg("clipPath", {id});
        const played = svg("rect", {x: 0, y: 0, width: 0, height: 64});
        mask.append(played);
        const defs = svg("defs", {});
        defs.append(mask);
        const peak = Math.max(0.015, ...peaks);
        const shape = Array.from(peaks, (value, index) => {
            const x = (index + 0.5) * width / barCount;
            const half = Math.max(0.5, 27 * Math.sqrt(value / peak));
            return "M" + x + " " + (32 - half) + "V" + (32 + half);
        }).join(" ");
        const base = svg("path", {d: shape, stroke: "#dc91bc", "stroke-width": 2.1, "stroke-linecap": "round"});
        const progress = svg("path", {d: shape, stroke: "#ffc0cb", "stroke-width": 2.1,
            "stroke-linecap": "round", "clip-path": "url(#" + id + ")"});
        const playhead = svg("line", {y1: 2, y2: 62, stroke: "#ffe3ed", "stroke-width": 1.5});
        plot.append(defs, base, progress, playhead);
        track.append(plot, seek);
        time.append(play, elapsed, duration);
        if (!audio.closest("#tts-source-audio")) time.append(save);
        panel.append(track, time);

        function update() {
            const x = width * Math.min(1, audio.currentTime / audio.duration);
            played.setAttribute("width", x);
            playhead.setAttribute("x1", x);
            playhead.setAttribute("x2", x);
            seek.value = String(audio.currentTime);
            elapsed.textContent = timestamp(audio.currentTime);
            duration.textContent = timestamp(audio.duration);
            seek.setAttribute("aria-valuetext", elapsed.textContent + " of " + duration.textContent);
        }
        seek.addEventListener("input", () => {
            audio.currentTime = Number(seek.value);
            update();
        });
        seek.addEventListener("keydown", event => {
            const offsets = {ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5};
            if (!(event.key in offsets)) return;
            event.preventDefault();
            audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + offsets[event.key]));
            update();
        });
        let frame = 0;
        const playbackEvents = new AbortController();
        const options = {signal: playbackEvents.signal};
        function animate() {
            update();
            frame = audio.paused || audio.ended ? 0 : requestAnimationFrame(animate);
        }
        function playbackChanged() {
            play.textContent = audio.paused ? "Play" : "Pause";
            cancelAnimationFrame(frame);
            animate();
        }
        play.addEventListener("click", () => {
            if (audio.paused) audio.play().catch(() => playbackChanged());
            else audio.pause();
        });
        for (const event of ["play", "pause", "ended"])
            audio.addEventListener(event, playbackChanged, options);
        playbackChanged();
        audio.before(panel);
        audio.classList.add("ilaria-waveform-native");
        return {update, dispose() {
            playbackEvents.abort();
            cancelAnimationFrame(frame);
            audio.classList.remove("ilaria-waveform-native");
            panel.remove();
        }};
    }

    function attach(audio) {
        let url = "";
        let request;
        let view;
        const events = new AbortController();

        function clear() {
            url = "";
            request?.abort();
            view?.dispose();
            view = null;
        }

        async function load() {
            const src = audio.currentSrc || audio.getAttribute("src");
            if (!src || src === url || !Number.isFinite(audio.duration) || audio.duration <= 0) return;
            clear();
            url = src;
            if (audio.duration > maxSeconds) return;
            const pending = new AbortController();
            request = pending;
            try {
                const bytes = await readAudio(src, pending.signal);
                const Decoder = window.OfflineAudioContext || window.webkitOfflineAudioContext;
                if (!bytes || !Decoder || pending.signal.aborted) return;
                const buffer = await new Decoder(1, 1, 24000).decodeAudioData(bytes);
                if (!pending.signal.aborted && audio.isConnected)
                    view = draw(audio, makePeaks(buffer));
            } catch (_) { /* The browser's audio player stays available. */ }
        }

        const options = {signal: events.signal};
        audio.addEventListener("loadedmetadata", load, options);
        audio.addEventListener("emptied", clear, options);
        audio.addEventListener("timeupdate", () => view?.update(), options);
        audio.addEventListener("seeked", () => view?.update(), options);
        players.set(audio, () => {
            clear();
            events.abort();
        });
        load();
    }

    function scan() {
        scheduled = false;
        for (const [audio, dispose] of players) {
            if (!root.contains(audio)) {
                dispose();
                players.delete(audio);
            }
        }
        root.querySelectorAll("audio").forEach(audio => {
            if (!players.has(audio)) attach(audio);
        });
    }

    root.ilariaWaveformObserver = new MutationObserver(() => {
        if (!scheduled) {
            scheduled = true;
            requestAnimationFrame(scan);
        }
    });
    root.ilariaWaveformObserver.observe(root, {childList: true, subtree: true});
    scan();
    return [];
}

"""

ILARIARVC_CSS = """
.gradio-container,
.gradio-container .container {
    background: #53355C !important;
}
.gradio-container {
    color: #FFFFFF;
}
.gradio-container .container {
    width: 100% !important;
    max-width: 1420px !important;
    margin: 0 auto !important;
    padding: 20px 20px 24px !important;
    box-sizing: border-box;
}
.gradio-container #ilaria-header {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}
.ilaria-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 36px;
    min-height: 208px;
    padding: 22px 28px;
    margin-bottom: 4px;
    border: 1px solid #A17BA9;
    border-radius: 22px;
    background: #3D2646;
    box-shadow: 0 12px 28px rgba(31, 15, 38, 0.22);
}
.ilaria-hero-copy {
    flex: 1 1 0;
    min-width: 0;
}
.ilaria-hero h1 {
    margin: 0;
    color: #FFFFFF;
    font-size: clamp(1.8rem, 3.6vw, 3.1rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.045em;
}
.ilaria-hero h1::after {
    content: "";
    display: block;
    width: 48px;
    height: 3px;
    margin-top: 12px;
    border-radius: 2px;
    background: #FFC0CB;
}
.ilaria-description {
    margin: 14px 0 16px;
    color: #F8EFFA;
    font-size: 1rem;
    line-height: 1.6;
}
.ilaria-support {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    color: #F8EFFA;
    font-size: 0.94rem;
}
.ilaria-shop {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    min-height: 42px;
    padding: 10px 16px;
    border: 1px solid #FFC0CB;
    border-radius: 10px;
    color: #58243D !important;
    background: linear-gradient(110deg, #FFC0CB 0%, #D8B4FE 100%);
    font-weight: 700;
    text-decoration: none !important;
    transition: background 150ms ease, border-color 150ms ease;
}
.ilaria-shop:hover {
    background: linear-gradient(110deg, #FFADBE 0%, #C9A4F4 100%);
    border-color: #FFADBE;
}
.ilaria-shop:focus-visible,
.gradio-container .gr-button:focus-visible {
    outline: 2px solid #FFE3E9 !important;
    outline-offset: 3px;
}
.ilaria-portrait {
    flex: 0 0 160px;
    width: 160px;
    height: 160px;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
}
.ilaria-portrait img {
    display: block;
    width: 100%;
    height: 100%;
    border-radius: 0;
    object-fit: contain;
    background: transparent;
}
.gradio-container #ilaria-main-tabs {
    margin: 0 !important;
    gap: 12px;
}
.gradio-container #ilaria-main-tabs > div:first-child {
    gap: 7px;
    padding: 6px;
    border: 1px solid #9972A3 !important;
    border-radius: 12px;
    background: #3D2646;
}
.gradio-container .tabs > div:first-child > button {
    margin: 0 !important;
    padding: 8px 18px;
    border: 0 !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #F8EFFA !important;
    font-weight: 600;
}
.gradio-container .tabs > div:first-child > button:hover {
    background: #62416B !important;
    color: #FFFFFF !important;
}
.gradio-container .tabs > div:first-child > button.bg-white {
    background: linear-gradient(110deg, #FFC0CB 0%, #D8B4FE 100%) !important;
    color: #58243D !important;
}
.gradio-container #single-inference-controls { gap: 14px !important; }
.gradio-container #batch-inference-settings,
.gradio-container #inference-settings,
.gradio-container #training-settings {
    padding: 12px;
    border: 1px solid #9972A3 !important;
    border-radius: 12px;
    background: #3D2646 !important;
    color: #FFFFFF;
}
.gradio-container #batch-inference-settings > div:first-child,
.gradio-container #inference-settings > div:first-child,
.gradio-container #training-settings > div:first-child {
    padding: 2px 0;
    color: #F8EFFA;
    font-weight: 600;
}
.gradio-container #training-settings > div:first-child:hover {
    color: #FFFFFF;
}
.gradio-container .tabitem {
    padding: 8px 0 0 !important;
    border: 0 !important;
}
.gradio-container #ilaria-main-tabs .gr-form,
.gradio-container #ilaria-main-tabs .gr-panel,
.gradio-container #ilaria-main-tabs .gr-box {
    border-color: #866190 !important;
    background: #3E2947 !important;
    color: #FFFFFF !important;
}
.gradio-container #ilaria-main-tabs .gr-box {
    border-radius: 10px;
}
.gradio-container #ilaria-main-tabs .gr-form:has(> .gr-block):not(:has(> .gr-block:not([class~="!hidden"]))) {
    display: none !important;
}
.gradio-container #ilaria-main-tabs input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container #ilaria-main-tabs textarea.gr-box,
.gradio-container #ilaria-main-tabs select {
    color: #FFFFFF !important;
    background: #2E1D36 !important;
    border-color: #866190 !important;
}
.gradio-container #ilaria-main-tabs input::placeholder,
.gradio-container #ilaria-main-tabs textarea::placeholder {
    color: #EAD9EF !important;
    opacity: 1;
}
.gradio-container #ilaria-main-tabs input:focus,
.gradio-container #ilaria-main-tabs textarea:focus,
.gradio-container #ilaria-main-tabs select:focus {
    border-color: #FFC0CB !important;
    box-shadow: none !important;
}
.gradio-container #ilaria-main-tabs .gr-input-label {
    color: #FFFFFF !important;
    background: #2E1D36 !important;
    border-color: #866190 !important;
}
.gradio-container #ilaria-main-tabs .text-gray-500,
.gradio-container #ilaria-main-tabs .text-gray-700,
.gradio-container #ilaria-main-tabs legend {
    color: #F8EFFA !important;
}
.gradio-container #ilaria-main-tabs input[type="range"] {
    accent-color: #FFC0CB;
}
.gradio-container #ilaria-main-tabs input[type="checkbox"],
.gradio-container #ilaria-main-tabs input[type="radio"] {
    accent-color: #FFC0CB;
    border-color: #B38ABD !important;
    background-color: #2E1D36 !important;
}
.gradio-container #ilaria-main-tabs input[type="radio"]:checked {
    border-color: #FFC0CB !important;
    background-color: #FFC0CB !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='8' cy='8' r='3.25' fill='%2353355C'/%3E%3C/svg%3E") !important;
}
.gradio-container #ilaria-main-tabs input[type="checkbox"]:checked {
    border-color: #FFC0CB !important;
    background-color: #FFC0CB !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M3.5 8.4 6.6 11.3 12.6 4.8' fill='none' stroke='%2353355C' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") !important;
}
.gradio-container .gr-prose,
.gradio-container .gr-prose h1,
.gradio-container .gr-prose h2,
.gradio-container .gr-prose h3 {
    color: #FFFFFF !important;
}
.gradio-container .gr-prose a {
    color: #FFC0CB !important;
}
.gradio-container .gr-button {
    align-self: center;
    min-height: 44px;
    padding: 12px 18px;
    border: 1px solid #FFC0CB !important;
    border-radius: 10px;
    background: linear-gradient(110deg, #FFC0CB 0%, #D8B4FE 100%) !important;
    color: #58243D !important;
    font-weight: 600;
    transition: background 150ms ease, border-color 150ms ease;
}
.gradio-container .gr-button:hover {
    background: linear-gradient(110deg, #FFADBE 0%, #C9A4F4 100%) !important;
    border-color: #FFADBE !important;
}
.gradio-container footer {
    display: none !important;
}
.gradio-container {
    font-size: 15px;
    line-height: 1.5;
}
.gradio-container #ilaria-main-tabs .gr-box,
.gradio-container #ilaria-main-tabs input,
.gradio-container #ilaria-main-tabs textarea,
.gradio-container #ilaria-main-tabs select,
.gradio-container #ilaria-main-tabs label,
.gradio-container #ilaria-main-tabs legend,
.gradio-container #ilaria-main-tabs .gr-input-label,
.gradio-container #ilaria-main-tabs .text-gray-500,
.gradio-container #ilaria-main-tabs .text-gray-700 {
    font-size: 15px !important;
    line-height: 1.5 !important;
}
.gradio-container .tabs > div:first-child > button {
    font-size: 16px !important;
}
.gradio-container #ilaria-main-tabs input[type="number"] {
    min-height: 40px;
    padding: 6px 10px;
}
.gradio-container #training-settings > div:first-child {
    font-size: 16px;
}
.ilaria-description {
    font-size: 1rem;
}
.ilaria-shop,
.gradio-container .gr-button {
    font-size: 16px !important;
    line-height: 1.4;
}
.gradio-container #training-workflow {
    gap: 14px !important;
}
.gradio-container #training-workflow > .row {
    gap: 14px !important;
    align-items: stretch !important;
}
.gradio-container #training-workflow > .row > .col:first-child {
    justify-content: center;
    gap: 12px !important;
}
.gradio-container #training-workflow .gr-button {
    width: 100%;
    align-self: stretch;
    min-height: 52px;
    margin: 0;
}
.gradio-container #training-workflow .gr-form {
    flex: 1 1 auto;
}
.gradio-container #training-workflow .gr-block {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    min-height: 90px;
}
.gradio-container #training-workflow .gr-block > label {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
}
.gradio-container #training-workflow textarea {
    flex: 1 1 auto;
    min-height: 40px;
    padding: 8px 10px;
}
.gradio-container #ilaria-main-tabs > .tabitem > .col {
    gap: 12px !important;
}
.gradio-container #training-step .gr-block {
    min-height: 114px;
}
@media (max-width: 720px) {
    .gradio-container .container { padding: 16px 14px 28px !important; }
    .ilaria-hero { gap: 20px; min-height: 0; padding: 20px 20px; }
    .ilaria-hero h1 { font-size: 2rem; }
    .ilaria-portrait { flex-basis: 140px; width: 140px; height: 140px; }
}
@media (max-width: 520px) {
    .ilaria-hero { flex-direction: column-reverse; align-items: flex-start; }
    .ilaria-portrait { flex: 0 0 auto; width: 132px; height: 132px; }
    .ilaria-support { align-items: flex-start; flex-direction: column; }
}
"""

TUTORIAL_INTRO = "# ILARIA RVC Glossary\n\n**New here? Start with section 1.** Support modules are stored in the `ilaria` folder; keep that folder intact. Start the app using the main `ILARIA_RVC.bat`. This guide explains the controls in this version of ILARIA RVC, from your first conversion to training your own voice model. You can leave most settings at their defaults. Open a section below whenever you need it.\n\n**Training** creates a voice model from recordings. **Inference** uses an existing model to convert another recording. Converting a file does not train or change the model. Opening this glossary does not start any processing.\n\n**About Ilaria's presets.** The default values in this app are Ilaria's personal settings for Ilaria's PC (NVIDIA GeForce RTX 4070) and usual datasets: approximately **35 minutes of very clean speech**, with varied intonation, laughter, and sighs. They are a starting point, not universal optimal settings or a requirement to collect exactly 35 minutes.\n\n**Adapt the settings to your own setup and taste.** GPU VRAM affects the batch size and whether GPU caching fits. GPU/CPU speed, storage speed, the number and length of audio chunks, sample rate, and other running programs affect processing time. Dataset cleanliness, voice consistency, pitch range, and variety affect what the model can learn. The best epoch count and inference settings depend on the dataset and the result you enjoy. Compare short listening tests, change one setting at a time, and lower memory-heavy settings if you run out of VRAM. More epochs, a larger batch, a higher sample rate, or a bigger index ratio do not guarantee a better voice.\n"

TUTORIAL_SECTIONS = (
    ('1. Your first conversion \u2014 start here',
     "1. Open **Inference** and choose the voice you want in **Voice Model**. The first model in alphabetical order is selected automatically; it may not be the one you intended to use.\n2. Open **Single Inference**. Drag an audio file into **Audio to convert**, or click the upload area to choose it.\n3. Leave **Pitch** at **0** for your first test. This keeps the source pitch unchanged while the model changes the voice character.\n4. You can leave **Inference Settings** closed. Its defaults include **rmvpe**, **Index Ratio 0**, **Resampling 0**, volume mix **0.25**, and protection **0.33**.\n5. Click **Conversion** once. Wait for the result, read **Output Log**, then listen to **Output Audio**. Use the audio player's download control to save the result.\n6. If you want a different result, change one setting and convert again. Start with a short clip so comparisons are quick.\n\nFor a song, convert the isolated vocal recording. Feeding an entire song with drums and instruments into a voice converter can produce unwanted sounds. This interface does not automatically separate a song into vocals and accompaniment.\n\n**If you have no models:** place an inference `.pth` model in `models` inside your ILARIA RVC folder, then click **Refresh**. If it has an accompanying `.index`, keep that file in a matching folder under `models`, for example `models/myvoice/added_..._myvoice_v2.index` for `myvoice.pth`."),
    ('2. Your first training \u2014 follow the four buttons',
     '1. Prepare recordings of **one voice** in a dedicated dataset folder. Use clear recordings with little background noise, music, echo, clipping, or overlapping speakers. WAV is a straightforward input format; converting a poor recording to WAV does not repair it.\n2. In **Training**, click the empty **Model Name** box and type a unique name, such as `myvoice`. The faint `ilaria` example is a placeholder, not an entered name. Use a simple name without slashes. This name identifies the experiment folder and the exported model.\n3. Open **Training Settings & Other**. Set **Path to Dataset** to your recordings folder. The default `datasets` means the `datasets` folder inside the application folder. Put audio directly in that folder; the preprocessor does not recursively search subfolders.\n4. For a first attempt, keep **Sample Rate 32k**, **Pitch Guidance True**, **CPU Threads 8**, **Speaker ID 0**, **Pitch Extraction rmvpe**, **Total Epochs 300**, and **Batch Size 8**. These are starting values, not a promise that every dataset or GPU will need the same settings.\n5. Click **1. Process Data**. Wait for **Pre-Process done! | 100.0%** before continuing.\n6. Click **2. Feature Extraction**. Wait for **Feature Extraction done! | 100.0%**. This creates pitch data when enabled, then the audio features needed by training and the index.\n7. Click **3. Train Index**. Wait for **Train Index done! | 100.0%**. This creates the optional retrieval file; it does not train the `.pth` voice model.\n8. Click **4. Train Pth**. Wait for **DONE!** and the actual completion time in **Training Log**. Reaching the last epoch can still be followed by a final save.\n9. Open **Inference**, click **Refresh**, and select your newly exported model. Upload a short test recording and convert it.\n\nRun one operation at a time and wait for it to finish. Closing a browser tab does not reliably stop a running backend job. Keep the CMD window open while the application is working.\n\nThe index and `.pth` model are separate products of the same extracted features. Training the index first follows the order shown in this UI; it does not make the neural network learn faster or change its training loss.'),
    ('3. Model Name, dataset paths, sample rate, and speaker ID',
     '**Model Name \u2014 required; starts empty.** The faint `ilaria` text is a **placeholder**, which means an example displayed inside an empty box. It is not an entered name, is not submitted as a name, and will not automatically name your model. Click the box and type your own name; you do not need to select or delete the example first. If the box is empty again, the example can reappear.\n\n**Exactly what to type:** a simple experiment name such as `myvoice` or `myvoice_v2_test`. Type the name only: do not paste a folder path, and do not add `.pth`. Avoid slashes, backslashes, and Windows filename characters such as `: * ? " < > |`. A short name using letters, numbers, underscores, or hyphens is easy to recognize.\n\n**Example:** typing `myvoice` makes the experiment folder `logs/myvoice` and, after successful training, the inference model `models/myvoice.pth`. Typing a name does not create a trained voice by itself; you must still complete the four training steps.\n\n**Keep the same name through one experiment.** Process Data, Feature Extraction, Train Index, and Train Pth all use this field to locate that experiment. If you process data under `myvoice` and then change the field to `another_voice`, the next step looks in the other experiment folder. It does not move or rename the files you prepared earlier.\n\n**If you leave it blank:** all four training actions stop and display `Enter a Model Name in Training before starting this step.` Type the intended name and click the appropriate step again. Spaces alone do not count as a name.\n\n**New experiment versus continuing an old one:** use a unique new name for a fresh dataset or new configuration. To continue an existing experiment, enter its exact original name and keep compatible settings. Reusing a name can reuse old features, configuration, and saved G/D checkpoints; it is not a guaranteed fresh start. Changing Model Name does not change which voice is selected in Inference.\n\n**Path to Dataset \u2014 default `datasets`.** This is a folder containing the source audio, not the path to a single recording. Example: `C:/VoiceData/myvoice`. Relative paths are resolved from the application folder. Paste an actual path rather than a placeholder such as `%userprofile%`; not every field expands Windows environment variables. Keep unrelated files and subfolders out of the input folder.\n\n**Sample Rate \u2014 default `32k`; choices `32k`, `40k`, `48k`.** These mean 32,000, 40,000, or 48,000 audio samples per second for the trained model. They are not bitrates or training speed controls. A higher value does not recover detail missing from your recordings. Keep the sample rate consistent with the selected pretrained files. If you change it after processing, start a new experiment and repeat the preparation steps rather than mixing old and new data.\n\n**Pitch Guidance \u2014 default `True`.** This trains a model that uses an extracted pitch contour, also called F0. Leave it enabled for your first voice or singing model. `False` trains a model without that pitch input, skips F0 extraction, and changes which pretrained files are needed. Treat this as a choice made before preparing the experiment, not a switch to flip halfway through training.\n\n**Speaker ID \u2014 default `0`, range `0\u20134` in training.** This tags the training examples with a speaker number. For the normal one-voice workflow, leave it at `0`. Changing it does not select a GPU or automatically separate multiple speakers. Proper multi-speaker preparation needs additional planning outside this basic workflow.\n\n**Version.** This customized training interface uses RVC **V2**. You do not need to find a version switch.\n\n**Training Precision \u2014 default FP32; choices FP16 and FP32.** Open Training Settings & Other to choose. FP32 uses full-precision training. FP16 uses CUDA mixed precision: eligible calculations run in half precision, with FP32 sections and gradient scaling retained for stability. It is not a promise that every tensor becomes half precision. FP16 may use less VRAM and be faster on a compatible GPU such as Ilaria\'s RTX 4070; compare measured epoch times and output quality. For GTX 16xx cards, FP32 is the recommended starting point because FP16 can be unstable.\n\nThe choice is written into this experiment\'s config every time Train Pth is started, including when continuing saved training. Changing an old experiment to FP32 therefore does not silently keep its previous FP16 config. The running job keeps the precision with which it started; changing the radio while it runs does not change that job. Wait for the current job to finish before starting another.\n\nThe same choice controls newly computed HuBERT features on CUDA and CUDA RMVPE extraction. CPU/MPS/DirectML feature extraction uses FP32, and CPU RMVPE remains FP32. Previously extracted files are reused; changing this control does not retroactively regenerate them. For a full comparison, use a new experiment name and repeat preparation. Switching only the PTH training precision does not require regenerating valid features.\n\nFP16 training requires CUDA in this UI. On a CPU setup choose FP32. FP32 normally uses more VRAM: lower Batch Size if you get CUDA out of memory. Training Precision does not change the Inference settings or the Other > Merge models Output precision control. The normal final training export still uses the existing inference-export format (half-precision weights); FP32 training does not mean the exported .pth is a full-precision or resumable training checkpoint.'),
    ('4. CPU Threads and every training pitch extractor',
     "**CPU Threads \u2014 default `8`, range `1\u201312`.** This controls worker processes for data preparation and CPU pitch extraction. It is not the neural network's GPU batch size and does not set every CPU thread used by every library. More workers can increase CPU/RAM use and are not always faster. Reduce the value if preparation overloads your computer.\n\n**Pitch Extraction \u2014 default `rmvpe`.** The extractor measures how the recorded pitch changes over time. It does not choose the target voice. In this build:\n\n| Option | What it does here |\n| --- | --- |\n| `pm` | Uses the Parselmouth pitch estimator on CPU. An alternative to compare on the same recording. |\n| `harvest` | Uses WORLD Harvest on CPU. Another pitch estimation method; the results and processing time can differ from RMVPE. |\n| `dio` | Uses WORLD DIO on CPU. Another available CPU method. |\n| `rmvpe` | Uses the RMVPE neural pitch extractor **on CPU in the training extraction script**. This is the default. |\n| `rmvpe_gpu` | Runs RMVPE through the separate GPU extraction route. It exposes the RMVPE GPU process field below. |\n\nYou can compare methods by listening to resulting conversions, but keep a single experiment's prepared files consistent. **CREPE is an inference option in this build, not a training option.** **FCPE** (Fast Context-based Pitch Estimation) is available in Training and Inference. It uses CPU for dataset extraction and CUDA for inference when available.\n\n**RMVPE GPU process field.** It appears when Pitch Guidance is enabled and `rmvpe_gpu` is selected on a supported setup. `0` runs one extraction process on GPU 0. `0-0` runs two processes on GPU 0. `0-0-2` runs two on GPU 0 and one on GPU 2. Repeating a number creates more processes; it does not create more physical GPUs and can use more VRAM.\n\n**CPU RMVPE versus GPU RMVPE:** changing this affects the preparation stage. It does not directly reduce the number of neural-network training epochs."),
    ('5. GPU selection, Batch Size, and Cache data to GPU',
     "**GPU Information** shows detected cards and their numerical IDs. The IDs begin at `0`. A computer with one detected NVIDIA GPU normally uses `0`.\n\n**GPU ID's ex: 0-2 Uses GPU's 0 & 2** selects the cards used for audio feature extraction. **Enter cards to be used EX: 0-1** selects cards for `.pth` training. These are separate fields. Use `0` in both for a one-GPU setup. `0-2` means GPU 0 and GPU 2; it does not mean all IDs between them. Only enter IDs that actually exist in GPU Information.\n\n**Batch Size \u2014 default `8`, range `1\u201316`.** This is the number of prepared audio examples processed together per training batch, per GPU in this trainer. Larger batches use more VRAM. Smaller batches usually require more steps to cover the same dataset. Increasing the batch size does not guarantee better quality or faster epochs.\n\nIf you see **CUDA out of memory**, lower Batch Size, for example from `8` to `4`, and leave Cache data to GPU disabled. Unload an inference model if one is occupying VRAM. A failed run is not a completed model.\n\n**Cache data to GPU \u2014 default `No`.** `Yes` stores prepared training batches in GPU memory to reduce repeated data transfer. This can help when the dataset fits, but it needs additional VRAM. The first epoch may include cache setup and take longer. Leave it at `No` if you are unsure or memory is limited.\n\nThere is no universal fastest combination: dataset length, chunk count, GPU memory, batch size, and caching all matter. Compare several epochs after startup using the same dataset and settings."),
    ('6. Epochs, cumulative steps, checkpoints, and pretrained files',
     '**Total Epochs \u2014 default `300`, range `2\u20131000`.** One epoch is one full pass through the prepared training dataset. This is the target epoch number, not an amount of extra epochs to append to an existing run. More epochs take more time and do not automatically improve the voice. Listen to test outputs when deciding whether a model needs further training.\n\n**Total Steps** counts training batches completed across epochs. It is cumulative: if each epoch has 53 batches, the total is 53 after epoch 1, 106 after epoch 2, and 159 after epoch 3. Batch size and the amount of prepared data affect this count. On resume, this trainer reconstructs the starting count from the saved epoch and the current number of batches, so changing the dataset or batch size can change its interpretation.\n\n**Save frequency \u2014 default `100`, range `1\u20131000`.** This saves full training checkpoints at that epoch interval. With the defaults of 300 target epochs and a save frequency of 300, the interval checkpoint is saved at epoch 300 and the inference model is exported at normal completion. Choose a smaller save frequency if you want recovery checkpoints earlier in the run. A long interval means more work can be lost if the run is interrupted before the next checkpoint. This setting is not a timer measured in seconds.\n\n**Save last ckpt as final Model \u2014 default `Yes`.** Despite its label, this controls whether the interval training checkpoints overwrite the same `G_2333333.pth` and `D_2333333.pth` files. `Yes` keeps the latest pair at the fixed filenames. `No` keeps differently numbered interval checkpoints, using more disk space. Both settings still export the final inference model when training finishes normally.\n\n**Create model at save frequency \u2014 default `No`.** `Yes` also exports a smaller inference-ready `.pth` model at each save interval, so you can compare intermediate results. `No` skips those intermediate inference exports; the normal final export still happens. An inference model is different from a full G/D training checkpoint.\n\n**Pretrained model (G + D).** One dropdown selects the starting Generator and Discriminator weights together. The generator learns to produce output; the discriminator supplies training feedback. Auto resolves the original pair for the current sample rate, pitch guidance and RVC version. Installed custom pairs are selected as one entry; Pretrained Log confirms both files. None explicitly trains from scratch. Download pretrained pairs in Other or place both files together under assets/pretrained_custom/<name>, then refresh this dropdown. Finished inference voice models belong in models and are selected in Inference, not here.\n\n**Continuing an interrupted run.** Keep the same Model Name, prepared data and compatible training settings. Managed checkpoints restore cumulative steps, optimizers and the saved epoch/batch position. Existing checkpoints take priority over new pretrain selection; changing this dropdown does not reset a saved model. Use a new Model Name for a fresh model. An exported inference .pth alone does not contain the full resumable state.'),
    ('7. What the four actions and their percentages mean',
     "**1. Process Data** loads the source audio, slices it into shorter training chunks, and writes training-rate audio plus 16 kHz audio. The one-line **Pre-Process Log** shows the current status and the percentage of source files reported as successfully processed. A dataset containing one long file can stay at 0% for a while and then jump when that file finishes. This is a file counter, not an estimate of remaining time.\n\n**2. Feature Extraction** first creates F0 data when Pitch Guidance is enabled, then extracts HuBERT audio features. Its percentage counts saved results matching the prepared WAV files. With pitch enabled, pitch results and HuBERT features each contribute half of the total; they may take different amounts of time. Already matching saved results can contribute immediately on a rerun. A file count does not measure how much time each file needs. The final 100% appears when the operation completes successfully.\n\n**3. Train Index** builds a feature-retrieval database from the extracted features. Its progress uses allocated phases: loading features up to 20%, optional clustering, index training at 40%, adding feature blocks from 60% to 95%, and 100% after the final index file is written. Clustering and FAISS training do not expose continuous progress here, so the number can pause during those phases. These percentages are workflow progress, not a precise fraction of time elapsed.\n\n**4. Train Pth** trains the neural voice model. **Training Info** shows the percentage of fully completed epochs, `Epochs`, cumulative `Total Steps`, `Remaining`, and `Estimated finish`. The second line shows the latest completed epoch and its duration. The CMD window shows the same clean epoch summary.\n\nThe remaining time is a prediction based on recent measured epochs, with greater weight given to newer measurements. It can change as the computer's speed changes. It appears after the first completed epoch. The finish time uses the PC's local clock and the format **day-month-year hours:minutes:seconds**. Final model saving may add time after the last epoch. **DONE!** includes the observed completion time after the training process ends successfully.\n\nFor detailed diagnostics, check `logs/<Model Name>/preprocess.log`, `extract_f0_feature.log`, `train.log`, and `training_console.log`. A one-line progress display is deliberately shorter than these files. A failed operation should be fixed before starting the next step.\n\n**Training while CMD is in the background (Windows).** The trainer and its data-loading worker processes request HighQoS (execution-speed power throttling off). CPU scheduling priority is left unchanged by default: raising all worker priorities is not a guaranteed speed improvement. This is a temporary per-process request; it does not change the system power plan, registry, GPU settings, or the priority of unrelated applications. Existing higher priorities are kept. It can reduce CPU scheduling/power-related slowdowns after the terminal loses focus, but it does not guarantee identical epoch times: GPU contention, temperature, memory pressure, and other workloads still matter. On unsupported systems the optional request is skipped or logged, and training continues. It takes effect when a new training job starts; it does not modify an already running job."),
    ('8. Voice Model, Refresh, and unloading VRAM',
     '**Voice Model** selects the inference-ready `.pth` file. This selection is shared by Single and Batch Inference. Selecting a different model loads it and tries to match an index. The application selects the first model alphabetically at startup; **Refresh** also resets the model choice to the first alphabetical entry. Check the selection again after refreshing.\n\n**Refresh** rescans available models and indexes. Use it after adding or exporting files. It does not train a model or repair an invalid file.\n\n**Unload Voice from VRAM** clears the current voice selection and requests cleanup of loaded inference resources. Use it when you need GPU memory for another operation. It does not delete model files. Select a model again before converting more audio.\n\n**Speaker ID** lives in Inference Settings and may be hidden until a model is loaded. Leave it at `0` for a normal single-speaker model. It chooses a speaker slot already stored in the loaded model; it is not a GPU number or a way to load a different `.pth` file.'),
    ('9. Audio to convert and Pitch',
     '**Audio to convert** is the source recording whose voice will be changed. Single Inference accepts a dropped/uploaded audio file, so you do not need to type a filesystem path. Upload a short, clearly audible recording for your first test. The source words, timing, and performance come from this recording; choosing a model does not write new words or a melody.\n\n**Pitch \u2014 default `0`, range `\u221224` to `+24`.** This shifts the extracted pitch in semitones when the model uses pitch guidance. Positive values raise it; negative values lower it. `+12` is one octave up, `\u221212` one octave down, and `\xb124` two octaves. Start at `0`, then try small adjustments and listen. This slider is not a volume or playback-speed control. Models trained without pitch guidance do not use this pitch contour in the same way.\n\n**Conversion** processes the single uploaded recording. **Output** reports success or an error and may include processing timings. **Output Audio** is the result to play and download. If there is no successful audio result, read Output before repeatedly clicking the button.'),
    ('10. Inference pitch extractors',
     "These choices estimate the source recording's pitch during conversion. They do not retrain the model. The selection is separate in Single and Batch Inference.\n\n| Option | Meaning in this installation |\n| --- | --- |\n| `pm` | Parselmouth pitch estimation. A CPU alternative you can compare with the default. |\n| `harvest` | WORLD Harvest estimation. This is the method affected by the median-filter setting. |\n| `crepe` | Neural CREPE pitch estimation. Available in the standard inference menus; hidden in DirectML mode in this UI. |\n| `rmvpe` | RMVPE neural pitch estimation, using the configured inference device. Default in both inference tabs. |\n\nThe training menu's `rmvpe_gpu` label does not appear separately here. Inference RMVPE uses the application's configured device rather than the training CPU extraction route. Training also offers `dio`, but this inference UI does not. Do not expect methods from a newer RVC release to appear in this installation.\n\nIf pitch sounds unstable, compare another available extractor on the same short clip. Changing the extractor cannot remove accompaniment, recover clipped audio, or fix an unrelated index file."),
    ('11. Index paths and Index Ratio \u2014 why selecting an index may change nothing',
     'The **`.pth` voice model** contains the trained neural network. The **`.index` file** is a separate database of extracted dataset features used during retrieval. A model can convert audio without an index.\n\n**Auto-detect index path** is a dropdown of available index files. When you select a Voice Model, this customized UI searches for matching folder/name information and fills the matching index in both inference tabs. If no match is found, the old paths are cleared. Keep model and experiment names consistent to help matching. Use the completed `added_...index` file, rather than the intermediate `trained_...index` file.\n\n**Path of index / Path to index** is the manual index path. **A nonempty manual path takes priority over the dropdown.** Model selection fills both fields. If you choose a different entry in the dropdown, also clear the manual text field or update it to the same path. Otherwise the earlier manual path can still be used.\n\n**Index Ratio \u2014 default `0`, range `0\u20131`.** This sets how strongly retrieved training features are mixed into the conversion. At **0**, retrieval is disabled even if an index is selected. To test a matching index, try a moderate nonzero value such as `0.25`, convert again, and compare. Larger values are stronger, not automatically better; using an unrelated index can make the result worse.\n\nIf selecting an index seems to do nothing, check Index Ratio first, then check which path field is taking priority. When comparing indexes, keep the same model, input, Pitch, and other settings.'),
    ('12. Every other Inference Setting',
     "**Resampling, 0=none \u2014 default `0`, displayed range `0\u201348000`.** Leave `0` to keep the model's output sample rate. This backend only applies a new rate of **16000 Hz or above**; a nonzero value below 16000 does not request a valid output-rate change here. Use it to meet an output file's sample-rate requirement, not to make the model learn or recover lost audio detail.\n\n**0=Input source volume, 1=Normalized Output \u2014 default `0.25`.** This controls the blend of the source and converted audio's loudness envelope. Closer to `0` follows the source's loud/quiet pattern more strongly. At `1`, this extra source-envelope matching is skipped and the converted signal keeps its own envelope. Despite the label, `1` is not a guarantee of a particular loudness level. This is separate from the volume knob on your speakers.\n\n**Protect clear consonants and breathing sounds\u2026 \u2014 default `0.33`, range `0\u20130.5`.** This reduces the influence of retrieved features on unvoiced sounds such as consonants and breaths in pitch-guided conversion. **Lower values give stronger protection; `0.5` disables it.** It matters when retrieval changes the features, so its effect can be minimal with Index Ratio `0`. The label's \u201c0.5 does not open\u201d means \u201cprotection is off at 0.5.\u201d\n\n**>=3 apply median filter to the harvested pitch results \u2014 default `3`, range `0\u20137`.** This smooths the pitch contour specifically for **harvest** when the value is at least `3`. Values `0\u20132` leave this optional filter off. In this implementation, values `3\u20137` all enable the same three-sample filter; `7` does not mean stronger smoothing than `3`. It does not smooth RMVPE, PM, or CREPE in this pipeline.\n\n**F0 curve file [optional]** is hidden in the normal UI. The backend has an advanced custom pitch-curve input; it is not required for regular conversion. You do not need to upload an additional pitch file for this glossary's workflow."),
    ('13. Batch Inference \u2014 convert multiple files',
     "Batch Inference uses the same selected Voice Model as Single Inference, but its own Pitch and Inference Settings. A setting changed in Single Inference does not automatically change the batch setting. Model selection does synchronize the matched index paths.\n\n1. Choose the intended **Voice Model**.\n2. Open **Batch Inference** and drop multiple recordings into **Audio to convert**.\n3. Leave **Enter the path to the audio folder to be processed** empty when using uploaded files. A valid nonempty folder path takes priority over uploads.\n4. Set **Pitch**, then open **Inference Settings** if you need to change the batch conversion settings.\n5. Check the **Output** folder, default `InferOutput`, and choose **Export Format**.\n6. Click **Conversions**. Read the per-file result and find successful exports in the chosen output folder.\n\n**Input folder alternative.** Enter a real folder path instead of uploading files. Put only intended audio inputs there: the backend lists the folder directly and does not recursively discover audio in subfolders. Keep the output folder separate from the input folder.\n\n**Output \u2014 default `InferOutput`.** This is a disk folder, not the name of one audio file. A relative folder is inside the application folder. The application creates it when needed. A name can include the original extension, such as `song.wav.flac`. Reusing the same output name can overwrite an earlier export; choose a different output folder if you want to preserve comparisons.\n\n**Export Format \u2014 default `flac`.** `flac` and `wav` are lossless output choices; `mp3` and `m4a` are compressed delivery formats. Changing the format does not change what voice the model learned. Single Inference's downloadable audio player is separate from this batch export-format control."),
    ('14. Files and folders \u2014 what belongs where',
     "| Location | Purpose |\n| --- | --- |\n| `datasets/` or your chosen input folder | Original recordings you provide for training. |\n| `logs/<Model Name>/0_gt_wavs/` | Prepared chunks at the training sample rate. |\n| `logs/<Model Name>/1_16k_wavs/` | 16 kHz chunks used by extraction. |\n| `logs/<Model Name>/2a_f0/` and `2b-f0nsf/` | Pitch results for pitch-guided training. |\n| `logs/<Model Name>/3_feature768/` | Extracted audio features for this V2 workflow. |\n| `logs/<Model Name>/G_*.pth` and `D_*.pth` | Full training checkpoints used to continue training. |\n| `models/<Model Name>/added_*.index` | Completed retrieval index for inference. |\n| `models/<Model Name>/<Model Name>.pth` | Exported voice model selected in Voice Model. |\n| `assets/pretrained_v2/` | Starting pretrained G/D models; not your exported voice collection. |\n| `InferOutput/` or your chosen output folder | Batch-converted audio. |\n\n**A `.pth` extension alone does not identify the file's role.** A full Generator/Discriminator checkpoint and an exported voice model are different. Do not fill Voice Model's weights folder with training checkpoint pairs. The special `logs/mute` folder is used by the application's training preparation; leave it in place."),
    ('15. Troubleshooting \u2014 check this before starting over',
     '**The name box shows an example but the action asks for a name.** Example text is only a placeholder. Click the box and type an actual name. Training needs Model Name; merging needs New model name. They are independent.\n\n**\u201c3_feature768\u201d is missing, or \u201cError: Run 2. Feature Extraction first.\u201d** Run Process Data and then Feature Extraction for the same Model Name. Wait for each success message. If extraction failed, read its log; creating an empty folder does not create the needed features.\n\n**The model does not appear.** Check that an exported inference `.pth` is in `models`, click Refresh, and select the intended model again. An interrupted training run may not have reached an inference export yet.\n\n**The index is missing or the old one is used.** Finish Train Index, check naming, select the model again, and inspect both index path fields. The manual field has priority. At Index Ratio `0`, no index retrieval is applied.\n\n**Out of GPU memory.** Lower Batch Size, disable Cache data to GPU, unload the inference voice, and close other GPU-heavy tasks before retrying. For RMVPE GPU extraction, reduce repeated GPU IDs/processes if needed.\n\n**Progress stays at the same number.** A long source file, first-epoch setup, clustering, or FAISS training can take time without a new completed unit. Check the CMD/log files for errors. Clicking the action repeatedly can start competing work; do not use repeated clicks as a progress refresh.\n\n**The conversion is distorted or sounds wrong.** Confirm the correct Voice Model, test Pitch `0`, compare a clean vocal-only input, and test with Index Ratio `0` to remove retrieval as a variable. Then adjust one control at a time.\n\n**The finished model did not improve after many epochs.** Extra epochs cannot repair a mixed-speaker, noisy, or clipped dataset. Review the recordings and preparation before increasing the training target.\n\n**A change seems to have no effect.** Check that you changed the correct Single/Batch settings and actually ran Conversion again. For application-code updates, restart the Web UI after any active job has finished.'),
    ('16. Quick reference and a default-settings checklist',
     '| Term | Plain-English meaning |\n| --- | --- |\n| Dataset | The recordings used to teach a voice model. |\n| Inference | Using a trained model to convert audio. |\n| F0 / pitch contour | The measured pitch over time. |\n| Feature | A numerical description of audio used by the model. |\n| Epoch | One pass through the prepared training dataset. |\n| Batch / step | A group of training examples and one batch update. Total Steps accumulates these updates. |\n| Checkpoint | Saved training state used to continue a run. |\n| Index | The optional database of dataset features used for retrieval. |\n| VRAM | Memory on the GPU; different from normal system RAM. |\n| ETA | Estimated remaining time; a prediction that can change. |\n\n**Training defaults:** V2; Training Precision **FP32**; Sample Rate **32k**; Pitch Guidance **True**; CPU Threads **8**; Speaker ID **0**; Pitch Extraction **rmvpe**; Save frequency **100**; Total Epochs **300**; Batch Size **8**; latest checkpoint **Yes**; Cache data to GPU **No**; Create model at save frequency **No**.\n\n**Inference defaults in both tabs:** Pitch **0**; Pitch Extraction **rmvpe**; Index Ratio **0**; Resampling **0**; volume-envelope mix **0.25**; protection **0.33**; Harvest median filter **3**; Speaker ID **0** where applicable. Batch output folder **InferOutput**, export format **flac**.\n\nUse these values to get a baseline, then make one deliberate change at a time. The best check is a controlled listening comparison using the same short input recording.'),
    ('17. RVC concepts and the wider ecosystem',
     "**RVC \u2014 Retrieval-based Voice Conversion.** A system that converts the voice character of existing audio. Its main pieces are a content encoder, a pitch estimator, an acoustic generator, and optional feature retrieval. It uses recorded audio as input; it is not a text-to-speech interface.\n\n**F0 / fundamental frequency.** The base repetition frequency of vocal-fold vibration in voiced speech or singing, closely related to perceived pitch. Pitch estimation or tracking measures it over time. Consonants and breaths can be unvoiced and do not always have a meaningful F0.\n\n**RMVPE \u2014 Robust Model for Vocal Pitch Estimation.** A neural pitch estimator. Its F0 output conditions pitch-guided RVC training and conversion, telling the generator about high and low notes. It does not teach the words or select the speaker on its own.\n\n**Embedder / content encoder.** A pretrained model that turns audio into numerical representations. A content encoder aims to preserve linguistic information while reducing speaker-specific information. It supplies learned speech features; it does not literally teach every language or guarantee correct pronunciation.\n\n**ContentVec / HuBERT.** Standard RVC uses a HuBERT-style content encoder associated with ContentVec, commonly distributed as `hubert_base.pt`. This installation loads `assets/hubert/hubert_base.pt`. The filename alone does not identify every detail of how those weights were trained.\n\n**Codebook.** A set of discrete representative codes used in some speech-representation training methods. It is different from the continuous feature vectors used by RVC's retrieval index. Codebook size is not a quality score: architecture, training data, and compatibility also matter. The glossary does not assume a particular codebook size for an arbitrary downloaded checkpoint.\n\n**SpinV2.** An alternative speech representation/embedding approach found in parts of the RVC ecosystem, rather than a selectable embedder in this UI. Using another embedder requires compatible extraction and model/pretraining support; renaming its file to `hubert_base.pt` is not a supported upgrade. A larger codebook alone does not establish that it will sound better here.\n\n**AI architecture.** The design of a model: its components, connections, and how information flows through them. A different architecture can require different weights and training code.\n\n**Mainline and fork.** Mainline refers here to the [original RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). A fork is a separately modifiable copy of a repository, often used for experiments, fixes, and new features. Features in a fork are not automatically present in this installation.\n\n**Vocoder.** A neural component that generates an audio waveform from conditioning features. Standard RVC uses an NSF-HiFi-GAN-style waveform generator conditioned on learned features and F0, rather than requiring a plain mel spectrogram as its only input. RefineGAN, HiFi-GAN, and RingFormer are names you may encounter for other waveform-generation designs in the wider ecosystem; they are not interchangeable settings exposed in this app.\n\n**Pretrain / pretrained model.** Weights already learned from a larger training task, reused as a starting point. **Fine-tuning** continues training those weights on a more specific dataset, such as Ilaria's voice recordings. Your dataset, pitch setup, sample rate, architecture, and pretrained files must be compatible.\n\n**VCTK.** A multi-speaker English speech dataset often discussed in speech-model training. It is not your personal training dataset. Do not infer the exact training data of every RVC pretrain from its filename, or assume all pretrained models use the same corpus. Speech recordings and expressive singing can have different coverage and demands.\n\n**Epoch and batch size.** An epoch is a pass through the prepared training data; batch size is how many prepared examples are handled together. An epoch is not a second, and batch size is not dataset duration. See section 6 for cumulative steps and resume behavior.\n\n**Overtraining / overfitting.** Training can become too specialized to its examples and generalize poorly to new performances. More training can stop improving or worsen results; robotic artifacts alone do not prove overfitting, because bad input, pitch errors, and other settings can also cause them. Compare held-out test audio and intermediate exports when available.\n\n**Generator and discriminator.** The generator produces audio; the discriminator learns to distinguish generated audio from real training audio. Their competing objectives provide feedback during training. The discriminator helps train the generator but is not the voice model selected for normal inference.\n\n**VITS.** Originally an end-to-end text-to-speech architecture. RVC adapts ideas from VITS for audio-conditioned voice conversion; this does not make this UI a text-to-speech tool. See the [original VITS repository](https://github.com/jaywalnut310/vits).\n\n**RVC-Boss.** A prominent developer/maintainer associated with the RVC project. RVC includes contributions from a team and upstream research projects; credit and checkpoint provenance should be checked in the relevant repositories.\n\n**Further reading:** [Understanding RVC \u2014 Retrieval-based Voice Conversion](https://gudgud96.github.io/2024/09/26/annotated-rvc/) explains content extraction, RMVPE, the VITS-based acoustic model, and retrieval in more depth. It is an external technical explanation; this glossary's control descriptions refer to the local ILARIA RVC build."),
    ('18. Downloading, importing, and merging models',
     "Open **Other**, between Inference and Glossary. All three sections start closed.\n\n**Download models from Hugging Face, Google Drive or MEGA:** paste a public Hugging Face repository or direct ZIP URL, or a Google Drive or MEGA share link to one `.zip`, `.pth`, or `.index` file. For Drive, set General access to Anyone with the link and allow downloads. For MEGA, copy the full file link including the key after #. Click **Download and install model**. If a Hugging Face repository contains several ZIPs, choose a direct link from the log. For a model and its index stored as separate Drive or MEGA files, paste each link in turn or put both in one ZIP.\n\n**Import local models:** drop one or more `.zip`, `.pth`, and `.index` files, then click **Import model files**. ZIPs can contain nested folders; the importer takes the supported model/index files rather than installing the entire archive. Inference models go into `models`, and indexes go into matching folders under `models`. Training G_/D_ checkpoint pairs are not inference model exports.\n\nAfter success, go to **Inference**, click **Refresh**, and choose the intended model. Refresh selects the first alphabetical model, so check the selection. A matching index is optional, and Index Ratio defaults to zero. Existing different files are kept and a naming conflict is reported; identical files are left unchanged. Importing copies files and does not train or convert audio. Only select model files you intend to use.\n\n**Merge models** is the first section. Select two installed inference models, click Check model compatibility, and read the rate, version, pitch mode, and shape checks. Both models must match: this tool does not convert a 32 kHz model into a 40 or 48 kHz model. Auto detects their shared rate; a manual rate checks that it matches.\n\nThe Model A contribution slider uses percentages: 50 means 50% A + 50% B, 100 keeps A, and 0 keeps B. This is interpolation of neural-network weights, not a guarantee of an equally recognizable voice mixture. Choose FP16 for a smaller exported model or FP32 to retain full-precision blended weights. Calculations run on CPU in FP32. Choose a new model name and optional notes, then click Merge and save model. Models with different tensor shapes (including incompatible speaker embeddings) are rejected rather than silently truncated.\n\nThe result goes into models without replacing the originals. Refresh Inference and select the output. No index is merged or created; first test with Index Ratio 0. Import, download, compatibility, and merge logs show phase information and progress. Download progress is based on bytes when the server provides a file size; workflow stage percentages are not time estimates. Completion reaches 100% only after the relevant work finishes successfully.\n\n**New model name \u2014 required; starts empty.** The faint `merged_voice` text is an example placeholder, not a name that has already been filled in. Click the box and type a new name such as `voice_a_b_mix50`; there is no prefilled text to erase. This field names the **new merged output**, not Model A or Model B, and does not rename either original.\n\n**A complete merging example:**\n\n1. Open Other, then Merge models. Choose an installed voice in Model A and another compatible voice in Model B.\n2. Click Check model compatibility and wait for its result. Choose Auto (from models) for sample rate unless you are checking a specific matching rate.\n3. Set Model A contribution to `50` for a 50% A / 50% B weight blend. Keep FP16 for a smaller output, or choose FP32 if you want full-precision blended weights.\n4. In New model name, type `voice_a_b_mix50`. Use a simple filename without a folder path; `.pth` is added automatically. If you do include one `.pth` suffix, the merger handles it without adding a second suffix.\n5. Optionally type a description in Model notes (optional), for example `50% Voice A and 50% Voice B, first test`. You can leave notes empty. Notes describe the model; they do not set its filename or change the blend ratio.\n6. Click Merge and save model. Wait for `Merge done!` and 100%. With this example name, the result is `models/voice_a_b_mix50/voice_a_b_mix50.pth`.\n7. Open Inference, click Refresh, and select `voice_a_b_mix50.pth` yourself. Refresh selects the first alphabetical model, which might be a different voice. Test with a short recording and Index Ratio 0 first.\n\n**Blank output name:** the merger reports an invalid output name and does not save a nameless model. Seeing the placeholder does not satisfy this requirement. Type your own name before merging.\n\n**Existing output name:** the merger refuses to overwrite an existing model. Choose another name such as `voice_a_b_mix50_test2`; do not use the filename of Model A or Model B as the output name. The Training Model Name and the merging New model name are separate fields: entering one does not automatically fill the other.\n\n**Placeholder versus default versus optional:** a placeholder is only a hint in an empty box; a default is an actual selected value such as the 50% blend or FP16 precision; optional means the field may stay empty. New model name and Training's Model Name are required even though they show example text. Model notes really is optional.\n\n\n**Model information (last menu in Other).** Open the menu and select a Voice Model to read it automatically, or click Read model information to read it again. Refresh model list discovers newly imported or merged files. Inspection reads the checkpoint on CPU; it does not start inference or training.\n\nThe report shows file size, last modification time from your PC clock, SHA-256 fingerprint, saved RVC version, sample rate, pitch guidance, speaker embedding slots, stored tensor precision and size, any saved epoch/step information, notes and matching index filenames. Expand Saved metadata and architecture for the raw metadata and configuration fields. Expand Stored tensors for every displayed weight's name, shape, type and size.\n\nMissing facts are labelled as not stored. FP16/FP32 in this report describes the saved weights, not proof of the precision used during training. The last file modification time is not a training-completion timestamp; speaker slots do not prove that every slot was trained. Matching indexes are identified by names, not by a compatibility test. No training duration, dataset quality or GPU is guessed from the weights.\n\n\n**Program updates (Other > Update Ilaria RVC).** The UI checks for releases when the page opens. Installed version shows the version on your PC. Check for updates repeats the check and shows the publisher's release notes. Download and restart appears when a newer compatible release is published.\n\nFinish training, conversion, importing, merging and other active operations before updating. The updater refuses to interrupt them, including training processes that continue after a cancelled UI request. Start the program with ILARIA_RVC.bat so it can install and restart automatically.\n\nClick Download and restart to download the release with progress, verify its contents, create backups and restart the UI. Keep the CMD window open. This page reloads after the new version starts; if your browser does not reload, refresh it manually. The CMD shows installation progress while the web server is closed.\n\nOnly managed program files are replaced. Your dataset, voice models, indexes, experiment logs and runtime are preserved. Backups are stored inside .ilaria-updates/backups in the installation folder. A failed installation restores the previous files; after an interrupted installation, the launcher attempts recovery before opening the UI.\n\nIf a second launch is blocked, use the existing UI address or return to the original CMD window. Closing a browser tab does not stop the backend. Before restarting, finish training or use Stop and save and wait for confirmation. The launcher message identifies the existing process and address when available; do not delete launcher.lock to stop the program.\n\nAn internet error or an update feed that has not been published does not prevent you from using the program. Check again later. The first installation of the updater must be copied into an existing compatible Ilaria RVC installation; subsequent published versions can use the built-in updater.\n"),
)

TUTORIAL_SECTIONS += (('19. Dataset checker: read the report before training', "Open Training > Dataset checker. Set Training Settings & Other > Path to Dataset to the folder containing your original recordings, then click Check dataset. The checker reads supported audio files directly inside that folder, just as Process Data does. It does not recursively inspect subfolders, repair recordings, normalize volume, delete silence or change any source file. Wait for the final report before deciding what to fix.\n\nSilence threshold defaults to -45 dBFS. A quieter analysis window is counted as silence; pauses lasting at least 2 seconds are flagged. Lower values, such as -55, count only quieter passages as silence. The duration control changes how long a pause must be to trigger a warning. Breaths and natural pauses can be useful: review a flagged recording before removing anything.\n\nClipping checks samples very close to full digital scale and reports possible overload. A file may already contain distortion even if it was later made quieter; this analysis cannot prove a recording is clean. Damaged, empty, unreadable or invalid files are reported separately. Fix or remove broken recordings yourself and run the check again.\n\nVolume warnings compare the level of active audio between files. The default tolerance is 6 dB from the dataset's median active level. This is a screening measurement, not a loudness-normalization target or a quality score. Quiet expressions can be intentional. Compare recordings by ear and adjust inconsistent recording gain without flattening useful expression. Sample-rate differences are information; Process Data resamples audio to the chosen training rate.\n\nThe percentage measures files checked, not time remaining. Long recordings take longer than short ones. The report lists each file and measurements so you can find the recording that needs attention. A successful scan can still contain warnings. No warning is a guarantee that a voice dataset is suitable; background music, echo and overlapping speakers still need listening checks."), ('20. Training previews: compare the same clip over time', 'Open Training > Training previews before starting Train Pth. Tick Generate training previews, upload a clean test recording no longer than 30 seconds, and set Preview every N epochs (default 10). Use the same source performance when comparing epochs; an audio clip not included in the training dataset is a useful listening test. Set Preview Pitch to the transposition you want, from -24 to +24 semitones (default 0), and choose Preview Pitch Extraction. rmvpe is the default CPU option; the other displayed methods offer alternative pitch tracking.\n\nAt the configured epoch boundaries the trainer exports a temporary snapshot of the current generator and converts the saved source clip. It uses CPU FP32 and Index Ratio 0, so the preview does not allocate another model on the training GPU or depend on a matching retrieval index. Preview generation adds waiting time; choose a larger interval or disable previews if speed matters more. The preview uses its own conversion process and does not replace the voice selected in Inference.\n\nPreview Log and Training Preview Audio update when a saved result becomes available. Open Saved preview epoch to listen to an earlier result; Refresh previews reloads saved results for the Model Name field. Files are kept under logs/<Model Name>/previews. Keep enough disk space for the WAV files. Source audio is copied into the experiment so a temporary upload does not disappear mid-training.\n\nPreview settings are captured when Train Pth starts. Editing these fields during a run does not alter that run. A preview failure is reported and the trainer continues; read Preview Log for its cause. A preview is a listening aid, not an automatic quality score. Compare pronunciation, pitch stability and artifacts using the same source, pitch and playback volume.'), ('21. Stop and save: finish an epoch and resume safely', 'While Train Pth is running, click Stop and save once. Stop / Resume confirms the request. Training finishes the current epoch, then writes a resumable generator/discriminator checkpoint pair and training state. Wait for STOPPED! Training saved in Training Log and CMD before closing the application. The button does not immediately cancel a batch, and a long epoch or preview may take time to finish. Closing CMD or terminating Python before confirmation can lose progress since the previous checkpoint.\n\nTo resume, open the same installation, type the exact same Model Name, keep the dataset, sample rate, RVC version, pitch guidance, Speaker ID, batch size, GPU count and architecture compatible, and click 4. Train Pth again. Do not repeat Process Data or Feature Extraction for unchanged prepared data. Total Epochs is the final target, not the number of additional epochs: if you stopped after 80 and choose 300, training continues at epoch 81 toward 300. You may increase that target.\n\nNew checkpoints preserve the completed epoch, cumulative steps, generator and discriminator weights, both optimizers, learning-rate schedulers, mixed-precision scaler and random-number states. Resuming starts at the next epoch instead of counting the saved epoch twice. A resumed training log confirms the last saved epoch and steps. ETA is recalculated from new measurements; time spent closed is not active training time.\n\nKeep the entire logs/<Model Name> experiment, including filelist.txt, config.json, prepared audio/features/pitch and training-state. The small .pth voice file in models is for inference; it is not a substitute for these resumable files. Managed checkpoints are also saved at Save frequency and normal completion. Save last ckpt retains the latest managed generation when enabled; disabling it keeps earlier generations and uses more disk space. Broken or incompatible managed state is reported rather than silently starting a new model. Use a new Model Name for a different dataset or incompatible settings.\n\nSTOPPED! means safely saved partway through training. DONE! Training completed means the total target was reached and the final inference export succeeded. An early stop does not display a completed-training message. Training previews remain available independently of whether you stopped or finished.'))

ILARIARVC_CSS += '\n/* Tutorial: readable documentation using the existing Ilaria palette. */\n#tutorial-content { gap: 12px; max-width: 1080px; margin: 0 auto; }\n#tutorial-content .gr-prose { font-size: 15px; line-height: 1.7; }\n#tutorial-content .gr-prose p, #tutorial-content .gr-prose li { color: #F8EFFA; }\n#tutorial-content .gr-prose strong { color: #FFC0CB; }\n#tutorial-content .gr-prose h1 { font-size: 1.8rem; margin-bottom: 12px; }\n#tutorial-content .gr-prose code {\n    background: #2E1D36; color: #F8EFFA; border-radius: 4px;\n    padding: 2px 5px; overflow-wrap: anywhere;\n}\n#tutorial-content .gr-prose table {\n    display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse;\n}\n#tutorial-content .gr-prose th, #tutorial-content .gr-prose td {\n    border: 1px solid #866190; padding: 9px 12px; color: #F8EFFA;\n}\n#tutorial-content .gr-prose th { background: #2E1D36; color: #FFC0CB; }\n#tutorial-content .gr-prose td { background: #412849; }\n#tutorial-content .gr-prose li { margin: 6px 0; }\n#tutorial-content .gr-prose ol { padding-left: 24px; }\n#tutorial-content .gr-box { border-radius: 12px; }\n'

ILARIARVC_CSS += '\n/* Other: visible pink-lilac outlines on the purple background. */\n.gradio-container #ilaria-main-tabs #other-tab .gr-box,\n.gradio-container #ilaria-main-tabs #other-tab .gr-panel,\n.gradio-container #ilaria-main-tabs #other-tab .gr-form {\n    border: 1px solid #C59ACD !important;\n}\n.gradio-container #ilaria-main-tabs #other-tab .gr-accordion {\n    border: 2px solid #D8B4FE !important;\n    border-radius: 12px !important;\n    background: #3D2646 !important;\n}\n.gradio-container #other-tab .gr-accordion > .label-wrap {\n    color: #FFFFFF !important;\n}\n.gradio-container #ilaria-main-tabs #other-tab .gr-accordion:hover {\n    border-color: #FFC0CB !important;\n}\n'

ILARIARVC_CSS += '\n/* Use explicit accordion IDs: this Gradio version has no gr-accordion class. */\n.gradio-container #ilaria-main-tabs #other-tab #merge-models,\n.gradio-container #ilaria-main-tabs #other-tab #download-models,\n.gradio-container #ilaria-main-tabs #other-tab #import-models {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n    box-shadow: none !important;\n    outline: none !important;\n}\n#other-tab #merge-models > *,\n#other-tab #download-models > *,\n#other-tab #import-models > * {\n    border-color: #D8B4FE !important;\n}\n#other-tab #merge-models > .label-wrap,\n#other-tab #download-models > .label-wrap,\n#other-tab #import-models > .label-wrap {\n    background: #62416B !important;\n    color: #FFFFFF !important;\n    box-shadow: none !important;\n}\n'

ILARIARVC_CSS += '\n/* Keep explanatory text clear of panel outlines. */\n.gradio-container #ilaria-main-tabs #download-models-help,\n.gradio-container #ilaria-main-tabs #import-models-help,\n.gradio-container #ilaria-main-tabs #model-info-help {\n    box-sizing: border-box;\n    padding: 12px 18px !important;\n    text-align: center !important;\n    line-height: 1.65 !important;\n}\n#download-models-help p, #import-models-help p, #model-info-help p {\n    text-align: center !important;\n    margin: 0 !important;\n}\n.gradio-container #ilaria-main-tabs #other-tab #model-information {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n}\n.gradio-container #ilaria-main-tabs #model-info-summary,\n.gradio-container #ilaria-main-tabs #model-info-tensors {\n    padding: 12px 18px !important;\n    text-align: left;\n    overflow-x: auto;\n}\n#model-info-tensors { max-height: 460px; overflow-y: auto; }\n#model-info-summary table, #model-info-tensors table { width: 100%; border-collapse: collapse; }\n#model-info-summary td, #model-info-summary th,\n#model-info-tensors td, #model-info-tensors th {\n    padding: 8px 12px;\n    border: 1px solid #A17BA9;\n    text-align: left;\n    overflow-wrap: anywhere;\n    color: #F8EFFA !important;\n}\n#model-info-summary th, #model-info-tensors th { background: #2E1D36; color: #FFC0CB !important; }\n#model-information .gr-panel { border-color: #C59ACD !important; }\n.gradio-container #ilaria-main-tabs #easter-egg-image {\n    border: 0 !important;\n    background: transparent !important;\n    padding: 20px !important;\n    text-align: center;\n}\n#easter-egg-image img { display: block; max-width: 100%; max-height: 75vh; margin: 0 auto; border-radius: 12px; }\n'

ILARIARVC_CSS += '\n/* Merge descriptions use the same centered, padded layout as download help. */\n.gradio-container #ilaria-main-tabs #merge-models .gr-prose,\n.gradio-container #ilaria-main-tabs #merge-models .prose {\n    box-sizing: border-box;\n    padding: 12px 18px !important;\n    text-align: center !important;\n    line-height: 1.65 !important;\n}\n#merge-models .gr-prose p, #merge-models .prose p,\n#merge-models .gr-prose h3, #merge-models .prose h3 {\n    text-align: center !important;\n}\n'

ILARIARVC_CSS += '\n/* Glossary: centered explanations with breathing room inside panel borders. */\n.gradio-container #ilaria-main-tabs #tutorial-content .gr-prose,\n.gradio-container #ilaria-main-tabs #tutorial-content .prose {\n    box-sizing: border-box;\n    padding: 12px 18px !important;\n    text-align: center !important;\n    line-height: 1.7 !important;\n}\n#tutorial-content .gr-prose p, #tutorial-content .prose p,\n#tutorial-content .gr-prose h1, #tutorial-content .prose h1,\n#tutorial-content .gr-prose h2, #tutorial-content .prose h2,\n#tutorial-content .gr-prose h3, #tutorial-content .prose h3 {\n    text-align: center !important;\n}\n/* Keep list markers beside their text and tables easy to compare. */\n#tutorial-content .gr-prose ul, #tutorial-content .gr-prose ol,\n#tutorial-content .prose ul, #tutorial-content .prose ol {\n    display: table;\n    margin-left: auto;\n    margin-right: auto;\n    max-width: 100%;\n    text-align: left;\n}\n#tutorial-content .gr-prose table, #tutorial-content .prose table {\n    margin-left: auto;\n    margin-right: auto;\n    text-align: left;\n}\n'

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #other-tab #program-updates {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n}\n.gradio-container #ilaria-main-tabs #program-updates-help {\n    padding: 12px 18px !important;\n    text-align: center !important;\n}\n#program-updates-help p { text-align: center !important; }\n#program-update-notice { padding: 12px 18px !important; border: 1px solid #FFC0CB !important; border-radius: 12px; text-align: center; color: #FFFFFF; }\n'

# Track work independently of Gradio cancellation, including subprocesses.
for operation_name, operation_label in (
    ('preprocess_dataset', 'data preprocessing'), ('extract_f0_feature', 'feature extraction'),
    ('click_train', 'model training'), ('train_index', 'index training'),
    ('select_voice_model', 'model loading'), ('batch_inference_audio', 'batch conversion'),
    ('merge_voice_models', 'model merging'), ('inspect_merge_models', 'model inspection'),
    ('download_voice_models', 'model download'), ('import_voice_models', 'model import'),
    ('inspect_voice_model', 'model information'),
    ('scan_training_advice', 'PC training advice'),
    ('test_training_alerts', 'notification test'),
    ('import_training_dataset', 'dataset import'),
    ('export_voice_model', 'model export'),
    ('download_training_pretrained', 'pretrained download'),
    ('download_recommended_pretrained', 'recommended pretrained download'),
    ('add_training_to_queue', 'queue preparation'),
    ('check_training_dataset', 'dataset check'), ('read_training_preview', 'training preview'),
):
    globals()[operation_name] = update_operation_gate.track(globals()[operation_name], operation_label)
from ilaria_tts import build_tab as build_tts_tab

_convert_voice = vc.vc_single


def convert_voice(*args, **kwargs):
    with voice_lock:
        return _convert_voice(*args, **kwargs)


vc.vc_single = update_operation_gate.track(convert_voice, "audio conversion")


def convert_tts_voice(model, speaker, path, pitch, method, index, ratio,
                      median, sample_rate, envelope, protect):
    with voice_lock:
        if loaded_voice != model or getattr(vc, "net_g", None) is None:
            return "The selected model is not loaded. Select it above and try again.", None
        speaker = int(speaker or 0)
        if not 0 <= speaker < vc.cpt["config"][-3]:
            return "Choose a Speaker ID supported by this model.", None
        rate = 0 if sample_rate == "Model default" else int(sample_rate)
        message, audio = vc.vc_single(speaker, path, pitch, None, method, "",
                                      index or "", ratio, median, rate, envelope, protect)
        if audio is None and message.startswith("Traceback"):
            message = "RVC conversion failed. " + message.strip().splitlines()[-1]
        return message, audio


class NamedInferenceAudio(gr.Audio):
    def get_block_name(self):
        return "audio"

    def postprocess(self, value):
        if isinstance(value, str):
            path = pathlib.Path(value).resolve()
            if pathlib.Path(tmp).resolve() in path.parents:
                self.temp_files.add(str(path))
                return {"name": str(path), "data": None, "is_file": True}
        return super().postprocess(value)


from ilaria_inference_export import EXPORT_FORMATS


def named_inference_audio(audio, model, output_format="wav"):
    if audio is None:
        return gr.update(value=None)
    import tempfile
    from ilaria_inference_export import write_inference_export

    name = pathlib.Path(model or "Voice").stem + " output"
    if isinstance(audio, pathlib.Path):
        output_format = audio.suffix.lstrip(".").lower()
    output_format = str(output_format).lower()
    if output_format not in EXPORT_FORMATS:
        raise ValueError("Choose WAV, MP3, FLAC, OGG or M4A.")
    folder = pathlib.Path(tempfile.mkdtemp(prefix="inference-", dir=tmp))
    path = folder / (name + "." + output_format)
    try:
        if isinstance(audio, pathlib.Path):
            shutil.copyfile(audio, path)
        else:
            rate, samples = audio
            write_inference_export(path, samples, rate)
    except (ValueError, OSError, RuntimeError):
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return gr.update(value=str(path), label=name)


def convert_named_inference(*args):
    conversion_args, export_controls = args[:12], args[12:]
    output_format, *controls = export_controls or ["wav"]
    try:
        options = inference_options_from_controls(*controls)
    except ValueError as error:
        return str(error), gr.update(value=None)
    with voice_lock:
        message, audio = vc.vc_single(*conversion_args, inference_options=options)
        try:
            return message, named_inference_audio(audio, loaded_voice, output_format)
        except (ValueError, OSError, RuntimeError) as error:
            return "Export failed. " + str(error), gr.update(value=None)


AUTO_INFERENCE_EMBEDDER = "Auto (model default)"


def inference_options_from_controls(
    split_audio=False, autotune=False, autotune_strength=1.0,
    proposed_pitch=False, target_pitch=155, clean_audio=False, clean_strength=0.5,
    formant=False, formant_ratio=1.0, post_process=False, effects=None,
    post_pitch=0, gain_db=0, limiter_db=-6,
    embedder_model=AUTO_INFERENCE_EMBEDDER, custom_embedder=None,
):
    if embedder_model == "custom":
        if not custom_embedder:
            raise ValueError("Select or install a custom embedder first.")
        embedder_model = "custom:" + custom_embedder
    return {
        "embedder_model": None if embedder_model == AUTO_INFERENCE_EMBEDDER else embedder_model,
        "split_audio": split_audio,
        "autotune_strength": autotune_strength if autotune else None,
        "proposed_pitch_hz": target_pitch if proposed_pitch else None,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "formant_ratio": formant_ratio if formant else None,
        "post_process": post_process,
        "effects": effects or [],
        "post_pitch": post_pitch,
        "gain_db": gain_db,
        "limiter_db": limiter_db,
    }


def build_inference_embedder_controls(elem_id):
    from ilaria_embedders import INFERENCE_EMBEDDERS, custom_embedder_names, install_custom_embedder

    def refresh_custom(selected):
        names = custom_embedder_names(now_dir)
        return gr.update(choices=names, value=selected if selected in names else None)

    def save_custom(folder, weights, configuration):
        if not weights or not configuration:
            return "Choose both the .bin weights and the .json configuration.", gr.update()
        for file, suffix in ((weights, ".bin"), (configuration, ".json")):
            filename = getattr(file, "orig_name", file.name)
            if pathlib.Path(filename).suffix.casefold() != suffix:
                return "Choose a %s file for that input." % suffix, gr.update()
        try:
            installed = install_custom_embedder(now_dir, folder, weights.name, configuration.name)
        except (ValueError, OSError, RuntimeError) as error:
            return str(error), gr.update()
        return "Installed custom embedder: " + installed, refresh_custom(installed)

    with gr.Accordion("Embedder Settings", open=False, elem_id=elem_id):
        embedder = gr.Radio(
            [AUTO_INFERENCE_EMBEDDER, *INFERENCE_EMBEDDERS, "custom"],
            value=AUTO_INFERENCE_EMBEDDER, label="Embedder Model", interactive=True,
        )
        gr.Markdown(
            "Auto uses the embedder saved with the voice model. "
            "If you choose a different embedder, set Index Ratio to zero unless you have a matching index. "
            "Chinese, Japanese and Korean HuBERT Base require an RVC v2 voice model."
        )
        with gr.Column(visible=False) as custom_controls:
            with gr.Accordion("Custom Embedder", open=True):
                with gr.Row():
                    custom = gr.Dropdown(
                        choices=custom_embedder_names(now_dir),
                        label="Select Custom Embedder", interactive=True,
                    )
                    refresh = gr.Button("Refresh embedders")
                folder = gr.Textbox(label="Folder Name", interactive=True)
                with gr.Row():
                    weights = gr.File(label="Upload .bin", file_types=["file"], interactive=True)
                    configuration = gr.File(label="Upload .json", file_types=["file"], interactive=True)
                install = gr.Button("Install custom embedder", variant="primary")
                status = gr.Textbox(label="Embedder Import Status", interactive=False)
        embedder.change(lambda selected: gr.update(visible=selected == "custom"),
                        [embedder], [custom_controls], queue=False, show_progress=False)
        refresh.click(refresh_custom, [custom], [custom], queue=False, show_progress=False)
        install.click(update_operation_gate.track(save_custom, "embedder installation"),
                      [folder, weights, configuration], [status, custom],
                      api_name="install_custom_embedder")
    return [embedder, custom]


def build_inference_audio_controls(elem_id):
    from ilaria_inference_effects import POST_EFFECTS

    with gr.Accordion("Audio Processing", open=False, elem_id=elem_id):
        split = gr.Checkbox(label="Split Audio", value=False)
        tune = gr.Checkbox(label="Autotune", value=False)
        tune_strength = gr.Slider(0, 1, value=1, step=0.05, label="Autotune Strength", visible=False)
        proposed = gr.Checkbox(label="Proposed Pitch", value=False)
        target = gr.Slider(50, 1200, value=155, step=1, label="Target Pitch (Hz)", visible=False)
        clean = gr.Checkbox(label="Clean Audio", value=False)
        clean_strength = gr.Slider(0, 1, value=0.5, step=0.05, label="Clean Strength", visible=False)
        formant = gr.Checkbox(label="Formant Shifting", value=False)
        formant_ratio = gr.Slider(0.7, 1.4, value=1.0, step=0.01, label="Formant Ratio", visible=False)
        post = gr.Checkbox(label="Post-Process", value=False)
        with gr.Column(visible=False) as post_controls:
            effects = gr.CheckboxGroup(
                list(POST_EFFECTS), value=[], label="Effects",
            )
            with gr.Row():
                post_pitch = gr.Slider(-12, 12, value=0, step=1, label="Post Pitch Shift (semitones)")
                gain = gr.Slider(-24, 24, value=0, step=0.5, label="Gain (dB)")
                limiter = gr.Slider(-24, 0, value=-6, step=0.5, label="Limiter Threshold (dBFS)")
        for toggle, controls in ((tune, tune_strength), (proposed, target), (clean, clean_strength),
                                 (formant, formant_ratio), (post, post_controls)):
            toggle.change(lambda enabled: gr.update(visible=enabled), [toggle], [controls],
                          queue=False, show_progress=False)
    return [split, tune, tune_strength, proposed, target, clean, clean_strength,
            formant, formant_ratio, post, effects, post_pitch, gain, limiter]


def select_voice_model_with_tts(sid, protect_single, protect_batch):
    result = select_voice_model(sid, protect_single, protect_batch)
    speaker = dict(result[0])
    speaker["value"] = 0
    return (*result, speaker, dict(result[3]))

ILARIARVC_CSS += '\n#dataset-check-report table th, #dataset-check-report table th * {\n    color: #FFFFFF !important;\n    background-color: #62416B !important;\n}\n#dataset-check-report table td, #dataset-check-report table td * {\n    color: #F8EFFA !important;\n}\n#dataset-check-report table th, #dataset-check-report table td {\n    border-color: #C59ACD !important;\n}\n'

ILARIARVC_CSS += '\n/* Consistent readable panels across every application tab. */\n.gradio-container #ilaria-main-tabs .gr-panel,\n.gradio-container #ilaria-main-tabs .gr-form,\n.gradio-container #ilaria-main-tabs .gr-box,\n.gradio-container #ilaria-main-tabs div:has(> .label-wrap) {\n    background: #62416B !important;\n    border: 1px solid #D8B4FE !important;\n    border-radius: 12px !important;\n    color: #F8EFFA !important;\n}\n.gradio-container #ilaria-main-tabs div:has(> .label-wrap) > .label-wrap {\n    background: #62416B !important;\n    color: #FFFFFF !important;\n    border-color: #D8B4FE !important;\n}\n.gradio-container #ilaria-main-tabs textarea,\n.gradio-container #ilaria-main-tabs textarea.gr-box,\n.gradio-container #ilaria-main-tabs input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),\n.gradio-container #ilaria-main-tabs select,\n.gradio-container #ilaria-main-tabs .gr-prose th,\n.gradio-container #ilaria-main-tabs .gr-prose td {\n    background: #62416B !important;\n    color: #FFFFFF !important;\n    border-color: #D8B4FE !important;\n}\n.gradio-container #ilaria-main-tabs input::placeholder,\n.gradio-container #ilaria-main-tabs textarea::placeholder { color: #DECBE4 !important; }\n.gradio-container #ilaria-main-tabs .gr-prose th *,\n.gradio-container #ilaria-main-tabs .gr-prose td * { color: #F8EFFA !important; }\n'

ILARIARVC_CSS += '\n/* Explicit accordion targets for Gradio 3.14, including nested panels. */\n.gradio-container #ilaria-main-tabs #tts-import-text,\n.gradio-container #ilaria-main-tabs #tts-speech-settings,\n.gradio-container #ilaria-main-tabs #tts-rvc-settings,\n.gradio-container #ilaria-main-tabs #training-settings #dataset-checker,\n.gradio-container #ilaria-main-tabs #training-settings #training-previews,\n.gradio-container #ilaria-main-tabs #tutorial-content [id^="glossary-section-"] {\n    background: #62416B !important;\n    border: 2px solid #D8B4FE !important;\n    border-radius: 12px !important;\n    color: #FFFFFF !important;\n    box-shadow: none !important;\n}\n.gradio-container #ilaria-main-tabs #tts-import-text > *,\n.gradio-container #ilaria-main-tabs #tts-speech-settings > *,\n.gradio-container #ilaria-main-tabs #tts-rvc-settings > *,\n.gradio-container #ilaria-main-tabs #dataset-checker > *,\n.gradio-container #ilaria-main-tabs #training-previews > *,\n.gradio-container #ilaria-main-tabs #tutorial-content [id^="glossary-section-"] > * {\n    background-color: #62416B !important;\n    border-color: #D8B4FE !important;\n    color: #FFFFFF !important;\n}\n.gradio-container #ilaria-main-tabs #dataset-checker > button *,\n.gradio-container #ilaria-main-tabs #training-previews > button *,\n.gradio-container #ilaria-main-tabs #tutorial-content [id^="glossary-section-"] > button * {\n    color: #FFFFFF !important;\n}\n'

ILARIARVC_CSS += '\n/* Spacing for collapsible settings and their controls across all tabs. */\n.gradio-container #ilaria-main-tabs .gr-accordion,\n.gradio-container #ilaria-main-tabs #training-settings,\n.gradio-container #ilaria-main-tabs #dataset-checker,\n.gradio-container #ilaria-main-tabs #training-previews,\n.gradio-container #ilaria-main-tabs #inference-settings,\n.gradio-container #ilaria-main-tabs #batch-inference-settings,\n.gradio-container #ilaria-main-tabs #program-updates,\n.gradio-container #ilaria-main-tabs #merge-models,\n.gradio-container #ilaria-main-tabs #download-models,\n.gradio-container #ilaria-main-tabs #import-models,\n.gradio-container #ilaria-main-tabs #model-information,\n.gradio-container #ilaria-main-tabs [id^="glossary-section-"] {\n    margin-top: 8px !important;\n    margin-bottom: 8px !important;\n}\n.gradio-container #ilaria-main-tabs .gr-row,\n.gradio-container #ilaria-main-tabs .gr-column,\n.gradio-container #ilaria-main-tabs .gr-form {\n    gap: 14px !important;\n}\n.gradio-container #ilaria-main-tabs #training-settings,\n.gradio-container #ilaria-main-tabs #dataset-checker,\n.gradio-container #ilaria-main-tabs #training-previews,\n.gradio-container #ilaria-main-tabs #inference-settings,\n.gradio-container #ilaria-main-tabs #batch-inference-settings,\n.gradio-container #ilaria-main-tabs #program-updates {\n    padding: 12px !important;\n}\n'

ILARIARVC_CSS += '\n/* PC advice follows the same clear panels and spacing as the other tools. */\n.gradio-container #ilaria-main-tabs #other-tab #training-advisor {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n    margin: 8px 0 !important;\n    padding: 12px !important;\n}\n.gradio-container #ilaria-main-tabs #training-advisor > * {\n    border-color: #D8B4FE !important;\n    color: #FFFFFF !important;\n}\n#training-advisor-help { padding: 12px 18px !important; text-align: center; }\n#training-advisor-hardware, #training-advisor-recommendations {\n    padding: 12px 18px !important; overflow-x: auto; color: #F8EFFA !important;\n}\n#training-advisor table { width: 100%; border-collapse: collapse; text-align: left; }\n#training-advisor th, #training-advisor td {\n    color: #FFFFFF !important; border: 1px solid #C59ACD !important; padding: 8px 12px;\n}\n#training-advisor a { color: #FFD5E4 !important; text-decoration: underline; }\n'

TUTORIAL_SECTIONS += (('22. PC training advisor: find a sensible starting point', "Open Other > PC training advisor, below Model information and above Update Ilaria RVC. This section starts closed. Choose the sample rate and precision you want to evaluate (32k and FP32 by default), then click Scan PC and suggest settings. The scan reads this PC's CPU, RAM, free space, PyTorch/CUDA runtime and visible GPUs. It does not upload hardware information, install drivers, run training or change any Training settings.\n\nThe hardware report contains measured facts and a PC-local scan timestamp. Total VRAM is the card's capacity; free VRAM is what is available at that moment, so Inference models and other programs can change it. If free memory cannot be read, the report says Unknown and the batch estimate uses a conservative assumption. With multiple GPUs, Auto chooses the largest compatible visible card. Select a specific GPU to evaluate it individually. Memory from separate cards is not added together. The hardware table uses visible scan IDs. The Training GPU ID advice accounts for numeric CUDA visibility mappings used by the trainer; custom UUID mappings must be confirmed manually. Windows Device Manager can use different numbering.\n\nThe settings table gives conservative starting estimates, not official maximum batches or a guarantee that training will fit. Changing the advisor's sample rate, precision or GPU recalculates the table from the existing snapshot; click Scan again for fresh free-memory figures. A higher sample rate or less free VRAM lowers the suggested batch. FP32 remains the default. GTX 16-series and Pascal SM 6.1 cards are advised to use FP32 even if you ask to evaluate FP16. Mixed precision can help some modern GPUs, but must be compared on your actual dataset. A GPU can be detected even when this older runtime cannot train on its architecture; read the runtime check before treating a suggestion as usable.\n\nCopy suitable values manually into Training > Training Settings & Other for a new experiment. Always use compatible pretrained G/D weights and prepared data for the chosen sample rate and pitch mode. If CUDA runs out of memory, halve Batch Size down to 1, keep Cache data to GPU set to No, free VRAM and check again. The suggested CPU Threads control concerns preparation; it does not change the trainer's separate DataLoader workers. Low available RAM lowers the suggested preparation thread count.\n\nTotal Epochs 300 and Save frequency 100 are Ilaria's personal baseline, not values computed from your GPU. Her usual dataset is about 35 minutes of clean expressive speech. Your recordings, voice, dataset variety and preferred result matter. Compare held-out conversions or Training previews; more epochs or a larger batch do not automatically improve quality. Lower Save frequency if you need more recovery points, accounting for save time and disk space.\n\nWhen resuming a saved experiment, preserve its original compatible data, batch size, GPU count and architecture. A fresh hardware suggestion is not permission to change an existing checkpoint's setup. This advisor cannot predict final quality, the best epoch or a precise training time from hardware alone. The report links the official RVC and PyTorch references behind the advice. Numeric batch/thread tiers are this app's own rules, clearly marked as estimates."),)

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #training-settings #training-notifications {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n    margin: 8px 0 !important;\n    padding: 12px !important;\n}\n.gradio-container #ilaria-main-tabs #training-notifications > * {\n    color: #FFFFFF !important;\n    border-color: #D8B4FE !important;\n}\n#training-notifications .gr-prose { padding: 10px 14px !important; }\n'
TUTORIAL_SECTIONS += (('23. Training notifications: hear when a run finishes', "Open Training > Training Settings & Other > Training notifications. Desktop alerts and Sound alerts are both enabled by default. You can turn either one off independently before starting Train Pth. The options are captured for that training job: changing the checkboxes while a job is running affects the next job, not the current one.\n\nClick Test alerts to request a test using the currently selected checkboxes. A desktop alert is a local Windows notification with Ilaria's artwork; a sound alert uses the PC's Windows sound settings. This does not require browser notification permission, an account or an internet service. Test alerts does not start training or change your model. If both options are off, the test explains that nothing was requested.\n\nTraining completed is sent only after the trainer exits successfully and its log confirms the final epoch and successful final model export. Training failed is sent if the training process exits with an error, including an out-of-memory failure before the first completed epoch, or if a final export cannot be confirmed. The notification identifies the model; read Training Log and logs/<Model Name>/training_console.log for the actual error details. A validation message that prevents a training process from starting is shown in the UI instead of producing repeated desktop alerts.\n\nStop and save is an intentional pause. It does not trigger a completed or failed alert. The notification also does not run for Train Index, Process Data, Feature Extraction, previews, import or merging; it concerns Train Pth. There is one final alert per training job, not one alert per epoch or per GPU worker.\n\nYou can minimize or close the browser tab and still receive alerts because monitoring happens in the local backend, not JavaScript in the page. Keep Ilaria RVC and its CMD/backend running. Terminating the program, logging out, sleeping the PC or shutting it down can prevent delivery; notifications are not a way to keep training running after the program is closed.\n\nWindows notification settings, Do not disturb/Focus assist, quiet time, disabled system sounds and the current output device can hide or silence an alert. A test request being accepted does not prove that Windows displayed it. Check the selected options and Windows settings if you do not see or hear the test. The helper does not change system volume, notification permissions or the power plan. It registers only the Ilaria RVC notification name and icon for the current Windows user; administrator permissions are not required. Delivery runs in a short-lived background process, so it does not slow down epoch computation. No audio recordings, models or logs are sent to an external service."),)

TUTORIAL_SECTIONS += (('24. Upload a dataset by drag and drop', 'Open Training Settings & Other and drop audio files or a ZIP in the box beside Path to Dataset. Import starts automatically. A ZIP may contain subfolders: supported audio files are placed directly in datasets, without creating a subfolder. Other ZIP contents are ignored. Your original recordings are kept. Duplicate filenames are renamed instead of overwriting another recording. All uploaded recordings are placed directly in datasets. Existing files are kept; duplicate filenames receive a number suffix. Process Data uses every audio file directly in this folder, so remove unrelated recordings before preprocessing. Wait for Dataset import done! before Process Data. Path to Dataset automatically points to the new folder, so the checker and Process Data use the uploaded recordings. Importing does not run preprocessing or train a model. All four training steps show a short error if Path to Dataset is missing, unreadable or contains no nonempty supported audio files directly inside it. If an import fails, Path to Dataset stays unchanged and incomplete output is removed. For a complete folder, use a ZIP or select its audio files; this upload box accepts files rather than a folder object. A batch is limited to 5,000 supported audio files and 16 GiB. Check the Dataset Import Log for progress, then use Dataset checker to review recording quality. Open Manage datasets, click Refresh datasets, select one item and click Delete selected dataset to permanently remove it from datasets. Original recordings elsewhere, models and checkpoints are kept. Deletion is blocked while operations are running; Path to Dataset resets if its selected dataset is removed.'),)

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #program-updates #update-all-changes {\n    background: #62416B !important; border: 2px solid #D8B4FE !important;\n    border-radius: 12px !important; margin: 10px 0 !important; padding: 12px !important;\n    color: #FFFFFF !important;\n}\n#all-changes-content { padding: 12px 18px !important; text-align: left; }\n#all-changes-content h3 { color: #FFC0CB !important; margin-top: 18px; }\n#all-changes-content p, #all-changes-content li { color: #F8EFFA !important; line-height: 1.65; }\n#all-changes-content li { margin: 8px 0; }\n'

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #training-settings #manage-datasets {\n    border: 2px solid #D8B4FE !important;\n    background: #62416B !important;\n    border-radius: 12px !important;\n    margin: 8px 0 !important;\n    padding: 12px !important;\n}\n.gradio-container #ilaria-main-tabs #manage-datasets > * {\n    border-color: #D8B4FE !important;\n    color: #FFFFFF !important;\n}\n'

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #dataset-upload-help {\n    box-sizing: border-box !important;\n    padding: 12px 18px !important;\n    text-align: center !important;\n    line-height: 1.65 !important;\n}\n#dataset-upload-help p {\n    text-align: center !important;\n    margin: 0 !important;\n}\n'

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #training-queue {\n    border: 2px solid #D8B4FE !important; background: #62416B !important;\n    border-radius: 12px !important; margin: 8px 0 !important; padding: 12px !important;\n}\n#training-queue-help { padding: 12px 18px !important; text-align: center !important; }\n#training-queue-table { padding: 12px 18px !important; overflow-x: auto; }\n#training-queue-table th, #training-queue-table td { color: #FFFFFF !important; border: 1px solid #D8B4FE !important; padding: 8px 12px; }\n'
TUTORIAL_SECTIONS += (('25. Training queue: run several models one after another', "Open Training > Training queue, below the four workflow steps. Choose a unique Model Name, Path to Dataset and all desired training/preview/notification settings above, then click Add current settings to queue. Wait for the confirmation before changing settings for the next model. Add each model separately. The queue saves a separate copy of its source audio and optional preview clip, so later uploads or changed controls cannot silently replace an earlier job's input. These copies need additional disk space.\n\nAutomatically run Process Data, Feature Extraction and Train Index is on by default. Each new job runs those three steps before Train Pth. Turn it off when that model already has suitable prepared data and you only want PTH training. Existing managed checkpoints are resumed using their original prepared files; preparation is skipped on resume to protect saved training compatibility. Keep batch size, GPU count, architecture and prepared data compatible with the checkpoint. Total Epochs is the final target, not additional epochs.\n\nClick Start queue after current operations finish. Only one queued model runs at a time. The table and Queue Log update while the page is open. Closing the browser does not stop the queue; keep Ilaria RVC's backend/CMD open. Manual training steps are blocked while the queue runs, and the updater/dataset deletion waits until it is idle. You may add another differently named job while the current model is training.\n\nPause after current model lets the entire current job finish, then leaves the remaining jobs pending. To stop PTH training at its next epoch boundary instead, use the existing Stop and save button; the queue pauses when that save is confirmed. An error also pauses the queue, rather than silently starting more jobs. Select the failed, stopped or interrupted job and choose Retry selected after fixing the cause, then Start queue again. A stopped model continues from its saved checkpoint. Missing or incompatible training state is an error, not a fresh start disguised as a resume.\n\nRemove selected job removes a nonrunning job from the list. Running jobs cannot be removed. Queue settings and source snapshots are stored locally under .ilaria-queue. On a backend restart, a previously running job is marked Interrupted and is not restarted automatically; review its log before retrying. Existing trained models and experiment checkpoints stay in their normal folders. This queue does not predict which settings or epoch will sound best; compare previews as usual."),)

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #other-tab #model-export {\n    border: 2px solid #D8B4FE !important; background: #62416B !important;\n    border-radius: 12px !important; margin: 8px 0 !important; padding: 12px !important;\n}\n#model-export-help { padding: 12px 18px !important; text-align: center !important; color: #F8EFFA !important; }\n'
TUTORIAL_SECTIONS += (('26. Export a model ZIP to share', 'Open Other > Export model ZIP, below PC training advisor. Select the installed .pth under Voice Model. If one likely matching retrieval index is found, it is selected automatically; check its name. If several possible indexes exist, choose the correct one yourself. File names can suggest a match but cannot prove that a model and index were created together.\n\nClick Refresh model list if you recently trained or imported a model. Select an index from models if you have its matching retrieval file. Leave Index empty to package only the .pth, for example for a merged model without a matching index. An index alone is not a usable voice model.\n\nClick Create model ZIP and watch Export Log. When it reaches 100%, click the file under Download model ZIP to save the ZIP on your PC. It contains only the selected .pth and optional .index, directly inside the archive. It excludes dataset audio, logs, resumable checkpoints, previews, configuration and runtime files. Original files stay where they are. The recipient can use Other > Import local models to install the ZIP, then Refresh in Inference and select the voice.\n\nThe ZIP is saved temporarily inside TEMP/model-exports and is not uploaded anywhere. Download it before closing or restarting Ilaria RVC, because temporary files may be cleaned at startup. Packaging requires free disk space roughly equal to the selected files; it uses a fast ZIP method rather than promising a smaller download. Avoid exporting while the selected model or index is being saved. A changed or missing source file produces an error instead of a finished download. The tool packages files without loading neural weights or testing voice quality. ZIP percentages describe copied bytes, not training progress.'),)

TUTORIAL_SECTIONS += (('27. Read index information', 'Index information is included when matching files are found under models. Each index shows its path, file size, local modification time, SHA-256 fingerprint, FAISS type, feature dimension, stored vector count, training status, distance metric and IVF cluster/search settings when available. A feature-dimension mismatch is flagged. Matching dimensions and filenames do not prove that an index belongs to the selected voice. Vectors are audio features, not a count of recordings or a dataset duration. Index files do not reliably store sample rate or model epochs. Unreadable indexes are reported without hiding the model information.'),)

TUTORIAL_SECTIONS += (('28. Vocoder, embedder, silence, pretrained models and training targets', 'Vocoder chooses the waveform decoder. HiFi-GAN keeps the original RVC decoder and supports the existing settings. RefineGAN is an alternative decoder: in this build it requires RVC v2, 32 kHz and Pitch Guidance enabled. It uses a different discriminator and mel objective. Select a compatible RefineGAN pretrained G/D pair or explicitly choose None to train from scratch. An incompatible pretrained model is rejected; it cannot be converted simply by renaming it. RefineGAN is not a promise of better quality or faster training. Use a new Model Name when changing vocoders.\n\nEmbedder Model chooses the speech-content representation, not the voice or pitch extractor. contentvec uses the existing hubert_base.pt. spin-v2 downloads its fairseq-compatible weights once (about 361 MiB), checks their SHA-256, and uses them for both feature extraction and inference. The embedder choice is stored in exported models and previews. Never reuse features or an index from another embedder: choose a new Model Name, then Process Data, Feature Extraction and Train Index again. Old models without embedder metadata default to contentvec. Both embedders can have the same feature dimension; equal dimensions do not establish compatibility.\n\nSilent training files adds silent examples to the training file list. Default 2; choose 0 to 10. These are repeated silence entries, not recordings added to your dataset folder. spin-v2 silence features are extracted with spin-v2 as well. Changing the silent count changes the training data and is rejected when resuming an existing incompatible checkpoint.\n\nTraining Target offers two alternatives. Epochs stops after Total Epochs (default 300). Steps stops after the cumulative Total Steps (default 10,000), including partway through an epoch. One step is one generator/discriminator batch update per training rank, not one audio file. Resuming uses the cumulative count: a target of 10,000 means finish at 10,000, not train 10,000 more. A saved partial epoch resumes at the correct remaining batch. Cache data to GPU must be No in Steps mode and when resuming a partial epoch. Save frequency and preview frequency still count complete epochs; reaching the final step always saves and exports. Step progress and ETA use measured step time, while completed epochs remain accurately counted.\n\nPretrained model is one menu: choose Auto, None or a locally installed complete G/D pair. Auto selects the original pretrained for the current sample rate, pitch setting and RVC version; it does not provide a default RefineGAN pretrained. None explicitly starts without a pretrained. Refresh pretrained list scans installed pairs under assets; it does not download them. Both G and D must exist and match the selected architecture. Matching file names alone do not prove compatibility; actual tensor shapes are checked by the trainer. A saved model checkpoint takes priority when resuming; switching pretrains does not reset that model.\n\nIndex Algorithm affects index creation, not the neural PTH trainer. Auto keeps the existing large-dataset policy: above 200,000 feature vectors it tries MiniBatchKMeans with up to 10,000 centers before FAISS indexing. Faiss builds an IVF Flat index using all extracted rows, without that reduction. KMeans explicitly reduces sufficiently large feature sets to representative centers before building the FAISS index; tiny sets are kept rather than clustered into invalid sizes. Clustering can reduce index size and index-building cost, but can lose detail and does not guarantee better conversion. KMeans is useful to consider for large datasets. All three ultimately produce a FAISS .index file. The queue saves these choices independently for every job.'),)

ILARIARVC_CSS += '\n#training-options-help { padding: 16px 22px !important; margin: 10px 0 !important; box-sizing: border-box; text-align: center !important; line-height: 1.65 !important; }\n'

TUTORIAL_SECTIONS += (('29. Additional pitch extraction algorithms', 'RMVPE remains the default pitch extractor and is a good starting point. FCPE estimates pitch using TorchFCPE. Swift and Beatrice use additional models adapted from Redpanda-vc. Swift uses SwiftF0 0.1.2 on CPU for compatibility with the bundled ONNX Runtime; it does not require replacing your CUDA libraries. Beatrice uses its own pitch estimator and runs on the selected inference device, or CPU during training extraction. Swift and Beatrice download their model files once if missing. FCPE uses its installed model. Parselmouth is the full name for the existing pm autocorrelation method, not a different voice model. These choices are available for training features, single and batch inference, and training previews. Keep RMVPE for an initial comparison, then listen to the same clip with other algorithms. Different extractors can make different octave or voicing errors; no method guarantees the best result for every recording.'),)

ILARIARVC_CSS += "\n#pitch-algorithm-help { padding: 12px 22px !important; text-align: center; line-height: 1.6; }\n"
TUTORIAL_SECTIONS += (('30. Why training controls change or disappear', 'RefineGAN restricts Sample Rate to 32 kHz and locks Pitch Guidance on. HiFi-GAN restores supported sample rates. Turning Pitch Guidance off hides pitch extraction and disables RefineGAN until pitch is enabled again. GPU pitch process settings appear only for rmvpe_gpu on supported CUDA hardware.\n\nEpochs and Steps show only their corresponding target slider. Steps sets Cache data to GPU to No and disables it; returning to Epochs unlocks the choice but leaves No selected. PCs without CUDA offer only FP32 and disable CUDA training controls.\n\nPretrained model is a single selector for a complete generator/discriminator pair. Auto and installed pairs show their resolved files in Pretrained Log; None explicitly starts from scratch. RefineGAN cannot use the original Auto pretrained: select a compatible installed pair or explicitly choose None. There are no separate G/D path boxes. The trainer still checks actual compatibility; filenames alone are not proof.\n\nPreview generation controls appear only when Training previews is enabled. Saved history, audio and logs remain accessible when it is off. Automatic UI restrictions do not convert old features or checkpoints. Use a new Model Name for incompatible architecture/embedder changes. Saved queue jobs retain their settings and are checked by the backend.'),)

TUTORIAL_SECTIONS += (('31. One pretrained selector for both files', 'Choose Pretrained model in Training Settings & Other. Every installed entry selects a complete generator (G) and discriminator (D) pair together. The two file paths are handled internally, not entered in separate boxes. Pretrained Log confirms which files will be used. Auto selects the original pair for the current RVC version, sample rate and pitch setting; None explicitly trains without a pretrained. RefineGAN requires its own compatible pair and does not offer the original Auto pretrained.\n\nFor custom pairs, put both .pth files together inside assets/pretrained_custom/your_pretrained_name, then click Refresh pretrained list. Supported names include G.pth/D.pth, G_name.pth/D_name.pth, f0G32k.pth/f0D32k.pth, generator.pth/discriminator.pth, and name_G.pth/name_D.pth. An incomplete pair is not listed. You can also use the pretrained downloader in Other. Voice model .pth exports belong in models and are not generator/discriminator pretrained pairs.\n\nOriginal pretrained entries are filtered by version, sample rate and Pitch Guidance. Custom pairs are listed by folder and filenames; their tensor architecture is still checked by the trainer. A filename alone cannot prove vocoder compatibility. Adding a queued job resolves both paths immediately and keeps that captured pair; changing the dropdown later does not change existing jobs. Keep the referenced files installed until those jobs finish.'),)

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #pretrained-picker { padding: 12px !important; margin: 10px 0 !important; border: 2px solid #D8B4FE !important; border-radius: 12px !important; background: #62416B !important; gap: 8px !important; }\n'

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #other-tab #pretrained-download {\n    border: 2px solid #D8B4FE !important; background: #62416B !important;\n    border-radius: 12px !important; margin: 8px 0 !important; padding: 12px !important;\n}\n#pretrained-download-help { padding: 12px 20px !important; line-height: 1.65 !important; text-align: center !important; color: #F8EFFA !important; }\n'
TUTORIAL_SECTIONS += (('32. Download pretrained pairs', 'Open Other > Download pretrained models, directly below the voice-model downloader. This tool downloads generator/discriminator pretrained pairs for training, not voice models for inference. Paste a public Hugging Face repository or folder page, a direct pretrained ZIP, or a direct G .pth link. For a direct file, it looks for the matching D beside it. You may paste the second direct link in Matching D .pth link when necessary; keep the two files from the same compatible pair.\n\nClick Download pretrained pair and follow Pretrained Download Log. Download, extraction and installation show progress. If a repository contains several pairs or ZIPs, the log lists direct links: choose the pair you want rather than relying on a random file. ZIPs may contain several complete pairs in separate folders; the tool installs complete G/D pairs and leaves unrelated files out.\n\nFiles go to assets/pretrained_custom/<package>, separate from inference weights and personal training logs. When the log confirms 100%, open Training Settings & Other and click Refresh pretrained list in the compact pretrained block. Select the installed entry: the single dropdown chooses G and D together. Keep vocoder, sample rate, RVC version and pitch settings compatible with the pair. A G/D filename match is not a tensor compatibility test: strict checks still occur when the trainer loads the pair.\n\nExisting different files are kept and reported as a conflict. Failed or interrupted downloads do not expose half-installed pairs in the menu. A single inference .pth with a .index is not a training pretrained pair and belongs in the voice-model downloader above. The pretrained downloader does not read checkpoint Python objects or promise a particular voice quality. For offline installation, place both files together under assets/pretrained_custom/<name> and refresh the training selector.'),)

ILARIARVC_CSS += '\n.gradio-container #ilaria-main-tabs #recommended-pretraineds { background: #62416B !important; border: 2px solid #D8B4FE !important; border-radius: 12px !important; margin: 12px 0 !important; padding: 12px !important; }\n#recommended-pretrained-help { padding: 12px 20px !important; text-align: center !important; line-height: 1.65 !important; }\n#recommended-pretrained-details { padding: 12px 18px !important; color: #FFFFFF !important; }\n#recommended-pretrained-details th, #recommended-pretrained-details td { color: #FFFFFF !important; border: 1px solid #D8B4FE !important; padding: 8px 12px !important; }\n'
TUTORIAL_SECTIONS += (('33. Recommended pretrained downloads', 'Other > Download pretrained models ends with Recommended pretraineds, closed by default. The alphabetical list contains all supplied families and 35 sample-rate/vocoder/embedder variants. Choose one entry, check Sample rate and the G/D source links, then click Download selected pretrained (G + D). Both files install together through the same pretrained downloader. Open Training, refresh the pretrained list and choose the installed pair. No models download merely by opening this menu. Nanashi v1 is a pretrained family name, not a request to switch RVC architecture. RefineGAN variants at 40 or 44.1 kHz can be downloaded, but this build cannot train those RefineGAN sample rates. Keep their actual rate; do not rename a 40/44.1 kHz pretrained as 32 kHz. Unknown publisher metadata is shown as Not specified instead of guessed. Recommendations are the supplied catalog, not a quality ranking.'),)

TUTORIAL_SECTIONS += ((
    "34. Text to speech",
    "Open **Inference > TTS**. Type up to 10,000 characters or import a UTF-8 .txt file. "
    "Choose a speech engine and a source voice. The full Edge voice "
    "catalog is available immediately. Filter by language or use Refresh voices to update it. Edge sends the text to Microsoft. "
    "Windows voices work offline and depend on the desktop voices installed on this PC. "
    "Speech settings control source rate and volume. Source pitch in Hz is available on Edge. "
    "To apply an RVC voice, select the Voice Model above and enable RVC conversion. "
    "RVC settings include pitch in semitones, pitch extraction, index ratio and audio options. "
    "Listen to the original and converted clips separately. Use Download in the converted audio player to save a WAV."
),)


ILARIARVC_CSS += r"""
.ilaria-waveform {
    width: calc(100% - 16px);
    box-sizing: border-box;
    padding: 8px 12px 6px;
    margin: 5px 8px 8px;
    border: 1px solid #c698ce;
    border-radius: 12px;
    background: linear-gradient(120deg, #4b2d52, #39243f);
}
.gr-block > .ilaria-waveform {
    margin-top: 30px;
}
.ilaria-waveform-track {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
}
.ilaria-waveform-track:has(.ilaria-waveform-seek:focus-visible) {
    outline: 2px solid #ffc0cb;
    outline-offset: 2px;
}
.ilaria-waveform-track svg {
    width: 100%;
    height: 58px;
    display: block;
}
.ilaria-waveform-track .ilaria-waveform-seek {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    opacity: 0;
    cursor: ew-resize;
}
.ilaria-waveform-time {
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: space-between;
    padding-top: 4px;
    color: #f6d8ef;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    line-height: 1.3;
}
audio.ilaria-waveform-native {
    display: none !important;
}
.ilaria-waveform-play, .ilaria-waveform-save {
    padding: 4px 10px;
    border-radius: 8px;
    color: #3b233e !important;
    background: #ffc0cb !important;
    font-weight: 600;
    text-decoration: none;
}

"""

ILARIARVC_CSS += """
.gradio-container #tts-tab #tts-intro {
    box-sizing: border-box;
    padding: 14px 18px !important;
}
"""

from ilaria_audio_analyzer_ui import build_audio_analyzer

TUTORIAL_SECTIONS += ((
    "35. Audio Analyzer",
    "Open **Other > Audio Analyzer**, below Export model ZIP. Upload up to eight audio files "
    "and click Analyze audio. File comparison shows source codec, duration and levels. "
    "The detailed view shows the waveform, spectrogram, average spectrum and optional pitch "
    "estimate for the first file that supports the chosen view. The original audio stays intact. "
    "Use Analysis settings to choose a time range, FFT window, frequency scale and channel view. "
    "Each file is limited to 180 seconds or 12 million samples across channels, whichever is reached first. "
    "The player previews up to 30 seconds from the chosen range. Pitch examines up to 60 seconds. "
    "Silence uses 20 ms RMS windows. Near-full-scale counts samples above the chosen threshold, "
    "which can help spot potential clipping. For stereo, negative correlation can indicate "
    "cancellation when summed to mono. Choose CSV for the comparison table, JSON for detailed "
    "measurements, or ZIP for both reports and the plot."
),)

ILARIARVC_CSS += """
.gradio-container #ilaria-main-tabs #other-tab #audio-analyzer,
.gradio-container #ilaria-main-tabs #other-tab #audio-analyzer-settings {
    border: 2px solid #D8B4FE !important;
    background: #62416B !important;
    border-radius: 12px !important;
    margin: 8px 0 !important;
    padding: 12px !important;
}
.gradio-container #ilaria-main-tabs #other-tab #audio-analyzer > *,
.gradio-container #ilaria-main-tabs #other-tab #audio-analyzer-settings > * {
    background-color: #62416B !important;
    border-color: #D8B4FE !important;
    color: #FFFFFF !important;
}
.gradio-container #audio-analyzer-help,
.gradio-container #audio-analyzer-detail {
    padding: 12px 18px !important;
    color: #F8EFFA !important;
}
"""

from ilaria_model_upload import register_binary_model_import
from ilaria_model_upload_ui import BINARY_MODEL_UPLOAD_JS, build_binary_model_upload


ILARIARVC_CSS += """
#other-tab .ilaria-other-section {
    border-bottom: 1px solid #B48FC4;
    padding: 9px 12px;
    margin: 14px 0 2px;
    color: #F8EFFA;
}
#other-tab .ilaria-other-section h3 {
    color: #FFC0CB;
    font-size: 17px;
    font-weight: 700;
    margin: 0 0 3px;
}
#other-tab .ilaria-other-section p { margin: 0; font-size: 13px; }
"""

from ilaria_release_history import RELEASE_HISTORY

ILARIARVC_CSS += """
#update-all-changes [id^="release-history-"] {
    border: 1px solid #D8B4FE !important;
    border-radius: 10px;
    margin: 6px 0;
    padding: 8px 12px;
}
#update-all-changes [id^="release-history-"] .prose { padding: 6px 10px; }
"""

ILARIARVC_CSS += """
.gradio-container #single-audio-processing,
.gradio-container #batch-audio-processing,
.gradio-container #single-embedder-settings,
.gradio-container #batch-embedder-settings {
    padding: 12px;
    border: 1px solid #D8B4FE !important;
    border-radius: 12px;
    background: #62416B !important;
    color: #F8EFFA;
}
"""

TUTORIAL_SECTIONS += ((
    "36. Advanced inference and audio export",
    "Single and Batch Inference each have an Export Format selector in Inference Settings. "
    "WAV and FLAC preserve audio quality. MP3, OGG and M4A compress the converted audio. "
    "The output player downloads the selected format. Batch also saves each file to the output folder. "
    "Speaker ID selects a voice slot supported by the loaded model.\n\n"
    "Open **Audio Processing** inside Inference Settings to find the optional controls. "
    "Split Audio adds conversion boundaries at quiet pauses while keeping their timing. "
    "Autotune moves voiced pitch towards the nearest chromatic note. Proposed Pitch adds a "
    "limited shift based on the median voiced pitch and the target in Hz. Both use an F0-guided "
    "model. Clean Audio reduces background noise after conversion. Formant Shifting adjusts "
    "vocal resonance before conversion. A ratio of 1.0 keeps the original resonances.\n\n"
    "Turn on Post-Process and choose one or more effects. Gain, Pitch Shift and Limiter "
    "have adjustable settings. The other effects use preset values."
), (
    "37. Inference embedders",
    "Open **Embedder Settings** below **Audio Processing** in Single or Batch Inference. Auto selects the embedder "
    "stored in the voice model, or ContentVec when the model has no embedder metadata. "
    "Choose a different embedder only when the voice model is compatible with its features. "
    "Set Index Ratio to zero unless your index was built with the same embedder. "
    "To install a custom HuBERT embedder, choose **custom**, enter a folder name, upload its "
    ".bin weights and .json configuration, then click **Install custom embedder**. "
    "The installed embedder appears in the custom selector."
),)

with gr.Blocks(title="♡  ILARIA RVC ♡", css=ILARIARVC_CSS, analytics_enabled=False) as app:
    gr.HTML(ILARIA_HEADER, elem_id="ilaria-header")
    program_update_notice = gr.Markdown("", visible=False, elem_id="program-update-notice")
    with gr.Tabs(elem_id="ilaria-main-tabs"):
        with gr.TabItem(i18n("Training")):
            exp_dir1 = gr.Textbox(label=i18n("Model Name"), value="", placeholder="ilaria")
            with gr.Accordion("Training Settings & Other", open=False, elem_id="training-settings") as training_settings:
                version19 = gr.Radio(
                    label=i18n("Version 2 only here"),
                    choices=["v2"],
                    value="v2",
                    interactive=False,
                    visible=False,
                )
                with gr.Row():
                    sr2 = gr.Radio(
                        label=i18n("Sample Rate"),
                        choices=["32k", "40k", "48k"],
                        value="32k",
                        interactive=True,
                    )
                    if_f0_3 = gr.Radio(
                        label=i18n("Pitch Guidance"),
                        choices=[True, False],
                        value=True,
                        interactive=True,
                    )
                    np7 = gr.Slider(
                        minimum=1,
                        maximum=MAX_CPU_THREADS,
                        step=1,
                        label=i18n("CPU Threads"),
                        value=CPU_THREADS,
                        interactive=True,
                    )
                with gr.Row():
                    training_vocoder = gr.Radio(label="Vocoder", choices=["HiFi-GAN", "RefineGAN"], value="HiFi-GAN", interactive=True)
                    training_embedder = gr.Radio(label="Embedder Model", choices=["contentvec", "spin-v2"], value="contentvec", interactive=True)
                silent_training_files = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Silent training files", interactive=True)
                index_algorithm_control = gr.Radio(label="Index Algorithm", choices=["Auto", "Faiss", "KMeans"], value="Auto", interactive=True)
                gr.Markdown("RefineGAN requires RVC v2, 32 kHz and Pitch Guidance. Use a compatible pretrained pair or choose None to train from scratch. Spin-v2 downloads its embedder once; use a new Model Name when changing embedders. Steps is an alternative to epochs; Save frequency still counts epochs. Cache data to GPU must be No for Steps.", elem_id="training-options-help")
                training_precision = gr.Radio(label="Training Precision", choices=["FP16", "FP32"], value="FP32", interactive=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        trainset_dir4 = gr.Textbox(
                            label=i18n("Path to Dataset"), value="datasets"
                        )
                    with gr.Column(scale=1):
                        dataset_upload = gr.File(label="Drop dataset audio files or a ZIP", file_count="multiple", file_types=["audio", "application", "video"])
                        dataset_upload_log = gr.Textbox(label="Dataset Import Log", lines=2, max_lines=3, interactive=False)
                        gr.Markdown("Imported automatically directly into `datasets`; Path to Dataset updates when ready. For a whole folder, drop a ZIP. Wait for import completion before Process Data.", elem_id="dataset-upload-help")
                        with gr.Accordion("Manage datasets", open=False, elem_id="manage-datasets"):
                            dataset_to_remove = gr.Dropdown(label="Dataset to remove", choices=[], interactive=True)
                            gr.Markdown("Permanently deletes the selected item inside `datasets`. Original recordings outside this folder are kept.")
                            with gr.Row():
                                refresh_datasets_button = gr.Button("Refresh datasets", variant="primary")
                                remove_dataset_button = gr.Button("Delete selected dataset", variant="secondary")
                            dataset_removal_log = gr.Textbox(label="Dataset Removal Log", lines=2, max_lines=3, interactive=False)
                spk_id5 = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=1,
                    label=i18n("Speaker ID"),
                    value=0,
                    interactive=True,
                )
                gr.Markdown("Pitch extraction estimates how high or low the voice sounds. RMVPE is the default and recommended for most cases.", elem_id="pitch-algorithm-help")
                with gr.Row():
                    f0method8 = gr.Radio(
                        label="Pitch extraction algorithm",
                        choices=["rmvpe", "swift", "beatrice", "fcpe", "parselmouth", "harvest", "dio", "rmvpe_gpu"],
                        value="rmvpe",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label=i18n(
                            "0-0-2 runs TWO processes on GPU 0 & ONE on GPU 2 (rmvpe_gpu)"
                        ),
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=False,
                    )
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=1000,
                        step=1,
                        label=i18n("Save frequency"),
                        value=100,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total Epochs"),
                        value=300,
                        interactive=True,
                    )
                    training_total_steps = gr.Slider(minimum=1, maximum=1000000, step=1, label="Total Steps", value=10000, visible=False, interactive=True)
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        label=i18n("Batch Size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                with gr.Row():
                    training_target_mode = gr.Radio(label="Training Target", choices=["Epochs", "Steps"], value="Epochs", interactive=True)
                    if_save_latest13 = gr.Radio(
                        label=i18n("Save last ckpt as final Model"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n("Cache data to GPU"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n("Create model at save frequency"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Group(elem_id="pretrained-picker"):
                    from ilaria_pretrain_selector import pretrain_choices, AUTO
                    with gr.Row():
                        with gr.Column(scale=4, min_width=240):
                            pretrain_selector = gr.Dropdown(label="Pretrained model (G + D)", choices=pretrain_choices(now_dir, "32k", True, "v2", "HiFi-GAN"), value=AUTO, interactive=True)
                        with gr.Column(scale=1, min_width=160):
                            pretrain_refresh = gr.Button("Refresh pretrained list", variant="primary")
                    pretrain_status = gr.Textbox(label="Pretrained Log", lines=2, max_lines=3, interactive=False)
                    pretrained_G14 = gr.State(value="")
                    pretrained_D15 = gr.State(value="")
                with gr.Row():
                    gpus6 = gr.Textbox(
                        label=i18n("GPU ID's ex: 0-2 Uses GPU's 0 & 2"),
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpus16 = gr.Textbox(
                        label=i18n("Enter cards to be used EX: 0-1"),
                        value=gpus,
                        interactive=True,
                    )
                gpu_info9 = gr.Textbox(
                    label=i18n("GPU Information"),
                    value=gpu_info,
                    visible=F0GPUVisible,
                )
                with gr.Accordion("Dataset checker", open=False, elem_id="dataset-checker"):
                    gr.Markdown("Check the folder in **Training Settings & Other > Path to Dataset** before Process Data. Reports possible problems; your recordings are not changed.")
                    with gr.Row():
                        checker_silence_db = gr.Slider(-70, -20, value=-45, step=1, label="Silence threshold (dBFS)")
                        checker_silence_seconds = gr.Slider(0.5, 10, value=2, step=0.5, label="Flag silences longer than (seconds)")
                        checker_volume_db = gr.Slider(3, 18, value=6, step=1, label="Volume difference warning (dB)")
                    dataset_check_button = gr.Button("Check dataset", variant="primary")
                    dataset_check_log = gr.Textbox(label="Dataset Check Log", lines=1, max_lines=2, interactive=False)
                    dataset_check_report = gr.Markdown("", elem_id="dataset-check-report")
                with gr.Accordion("Training previews", open=False, elem_id="training-previews"):
                    gr.Markdown("Enable before starting training. Upload one short test clip (up to 30 seconds) and listen to the same conversion at each interval. CPU previews pause training briefly and use Index Ratio 0.")
                    preview_enabled = gr.Checkbox(label="Generate training previews", value=False)
                    preview_source = gr.Audio(label="Preview source audio (maximum 30 seconds)", source="upload", type="filepath")
                    with gr.Row():
                        preview_every = gr.Slider(1, 100, value=10, step=1, label="Preview every N epochs")
                        preview_pitch = gr.Slider(-24, 24, value=0, step=1, label="Preview Pitch")
                        preview_method = gr.Dropdown(["rmvpe", "swift", "beatrice", "fcpe", "parselmouth", "harvest", "crepe"], value="rmvpe", label="Preview Pitch Extraction")
                    with gr.Row():
                        preview_history = gr.Dropdown(choices=[], label="Saved preview epoch", interactive=True)
                        preview_refresh = gr.Button("Refresh previews", variant="primary")
                    with gr.Row():
                        preview_log = gr.Textbox(label="Preview Log", value="Training previews are optional. Enable above before starting training.", lines=2, max_lines=4, interactive=False)
                        preview_audio = gr.Audio(label="Training Preview Audio", interactive=False)
                with gr.Accordion("Training notifications", open=False, elem_id="training-notifications"):
                    gr.Markdown("Get a Windows desktop alert and a sound when Train Pth completes or fails. Options are captured when you start training; the browser can be minimized or closed while Ilaria RVC stays running.")
                    with gr.Row():
                        notification_desktop = gr.Checkbox(label="Desktop alerts (completion or error)", value=True)
                        notification_sound = gr.Checkbox(label="Sound alerts (completion or error)", value=True)
                    notification_test = gr.Button("Test alerts", variant="primary")
                    notification_test_log = gr.Textbox(label="Notification Test Log", value="Click Test alerts to check the selected options. Windows notification and sound settings also apply.", lines=2, max_lines=3, interactive=False)
                    notification_test.click(test_training_alerts, [notification_desktop, notification_sound],
                        [notification_test_log], show_progress=False, api_name="test_training_alerts")
                with gr.Accordion("Training queue", open=False, elem_id="training-queue"):
                    gr.Markdown("Set a Model Name, dataset and training options above, then Add current settings. Each job saves its own dataset copy. Start queue runs jobs in order, one model at a time. Keep the backend open.", elem_id="training-queue-help")
                    queue_auto_prepare = gr.Checkbox(label="Automatically run Process Data, Feature Extraction and Train Index for this job", value=True)
                    queue_add_button = gr.Button("Add current settings to queue", variant="primary")
                    with gr.Row():
                        queue_start_button = gr.Button("Start queue", variant="primary")
                        queue_pause_button = gr.Button("Pause after current model", variant="secondary")
                        queue_refresh_button = gr.Button("Refresh queue", variant="secondary")
                    queue_action_log = gr.Textbox(label="Queue Action Log", lines=2, max_lines=3, interactive=False)
                    queue_table = gr.Markdown("", elem_id="training-queue-table")
                    queue_job = gr.Dropdown(label="Select queue job", choices=[], interactive=True)
                    with gr.Row():
                        queue_retry_button = gr.Button("Retry selected", variant="secondary")
                        queue_remove_button = gr.Button("Remove selected job", variant="secondary")
                    queue_status = gr.Textbox(label="Queue Log", lines=4, max_lines=5, interactive=False)
                    gr.Markdown("Errors and Stop and save pause the queue. Retry selected keeps the captured settings; saved training resumes at the next epoch. Pause after current model lets that model finish before stopping the queue.")
                    queue_start_button.click(start_training_queue, [], [queue_action_log], show_progress=False)
                    queue_pause_button.click(pause_training_queue, [], [queue_action_log], queue=False, show_progress=False)
                    queue_remove_button.click(remove_queued_training, [queue_job], [queue_action_log], show_progress=False)
                    queue_retry_button.click(retry_queued_training, [queue_job], [queue_action_log], show_progress=False)
                    queue_refresh_button.click(refresh_training_queue, [queue_job], [queue_table, queue_status, queue_job], show_progress=False)
                    queue_job.change(show_selected_queue_job, [queue_job], [queue_table, queue_status], show_progress=False)
                    queue_add_button.click(add_training_to_queue, [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        training_precision,
                        preview_enabled, preview_source, preview_every, preview_pitch, preview_method,
                        notification_desktop, notification_sound,
                        trainset_dir4,
                    training_vocoder, training_embedder, silent_training_files,
                    training_target_mode, training_total_steps, index_algorithm_control, pretrain_selector,
                        np7, gpus6, gpus_rmvpe, f0method8, queue_auto_prepare,
                    ], [queue_action_log], show_progress=False, api_name="add_training_to_queue")
            with gr.Column(elem_id="training-workflow"):
                with gr.Row(elem_id="preprocess-step"):
                    with gr.Column(scale=1, min_width=240):
                        but1 = gr.Button('1. Process Data', variant="primary")
                    with gr.Column(scale=3, min_width=320):
                        info1 = gr.Textbox(label="Pre-Process Log", value="", lines=1, max_lines=1)
                with gr.Row(elem_id="extraction-step"):
                    with gr.Column(scale=1, min_width=240):
                        but2 = gr.Button('2. Feature Extraction', variant="primary")
                    with gr.Column(scale=3, min_width=320):
                        info2 = gr.Textbox(label="Feature Extraction Log", value="", lines=1, max_lines=1)
                with gr.Row(elem_id="training-step"):
                    with gr.Column(scale=1, min_width=240):
                        but4 = gr.Button('3. Train Index', variant="primary")
                        but3 = gr.Button('4. Train Pth', variant="primary")
                        stop_training_button = gr.Button('Stop and save', variant="secondary")
                    with gr.Column(scale=3, min_width=320):
                        info3 = gr.Textbox(label="Training Log", value="", lines=2, max_lines=4)
                        stop_training_log = gr.Textbox(label="Stop / Resume", value="To resume a saved model, keep its Model Name and settings, then click 4. Train Pth.", lines=2, max_lines=3, interactive=False)
            refresh_datasets_button.click(refresh_dataset_choices, [], [dataset_to_remove], show_progress=False)
            remove_dataset_button.click(delete_training_dataset, [dataset_to_remove, trainset_dir4],
                [dataset_removal_log, dataset_to_remove, trainset_dir4], show_progress=False)
            dataset_upload.upload(import_training_dataset, [dataset_upload, exp_dir1],
                [dataset_upload_log, trainset_dir4], show_progress=False, api_name="import_training_dataset")
            dataset_check_button.click(check_training_dataset,
                [trainset_dir4, checker_silence_db, checker_silence_seconds, checker_volume_db],
                [dataset_check_log, dataset_check_report], show_progress=False, api_name="check_dataset")
            stop_training_button.click(stop_and_save_training, [], [stop_training_log],
                queue=False, show_progress=False, api_name="stop_and_save_training")
            preview_refresh.click(read_training_preview, [exp_dir1, preview_history],
                [preview_log, preview_audio, preview_history], queue=False, show_progress=False)
            preview_history.change(select_training_preview, [exp_dir1, preview_history],
                [preview_log, preview_audio], queue=False, show_progress=False)
            training_state_inputs = [version19, training_vocoder, sr2, if_f0_3,
                training_embedder, training_target_mode, if_cache_gpu17, training_precision,
                pretrain_selector, pretrained_G14, pretrained_D15, f0method8, preview_enabled]
            training_state_outputs = [sr2, if_f0_3, training_vocoder, training_embedder,
                if_cache_gpu17, training_precision, pretrain_selector, pretrained_G14,
                pretrained_D15, pretrain_status, f0method8, gpus_rmvpe,
                total_epoch11, training_total_steps, preview_source, preview_every,
                preview_pitch, preview_method, gpus6, gpus16]
            for control in (version19, training_vocoder, sr2, if_f0_3, training_embedder,
                            training_target_mode, if_cache_gpu17, training_precision,
                            pretrain_selector, f0method8, preview_enabled):
                control.change(update_training_controls, training_state_inputs,
                    training_state_outputs, show_progress=False)
            pretrain_refresh.click(update_training_controls, training_state_inputs, training_state_outputs, show_progress=False)
            but1.click(
                preprocess_dataset,
                [trainset_dir4, exp_dir1, sr2, np7],
                [info1],
                api_name="train_preprocess",
            )
            but2.click(
                extract_f0_feature,
                [
                    gpus6,
                    np7,
                    f0method8,
                    if_f0_3,
                    exp_dir1,
                    version19,
                    gpus_rmvpe,
                    training_precision,
                    trainset_dir4, training_embedder,
                ],
                [info2],
                api_name="train_extract_f0_feature",
            )
            but3.click(
                click_train,
                [
                    exp_dir1,
                    sr2,
                    if_f0_3,
                    spk_id5,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    if_save_latest13,
                    pretrained_G14,
                    pretrained_D15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    training_precision,
                    preview_enabled, preview_source, preview_every, preview_pitch, preview_method,
                    notification_desktop, notification_sound,
                    trainset_dir4,
                    training_vocoder, training_embedder, silent_training_files,
                    training_target_mode, training_total_steps, index_algorithm_control, pretrain_selector,
                ],
                [info3, preview_log, preview_audio, preview_history],
                api_name="train_start",
                show_progress=False,
            )
            but4.click(train_index, [exp_dir1, version19, trainset_dir4, index_algorithm_control], info3)
        with gr.TabItem(i18n("Inference")):
            with gr.Row():
                sid0 = gr.Dropdown(label="Voice Model", choices=sorted(names, key=str.casefold), value=sorted(names, key=str.casefold)[0] if names else None)
                with gr.Column():
                    refresh_button = gr.Button(i18n("Refresh"), variant="primary")
                    clean_button = gr.Button(i18n("Unload Voice from VRAM"), variant="primary")
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("Single Inference")):
                with gr.Column(elem_id="single-inference-controls"):
                    with gr.Column(elem_id="inference-audio-block"):
                        input_audio0 = gr.Audio(
                            label="Audio to convert",
                            source="upload",
                            type="filepath",
                            interactive=True,
                        )
                    with gr.Column(elem_id="inference-pitch-block"):
                        vc_transform0 = gr.Slider(label="Pitch", minimum=-24, maximum=24, step=1, value=0, interactive=True)
                    with gr.Accordion("Inference Settings", open=False, elem_id="inference-settings"):
                        spk_item = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=i18n("Speaker ID"),
                            value=0,
                            visible=False,
                            interactive=True,
                        )
                        f0method0 = gr.Radio(
                            label="Pitch Extraction",
                            choices=["rmvpe", "swift", "beatrice", "fcpe", "parselmouth", "harvest", "crepe"]
                            if config.dml is False
                            else ["pm", "harvest", "rmvpe", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        file_index1 = gr.Textbox(
                            label=i18n("Path of index"),
                            placeholder="%userprofile%\\Desktop\\models\\model_example.index",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label=i18n("Auto-detect index path"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Resampling, 0=none"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("0=Input source volume, 1=Normalized Output"),
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3 apply median filter to the harvested pitch results"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Index Ratio"),
                            value=0,
                            interactive=True,
                        )
                        format0 = gr.Radio(
                            list(EXPORT_FORMATS), value="wav", label="Export Format", interactive=True
                        )
                        audio_controls0 = build_inference_audio_controls("single-audio-processing")
                        embedder_controls0 = build_inference_embedder_controls("single-embedder-settings")
                        f0_file = gr.File(
                            label=i18n("F0 curve file [optional]"),
                            visible=False,
                        )
                    refresh_button.click(fn=change_choices, inputs=[], outputs=[sid0, file_index2], api_name="infer_refresh")

                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button("Convert", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="Output Log", lines=7, max_lines=10)
                            vc_output2 = NamedInferenceAudio(label=i18n("Output Audio"), elem_id="single-inference-output")

                        single_inputs = [
                            spk_item, input_audio0, vc_transform0, f0_file, f0method0,
                            file_index1, file_index2, index_rate1, filter_radius0,
                            resample_sr0, rms_mix_rate0, protect0,
                        ]
                        single_processing = [*single_inputs, format0, *audio_controls0]
                        but0.click(
                            convert_named_inference, [*single_processing, *embedder_controls0],
                            [vc_output1, vc_output2], api_name="infer_convert_with_embedder",
                        )
                        gr.Button(visible=False).click(
                            convert_named_inference, single_processing, [vc_output1, vc_output2],
                            api_name="infer_convert_with_options",
                        )
                        gr.Button(visible=False).click(
                            convert_named_inference, single_inputs, [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(i18n("Batch Inference")):
                with gr.Column():
                    with gr.Column():
                        inputs = gr.Audio(source="upload", type="filepath", label="Audio to convert")
                    with gr.Column():
                        vc_transform1 = gr.Slider(label="Pitch", minimum=-24, maximum=24, step=1, value=0, interactive=True)
                    with gr.Accordion("Inference Settings", open=False, elem_id="batch-inference-settings"):
                        spk_item_batch = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=i18n("Speaker ID"),
                            value=0,
                            visible=False,
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label="Pitch Extraction",
                            choices=["rmvpe", "swift", "beatrice", "fcpe", "parselmouth", "harvest", "crepe"]
                            if config.dml is False
                            else ["pm", "harvest", "rmvpe", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        file_index3 = gr.Textbox(
                            label=i18n("Path to index"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("Auto-detect index path"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("Resampling, 0=none"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("0=Input source volume, 1=Normalized Output"),
                            value=0.25,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3 apply median filter to the harvested pitch results"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Index Ratio"),
                            value=0,
                            interactive=True,
                        )
                        dir_input = gr.Textbox(
                            label=i18n("Enter the path to the audio folder to be processed"),
                            placeholder="%userprofile%\\Desktop\\covers",
                        )
                        opt_input = gr.Textbox(label=i18n("Output"), value="InferOutput")
                        format1 = gr.Radio(
                            label=i18n("Export Format"),
                            choices=list(EXPORT_FORMATS),
                            value="flac",
                            interactive=True,
                        )
                        audio_controls1 = build_inference_audio_controls("batch-audio-processing")
                        embedder_controls1 = build_inference_embedder_controls("batch-embedder-settings")
                    refresh_button.click(fn=lambda: change_choices()[1], inputs=[], outputs=file_index4, api_name="infer_refresh_batch")

                with gr.Column(elem_id="batch-inference-results"):
                    but1 = gr.Button("Convert", variant="primary")
                    with gr.Row():
                        vc_output3 = gr.Textbox(label="Output Log", lines=8, max_lines=16)
                        vc_output_batch_audio = NamedInferenceAudio(label=i18n("Output Audio"), elem_id="batch-inference-output")

                    batch_inputs = [
                        spk_item_batch, dir_input, opt_input, inputs, vc_transform1, f0method1,
                        file_index3, file_index4, index_rate2, filter_radius1, resample_sr1,
                        rms_mix_rate1, protect1, format1,
                    ]
                    batch_processing = [*batch_inputs, *audio_controls1]
                    but1.click(
                        batch_inference_audio, [*batch_processing, *embedder_controls1],
                        [vc_output3, vc_output_batch_audio], api_name="infer_convert_batch_with_embedder",
                    )
                    gr.Button(visible=False).click(
                        batch_inference_audio, batch_processing, [vc_output3, vc_output_batch_audio],
                        api_name="infer_convert_batch_with_options",
                    )
                    gr.Button(visible=False).click(
                        batch_inference_audio, batch_inputs, [vc_output3, vc_output_batch_audio],
                        api_name="infer_convert_batch",
                    )
            tts_pitch_methods = (
                ["rmvpe", "swift", "beatrice", "fcpe", "parselmouth", "harvest", "crepe"]
                if not config.dml else ["pm", "harvest", "rmvpe", "fcpe"]
            )
            tts_speaker, tts_index = build_tts_tab(
                gr, now_dir, sid0, convert_tts_voice, update_operation_gate.track,
                tts_pitch_methods, index_paths,
            )
            model_selection_outputs = [
                spk_item, protect0, protect1, file_index2, file_index4, file_index1,
                file_index3, spk_item_batch, tts_speaker, tts_index,
            ]
            sid0.change(
                fn=select_voice_model_with_tts,
                inputs=[sid0, protect0, protect1],
                outputs=model_selection_outputs,
                api_name="infer_change_voice",
            )

        with gr.TabItem("Other", elem_id="other-tab"):
            gr.HTML('<div class="ilaria-other-section"><h3>Get models</h3><p>Add voice models or download pretrained pairs for training.</p></div>')
            with gr.Accordion("Import local models", open=False, elem_id="import-models"):
                if not config.iscolab:
                    build_binary_model_upload(gr)
                # Keep the existing API available to integrations using Gradio File inputs.
                import_models_upload = gr.File(label="ZIP, PTH or INDEX files", file_count="multiple",
                                               file_types=[".zip", ".pth", ".index"], visible=config.iscolab)
                import_models_button = gr.Button("Import model files", variant="primary", visible=config.iscolab)
                import_models_log = gr.Textbox(label="Import Log", lines=3, max_lines=6,
                                              interactive=False, visible=config.iscolab)
                import_models_button.click(import_voice_models, [import_models_upload], [import_models_log],
                                           show_progress=False, api_name="import_voice_model")

            with gr.Accordion("Download models from Hugging Face, Google Drive or MEGA", open=False, elem_id="download-models"):
                model_url = gr.Textbox(label="Hugging Face, Google Drive or MEGA URL", placeholder="Paste a public model link")
                gr.Markdown("Paste a public Hugging Face repository or ZIP link, or a Google Drive or MEGA link to a `.zip`, `.pth`, or `.index` file. Include the decryption key in MEGA links.", elem_id="download-models-help")
                download_models_button = gr.Button("Download and install model", variant="primary")
                download_models_log = gr.Textbox(label="Download Log", lines=3, max_lines=6, interactive=False)
                download_models_button.click(download_voice_models, [model_url], [download_models_log], show_progress=False, api_name="download_voice_model")

            with gr.Accordion("Download pretrained models", open=False, elem_id="pretrained-download"):
                gr.Markdown("Paste a public Hugging Face repository, pretrained ZIP, or direct G .pth link. If needed, add its matching D link beside it. Both files are installed together in assets/pretrained_custom. Then open Training and click Refresh pretrained list.", elem_id="pretrained-download-help")
                with gr.Row():
                    pretrained_download_url = gr.Textbox(label="Repository, pretrained ZIP or G .pth link", placeholder="https://huggingface.co/author/pretrained")
                    pretrained_download_d_url = gr.Textbox(label="Matching D .pth link (optional)", placeholder="Leave empty for automatic discovery")
                pretrained_download_button = gr.Button("Download pretrained pair", variant="primary")
                pretrained_download_log = gr.Textbox(label="Pretrained Download Log", lines=3, max_lines=4, interactive=False)
                pretrained_download_button.click(download_training_pretrained,
                    [pretrained_download_url, pretrained_download_d_url], [pretrained_download_log],
                    show_progress=False, api_name="download_training_pretrained")

                with gr.Accordion("Recommended pretraineds", open=False, elem_id="recommended-pretraineds"):
                    gr.Markdown("Choose a pretrained from the alphabetical list. Every variant shows its sample rate. Download installs both G and D together; then click Refresh pretrained list in Training. These are the supplied recommendations, not a ranking.", elem_id="recommended-pretrained-help")
                    from ilaria_pretrain_catalog import choices, describe
                    recommended_pretrained_choice = gr.Dropdown(label="Recommended pretrained and sample rate", choices=choices(), value=None, interactive=True)
                    recommended_pretrained_details = gr.Markdown(describe(None), elem_id="recommended-pretrained-details")
                    recommended_pretrained_button = gr.Button("Download selected pretrained (G + D)", variant="primary")
                    recommended_pretrained_log = gr.Textbox(label="Recommended Pretrained Download Log", lines=3, max_lines=4, interactive=False)
                    recommended_pretrained_choice.change(recommended_pretrained_info, [recommended_pretrained_choice], [recommended_pretrained_details], show_progress=False)
                    recommended_pretrained_button.click(download_recommended_pretrained, [recommended_pretrained_choice], [recommended_pretrained_log], show_progress=False, api_name="download_recommended_pretrained")

            gr.HTML('<div class="ilaria-other-section"><h3>Manage voice models</h3><p>Inspect, blend or export your installed voices.</p></div>')
            with gr.Accordion("Model information", open=False, elem_id="model-information"):
                gr.Markdown("Select an installed voice model to read its saved metadata and weight details. Missing training information is shown as unknown.", elem_id="model-info-help")
                info_model = gr.Dropdown(label="Voice Model", choices=sorted(names, key=str.casefold), value=None, interactive=True)
                with gr.Row():
                    refresh_info_button = gr.Button("Refresh model list", variant="primary")
                    read_info_button = gr.Button("Read model information", variant="primary")
                model_info_status = gr.Textbox(label="Inspection Log", value="Select a Voice Model to inspect its details and matching index files.", lines=1, max_lines=3, interactive=False)
                model_info_summary = gr.Markdown("", elem_id="model-info-summary")
                with gr.Accordion("Saved metadata and architecture", open=False):
                    model_info_metadata = gr.JSON(label="Model metadata", value={})
                with gr.Accordion("Stored tensors", open=False):
                    model_info_tensors = gr.Markdown("", elem_id="model-info-tensors")
                info_outputs = [model_info_status, model_info_summary, model_info_metadata, model_info_tensors]
                refresh_info_button.click(refresh_model_info_choices, [info_model], [info_model], show_progress=False, api_name="refresh_model_information")
                info_model.change(inspect_voice_model, [info_model], info_outputs, show_progress=False, api_name="inspect_voice_model")
                read_info_button.click(inspect_voice_model, [info_model], info_outputs, show_progress=False)

            with gr.Accordion("Merge models", open=False, elem_id="merge-models"):
                gr.Markdown("### Blend two voices into a new model\nChoose two exported RVC `.pth` models already installed in `models`. The originals stay unchanged. The result is a weight blend, not a new training run and not an audio mix.")
                with gr.Row():
                    merge_model_a = gr.Dropdown(label="Model A", choices=sorted(names, key=str.casefold), value=None, interactive=True)
                    merge_model_b = gr.Dropdown(label="Model B", choices=sorted(names, key=str.casefold), value=None, interactive=True)
                refresh_merge_button = gr.Button("Refresh model list", variant="primary")
                merge_ratio = gr.Slider(label="Model A contribution (%)", minimum=0, maximum=100, step=1, value=50, interactive=True)
                merge_blend_info = gr.Markdown(describe_merge_ratio(50))
                with gr.Row():
                    merge_sample_rate = gr.Radio(label="Sample rate", choices=["Auto (from models)", "32k", "40k", "48k"], value="Auto (from models)", interactive=True)
                    merge_precision = gr.Radio(label="Output precision", choices=["FP16", "FP32"], value="FP16", interactive=True)
                gr.Markdown("**Compatibility:** both models must use the same sample rate, RVC version, pitch-guidance setting, architecture, and tensor shapes. Auto reads the rate from the models; choosing 32k, 40k or 48k checks it rather than resampling the weights.\n\n**Precision:** FP16 creates a smaller inference model; FP32 retains the blended weights in full precision. The blend is calculated on CPU in FP32. A 50/50 weight blend does not guarantee a perceptually equal mix of the two voices.")
                with gr.Row():
                    merge_output_name = gr.Textbox(label="New model name", value="", placeholder="merged_voice")
                    merge_model_info = gr.Textbox(label="Model notes (optional)", placeholder="Describe this voice blend", lines=2)
                gr.Markdown("Use a new output name: existing models are kept. The new `.pth` is saved in `models`. Index files are not merged; start with Index Ratio 0 when testing the result.")
                with gr.Row():
                    inspect_merge_button = gr.Button("Check model compatibility", variant="primary")
                    merge_models_button = gr.Button("Merge and save model", variant="primary")
                merge_models_log = gr.Textbox(label="Merge Log", lines=4, max_lines=8, interactive=False)
                refresh_merge_button.click(refresh_merge_model_choices, [merge_model_a, merge_model_b], [merge_model_a, merge_model_b], api_name="refresh_merge_models")
                merge_ratio.change(describe_merge_ratio, [merge_ratio], [merge_blend_info])
                inspect_merge_button.click(inspect_merge_models, [merge_model_a, merge_model_b], [merge_models_log], show_progress=False, api_name="inspect_merge_models")
                merge_models_button.click(merge_voice_models, [merge_model_a, merge_model_b, merge_ratio, merge_sample_rate, merge_output_name, merge_model_info, merge_precision], [merge_models_log], show_progress=False, api_name="merge_voice_models")

            with gr.Accordion("Export model ZIP", open=False, elem_id="model-export"):
                gr.Markdown("Choose a Voice Model and its matching index, then click Create model ZIP. Download the file below to share it. The ZIP contains only the selected .pth and .index, with no recordings or training logs. Leave Index empty to export only the .pth.", elem_id="model-export-help")
                export_model_choice = gr.Dropdown(label="Voice Model", choices=sorted(names, key=str.casefold), value=None, interactive=True)
                export_index_choice = gr.Dropdown(label="Index (optional; check that it matches your model)", choices=[], value=None, interactive=True)
                with gr.Row():
                    export_refresh_button = gr.Button("Refresh model list", variant="primary")
                    export_create_button = gr.Button("Create model ZIP", variant="primary")
                export_status = gr.Textbox(label="Export Log", lines=2, max_lines=3, interactive=False)
                export_download = gr.File(label="Download model ZIP", interactive=False)
                export_refresh_button.click(refresh_export_models, [export_model_choice], [export_model_choice, export_index_choice], show_progress=False)
                export_model_choice.change(select_export_index, [export_model_choice], [export_index_choice], show_progress=False)
                export_create_button.click(export_voice_model, [export_model_choice, export_index_choice], [export_status, export_download], show_progress=False, api_name="export_voice_model")

            gr.HTML('<div class="ilaria-other-section"><h3>Audio tools</h3><p>Inspect recordings and compare audio files.</p></div>')
            build_audio_analyzer(gr, now_dir, update_operation_gate.track)

            gr.HTML('<div class="ilaria-other-section"><h3>Training tools</h3><p>Find starting settings for your hardware.</p></div>')
            with gr.Accordion("PC training advisor", open=False, elem_id="training-advisor"):
                gr.Markdown("Check this PC and get starting settings for a new training experiment. The scan reads hardware locally; it does not run a training benchmark or change your settings.", elem_id="training-advisor-help")
                with gr.Row():
                    advisor_sample_rate = gr.Radio(choices=["32k", "40k", "48k"], value="32k", label="Sample rate to plan for")
                    advisor_precision = gr.Radio(choices=["FP32", "FP16"], value="FP32", label="Precision to evaluate")
                advisor_gpu = gr.Dropdown(choices=["Auto (largest compatible GPU)"], value="Auto (largest compatible GPU)", label="GPU to evaluate", interactive=True)
                advisor_scan = gr.Button("Scan PC and suggest settings", variant="primary")
                advisor_status = gr.Textbox(label="PC Advisor Log", value="Click Scan PC and suggest settings to inspect this computer.", lines=2, max_lines=3, interactive=False)
                advisor_hardware = gr.Markdown("", elem_id="training-advisor-hardware")
                advisor_recommendations = gr.Markdown("", elem_id="training-advisor-recommendations")
                advisor_snapshot = gr.State(value=None)
                advisor_scan.click(scan_training_advice,
                    [advisor_sample_rate, advisor_precision, advisor_gpu],
                    [advisor_status, advisor_hardware, advisor_recommendations, advisor_gpu, advisor_snapshot],
                    show_progress=False, api_name="scan_pc_training_advice")
                for advisor_control in (advisor_sample_rate, advisor_precision, advisor_gpu):
                    advisor_control.change(refresh_training_advice,
                        [advisor_snapshot, advisor_sample_rate, advisor_precision, advisor_gpu],
                        [advisor_status, advisor_recommendations], show_progress=False)

            gr.HTML('<div class="ilaria-other-section"><h3>Application</h3><p>Update or restart Ilaria RVC.</p></div>')
            with gr.Accordion("Update Ilaria RVC", open=False, elem_id="program-updates"):
                gr.Markdown("Check for a new Ilaria RVC release. Download and restart creates a backup, installs the update and reopens the UI. Finish all current operations first.", elem_id="program-updates-help")
                installed_version_display = gr.Textbox(label="Installed version", value=installed_program_version(), interactive=False, elem_id="updater-installed-version")
                with gr.Row():
                    check_updates_button = gr.Button("Check for updates", variant="primary")
                    install_update_button = gr.Button("Download and restart", variant="primary", visible=False)
                update_status = gr.Textbox(label="Update Log", value="Checking for updates when the UI opens...", lines=2, max_lines=4, interactive=False)
                update_notes = gr.Textbox(label="Version changelog", lines=4, max_lines=12, interactive=False)
                with gr.Accordion("All changes so far", open=False, elem_id="update-all-changes"):
                    gr.Markdown("Expand a version to read its changes. The latest notes are at the top.")
                    for number, (release, details) in enumerate(RELEASE_HISTORY):
                        with gr.Accordion(release, open=False, elem_id=f"release-history-{number}"):
                            gr.Markdown(details)
                check_updates_button.click(check_program_updates, [], [update_status, update_notes, install_update_button, program_update_notice], show_progress=False, api_name="check_program_updates")
                install_update_button.click(download_program_update, [], [update_status], show_progress=False, api_name=False)
                update_status.change(None, [update_status], [], _js=UPDATER_RELOAD_JS, queue=False)

            reload_ui_button = gr.Button("Reload UI", variant="secondary")
            reload_ui_log = gr.Textbox(label="Reload status", lines=1, interactive=False)
            reload_ui_instance = gr.Textbox(value=str(os.getpid()), visible=False, elem_id="reload-ui-instance")
            reload_ui_button.click(restart_local_ui, [], reload_ui_log, show_progress=False)
            reload_ui_log.change(None, [reload_ui_log, reload_ui_instance], [], _js=RELOAD_LOCAL_UI_JS, queue=False)

        with gr.TabItem("Glossary", elem_id="tutorial-tab"):
            with gr.Column(elem_id="tutorial-content"):
                gr.Markdown(TUTORIAL_INTRO)
                for section_number, (section_title, section_text) in enumerate(TUTORIAL_SECTIONS):
                    with gr.Accordion(section_title, open=False, elem_id="glossary-section-%s" % section_number):
                        gr.Markdown(section_text)

        with gr.TabItem("", elem_id="easter-egg-tab"):
            gr.Markdown("![ilaria](https://i.ytimg.com/vi/5PWqt2Wg-us/maxresdefault.jpg)", elem_id="easter-egg-image")

    gr.HTML('<div style="text-align: center; padding: 18px 0 4px; color: #F8EFFA; font-size: 15px;">Made with 💖 by Ilaria</div>')

    app.load(fn=select_voice_model_with_tts, inputs=[sid0, protect0, protect1], outputs=model_selection_outputs, show_progress=False)
    app.load(refresh_dataset_choices, [], [dataset_to_remove], show_progress=False)
    app.load(refresh_training_queue, [queue_job], [queue_table, queue_status, queue_job], every=2, show_progress=False)
    app.load(update_training_controls, training_state_inputs, training_state_outputs, show_progress=False)
    app.load(_js=UI_FAVICON_JS)
    if not config.iscolab:
        app.load(_js=BINARY_MODEL_UPLOAD_JS)
    app.load(_js=WAVEFORM_JS)
    presence_tab = gr.Textbox(visible=False)
    presence_open = gr.Checkbox(visible=False)
    presence_seq = gr.Number(visible=False)
    presence_event = gr.Button(visible=False)
    presence_event.click(browser_presence.record, [presence_tab, presence_open, presence_seq], [],
                         queue=False, show_progress=False, api_name="ui_presence")
    app.load(_js=BROWSER_PRESENCE_JS)
    app.load(check_program_updates, [], [update_status, update_notes, install_update_button, program_update_notice], show_progress=False)

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(
            share=True, favicon_path=str(UI_ICON_PATH), prevent_thread_lock=True
        )
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            favicon_path=str(UI_ICON_PATH),
            server_name="0.0.0.0",
            inbrowser=False,
            server_port=config.listen_port,
            quiet=True,
            prevent_thread_lock=True,
        )
    if not config.iscolab:
        register_binary_model_import(app.server_app, now_dir, update_operation_gate)
        if not config.noautoopen:
            import webbrowser
            webbrowser.open(app.local_url)


def warm_default_voice():
    global loaded_voice
    if not names or not str(config.device).startswith("cuda") or config.dml:
        return
    default_voice = min(names, key=str.casefold)
    try:
        with update_operation_gate.lock:
            if update_operation_gate.updating or any(
                child.poll() is None for child in update_operation_gate.children
            ):
                return
            update_operation_gate.enter("inference warmup")
        try:
            with voice_lock:
                if voice_selection_started and loaded_voice != default_voice:
                    return
                free_vram, _ = torch.cuda.mem_get_info(config.device)
                if free_vram < 3 * 1024**3:
                    return
                if loaded_voice != default_voice:
                    vc.get_vc(default_voice, find_index=False, reuse_loaded=True)
                    loaded_voice = default_voice
                embedder = vc.cpt.get("embedder_model") or "contentvec"
                from ilaria_hubert_assets import inference_embedder_installed

                if not inference_embedder_installed(now_dir, embedder):
                    return
                vc.ensure_embedder_loaded()
            # Prepare the default volume-envelope path on a short synthetic signal.
            from ilaria_inference_pipeline import change_rms
            silence = np.zeros(16000, dtype=np.float32)
            change_rms(silence, 16000, silence, 16000, 0.25)
        finally:
            update_operation_gate.leave("inference warmup")
    except Exception:
        logger.debug("Inference startup warmup skipped; inference will retry.", exc_info=True)


notify_launcher_ready()
if not config.iscolab and os.environ.get("ILARIA_RVC_LIFECYCLE_V2") == "1":
    threading.Thread(target=browser_presence.shutdown_on_close, name="Browser presence", daemon=True).start()
threading.Thread(target=warm_default_voice, name="Default voice warmup", daemon=True).start()
wait_for_program_restart(app)
