"""Training terminal-event monitor independent of a browser request's lifetime."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import logging
from pathlib import Path
import re
import threading

LOGGER = logging.getLogger(__name__)
MAX_TAIL_BYTES = 256 * 1024
_DONE = re.compile(r"DONE! Training completed \| Completed at: (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
_EPOCH = re.compile(r"Training Model: .* ====> Epoch: (\d+) \|\| Total Steps: (\d+) \|\| Time taken:")
_STOPPED = re.compile(r"STOPPED! Training saved \| Epoch: (\d+) \| Total Steps: (\d+) \| Saved at:")
_STEP_DONE = re.compile(r"DONE! Step target reached \| Total Steps: (\d+)/(\d+)")


def read_training_tail(log_path, offset, limit=MAX_TAIL_BYTES):
    """Read a bounded suffix, never any bytes preceding this run's start."""
    offset = max(0, int(offset))
    limit = min(MAX_TAIL_BYTES, max(1, int(limit)))
    try:
        with Path(log_path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            if size <= offset:
                return ""
            start = max(offset, size - limit)
            stream.seek(start)
            data = stream.read(limit)
        # A suffix may start halfway through an old line. Do not interpret it
        # as an intact terminal record (or a partially cut UTF-8 character).
        if start > offset:
            _, separator, data = data.partition(b"\n")
            if not separator:
                return ""
        return data.decode("utf-8", errors="replace")
    except (OSError, ValueError, TypeError):
        return ""


def classify_training_result(returncode, tail, total_epochs, total_steps=None):
    """Classify only a clean export plus successful parent/worker termination."""
    try:
        total_epochs = int(total_epochs)
    except (ValueError, TypeError, OverflowError):
        total_epochs = 0
    done = list(_DONE.finditer(tail))
    stopped = list(_STOPPED.finditer(tail))
    epochs = list(_EPOCH.finditer(tail))
    step_done = list(_STEP_DONE.finditer(tail))
    if returncode == 0:
        if stopped and (not done or stopped[-1].start() > done[-1].start()):
            return {"event": None, "detail": "Training stopped and saved at an epoch boundary."}
        if total_steps is not None:
            if (done and step_done and int(step_done[-1].group(1)) == int(total_steps)
                    and int(step_done[-1].group(2)) == int(total_steps)
                    and step_done[-1].end() < done[-1].start()):
                return {"event": "completed", "detail": done[-1].group(1)}
        elif done and epochs and total_epochs > 0 and int(epochs[-1].group(1)) == total_epochs:
            return {"event": "completed", "detail": done[-1].group(1)}
    return {"event": "error", "detail": "Training did not finish successfully. Check Training Log and training_console.log."}


def start_notification_monitor(process, log_path, total_epochs, offset, root,
                               model_name, desktop=True, sound=True,
                               run_id=None, gate=None, total_steps=None):
    """Monitor once per launched job; closing its browser tab cannot cancel it.

    Notification failures are optional UI failures and must never change a
    training result or escape into its caller. The update gate remains held
    while the small notification worker is using application files.
    """
    desktop, sound = bool(desktop), bool(sound)
    if not desktop and not sound:
        return None
    label = "training notification"
    entered = False
    try:
        if gate is not None:
            gate.enter(label)
            entered = True
    except Exception:
        LOGGER.debug("Notification monitor could not register with operation gate", exc_info=True)
        return None

    def monitor():
        try:
            code = process.wait()
            result = classify_training_result(code, read_training_tail(log_path, offset), total_epochs, total_steps)
            if result["event"] is None:
                return
            detail = result["detail"]
            from ilaria_training_notifications import notify_training_event
            sent = notify_training_event(root, model_name, result["event"], detail,
                                         desktop=desktop, sound=sound, run_id=run_id)
            worker = sent.get("_process") if isinstance(sent, dict) else None
            if worker is not None:
                try:
                    worker.wait(timeout=15)
                except Exception:
                    # The worker has its own delivery deadline; never terminate
                    # training or block the application's update gate forever.
                    LOGGER.debug("Notification worker did not finish within monitor deadline", exc_info=True)
        except Exception:
            LOGGER.debug("Optional training notification failed", exc_info=True)
        finally:
            if entered:
                try:
                    gate.leave(label)
                except Exception:
                    LOGGER.debug("Notification operation cleanup failed", exc_info=True)

    try:
        thread = threading.Thread(target=monitor, name="Ilaria training notification", daemon=True)
        thread.start()
    except Exception:
        if entered:
            try:
                gate.leave(label)
            except Exception:
                pass
        LOGGER.debug("Could not start optional notification monitor", exc_info=True)
        return None
    return thread
