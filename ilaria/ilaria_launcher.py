"""Keep the UI and file replacement in separate processes (Python 3.9+)."""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
from pathlib import Path
import os
import json
import errno
import stat
import subprocess
import sys
import uuid

RESTART_FOR_UPDATE = 42
RESTART_UI = 43


def requested_ui_url(arguments):
    port = 7865  # Default used by configs/config.py; ILARIA_RVC.bat passes 6969.
    for index, argument in enumerate(arguments):
        try:
            if argument == "--port":
                port = int(arguments[index + 1])
            elif argument.startswith("--port="):
                port = int(argument.split("=", 1)[1])
        except (ValueError, IndexError):
            return None
    return "http://127.0.0.1:%s" % port if 1 <= port <= 65535 else None


def already_running_message(root, info):
    pid = info.get("pid")
    header = "Ilaria RVC is already running" + (" (PID %s)." % pid if type(pid) is int and pid > 0 else ".")
    url = info.get("web_ui_url")
    lines = [header]
    if isinstance(url, str) and url.startswith("http://127.0.0.1:") and url.rsplit(":", 1)[-1].isdigit():
        lines.append("Open: " + url)
    lines.append("To restart: finish or save training, then close the original CMD and open ILARIA_RVC.bat again.")
    return "\n".join(lines)


class LauncherLock:
    def __init__(self, root, web_ui_url=None):
        self.root = Path(root).resolve()
        self.web_ui_url = web_ui_url
        self.path = Path(root) / '.ilaria-updates' / 'launcher.lock'
        self.handle = None

    def __enter__(self):
        folder = self.path.parent
        if folder.exists() and (folder.is_symlink() or
                getattr(folder.stat(), 'st_file_attributes', 0) & 0x400):
            raise RuntimeError('The update folder must be a normal local directory.')
        folder.mkdir(exist_ok=True)
        if self.path.exists() and (self.path.is_symlink() or
                getattr(self.path.stat(), 'st_file_attributes', 0) & 0x400):
            raise RuntimeError('The launcher lock must be a normal local file.')
        self.handle = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            info = {}
            try:
                self.handle.seek(1)
                info = json.loads(self.handle.read(8192).decode("utf-8"))
                if not isinstance(info, dict) or info.get("folder") != str(self.root):
                    info = {}
            except (OSError, ValueError, UnicodeError):
                pass
            self.handle.close()
            self.handle = None
            if error.errno not in (None, errno.EACCES, errno.EAGAIN, errno.EDEADLK) and getattr(error, "winerror", None) not in (32, 33, 36):
                raise RuntimeError("Windows could not access the launcher lock for this folder: %s. Details: %s" % (self.root, error)) from error
            raise RuntimeError(already_running_message(self.root, info)) from error
        try:
            self.handle.seek(1)
            self.handle.truncate()
            self.handle.write(json.dumps({"pid": os.getpid(), "folder": str(self.root),
                                          "web_ui_url": self.web_ui_url}).encode("utf-8"))
            self.handle.flush()
        except OSError:
            pass  # Optional diagnostics must not prevent a valid launch.
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.close()


def attach_windows_job(child):
    """Terminate server subprocesses when the launcher releases the job."""
    if os.name != "nt":
        return None
    import win32api
    import win32con
    import win32job

    job = None
    try:
        job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        process = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                                       False, child.pid)
        try:
            win32job.AssignProcessToJobObject(job, process)
        finally:
            process.Close()
        return job
    except Exception:
        if job is not None:
            job.Close()
        if child.poll() is not None:
            return None
        child.terminate()
        child.wait()
        raise


def run(root, arguments, python_executable=None):
    root = Path(root).resolve()
    python_executable = str(python_executable or root / 'runtime' / 'python.exe')
    updater = root / 'ilaria' / 'ilaria_updater.py'
    if not Path(python_executable).is_file():
        raise RuntimeError('The bundled Python or updater is missing. Extract all update files into the Ilaria RVC folder.')
    with LauncherLock(root, requested_ui_url(arguments)):
        payload_path = root / 'ilaria_release_payload.py'
        if payload_path.is_file():
            import importlib.util
            spec = importlib.util.spec_from_file_location('_ilaria_release_payload', payload_path)
            payload = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(payload)
            payload.ensure_layout(root)
        from ilaria_release_migration import cleanup
        cleanup(root, keep_updater=False)
        if not updater.is_file():
            raise RuntimeError('The updater module is missing from the ilaria folder.')
        # Recovery restores an interrupted transaction; it never installs a pending download.
        recovered = subprocess.run([python_executable, str(updater), '--recover', '--root', str(root)], cwd=str(root))
        if recovered.returncode:
            raise RuntimeError('An interrupted update could not be restored. See the update log in .ilaria-updates.')
        if not updater.is_file() and (root / 'ilaria_updater.py').is_file():
            updater = root / 'ilaria_updater.py'
        from ilaria_release_migration import _state as layout_state
        migration = layout_state(root)
        restarted = ((root / "ilaria/ilaria_release_payload.py").is_file()
                     or bool(migration and migration["phase"] == "pending"))
        suppress_browser = False
        while True:
            options = list(arguments)
            if suppress_browser and '--noautoopen' not in options:
                options.append('--noautoopen')
            environment = os.environ.copy()
            environment['ILARIA_RVC_MANAGED_LAUNCHER'] = '1'
            environment['ILARIA_RVC_NESTED_LAUNCHER'] = '1'
            environment['ILARIA_RVC_LIFECYCLE_V2'] = '1'
            ready_token = uuid.uuid4().hex
            ready_path = root / '.ilaria-updates' / ('ready-' + ready_token)
            environment['ILARIA_RVC_READY_TOKEN'] = ready_token
            child = subprocess.Popen([python_executable, str(root / 'ILARIA_RVC.py')] + options,
                                     cwd=str(root), env=environment)
            job = attach_windows_job(child)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                # The UI shares the console and receives Ctrl+C too.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print('Ilaria RVC is still closing. Wait for it to finish.', flush=True)
                    child.wait()
                return 0
            finally:
                if job is not None:
                    job.Close()
            ready = ready_path.is_file() and ready_path.read_text(encoding='ascii') == ready_token
            if ready:
                ready_path.unlink()
            if restarted and code not in (0, RESTART_FOR_UPDATE, RESTART_UI) and not ready:
                print('The updated UI could not start. Restoring the previous version...', flush=True)
                restored = subprocess.run([python_executable, str(updater), '--rollback-latest', '--root', str(root)], cwd=str(root))
                if restored.returncode:
                    print('Automatic recovery needs attention. Keep the backups in .ilaria-updates.', flush=True)
                    return restored.returncode
                if not updater.is_file() and (root / 'ilaria_updater.py').is_file():
                    updater = root / 'ilaria_updater.py'
                restarted = False
                continue
            if code == RESTART_UI:
                print('Status: Restarting Ilaria RVC...', flush=True)
                suppress_browser = True
                continue
            if code != RESTART_FOR_UPDATE:
                return code
            print('Status: Installing the downloaded update...', flush=True)
            applied = subprocess.run([python_executable, str(updater), '--apply-pending', '--root', str(root)], cwd=str(root))
            if applied.returncode:
                print('Update failed. Check the messages above before restarting Ilaria RVC.', flush=True)
                return applied.returncode
            print('Status: Restarting Ilaria RVC...', flush=True)
            restarted = True
            suppress_browser = True


if __name__ == '__main__':
    try:
        raise SystemExit(run(Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'ilaria' else Path(__file__).resolve().parent, sys.argv[1:]))
    except Exception as error:
        print('Launcher: ' + str(error), flush=True)
        raise SystemExit(1)
