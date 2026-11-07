"""Optional local Windows training alerts, isolated from training/browser lifetimes.

No extra Python packages or remote service. Registers a per-user Windows notification identity.
Windows may suppress desktop alerts or sound according to the user's settings.
"""

import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[1]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import argparse
import ctypes
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
import time
import unicodedata

LOGGER = logging.getLogger(__name__)
WORKER_SECONDS = 12.0


def _clean(value, limit):
    text = " ".join(str(value or "").split())
    text = "".join(char for char in text if not unicodedata.category(char).startswith("C"))
    # Windows fields count UTF-16 code units, not Python Unicode code points.
    return text.encode("utf-16-le", "replace")[:limit * 2].decode("utf-16-le", "ignore")


def _message(model_name, event, detail=""):
    model = _clean(model_name, 80) or "Unnamed model"
    if event == "test":
        return "Ilaria RVC: test notification", "Training notifications are ready. This is only a test; no training has finished or failed."
    if event == "completed":
        text = 'Training completed for "%s".' % model
        # Include only a known timestamp. Never send traceback text or paths to
        # the desktop where private diagnostics could be seen by other users.
        match = re.fullmatch(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", str(detail or ""))
        if match:
            text += " Completed at: " + match.group(0) + "."
        return "Ilaria RVC: training completed", _clean(text, 199)
    return "Ilaria RVC: training error", _clean(
        'Training for "%s" did not finish successfully. Read Training Log or training_console.log for details.' % model, 199)


def _asset_icon_path(root):
    for name in ("ilaria_rvc_icon_transparent.png", "ilaria_rvc_icon.png"):
        candidate = Path(root).resolve() / "assets" / name
        if candidate.is_file():
            return str(candidate)
    return None


def _icon_resource(icon_path, size):
    """Encode the existing UI artwork as an in-memory Windows icon resource."""
    from PIL import Image
    size = max(16, min(256, int(size)))
    with Image.open(icon_path) as source:
        frame = source.convert("RGBA")
        frame.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.alpha_composite(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
        encoded = io.BytesIO()
        canvas.save(encoded, format="ICO", sizes=[(size, size)], bitmap_format="bmp")
    data = encoded.getvalue()
    if data[:6] != b"\0\0\1\0\1\0" or len(data) < 22:
        raise ValueError("The artwork could not be encoded as an icon.")
    length, offset = struct.unpack_from("<II", data, 14)
    if length <= 0 or offset < 22 or offset + length > len(data):
        raise ValueError("Invalid in-memory icon resource.")
    return data[offset:offset + length]


def _create_asset_icon(user32, icon_path):
    """Return an owned HICON, or None so notification delivery can fall back."""
    if not icon_path:
        return None
    try:
        from ctypes import wintypes as wt
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int
        size = user32.GetSystemMetrics(11) or 32  # SM_CXICON for the large balloon
        resource = _icon_resource(icon_path, size)
        memory = ctypes.create_string_buffer(resource)
        user32.CreateIconFromResourceEx.argtypes = [ctypes.POINTER(ctypes.c_ubyte), wt.DWORD,
            wt.BOOL, wt.DWORD, ctypes.c_int, ctypes.c_int, wt.UINT]
        user32.CreateIconFromResourceEx.restype = wt.HICON
        return user32.CreateIconFromResourceEx(ctypes.cast(memory, ctypes.POINTER(ctypes.c_ubyte)),
            len(resource), True, 0x00030000, size, size, 0) or None
    except Exception:
        return None


def _claim(root, run_id, event):
    if not run_id:
        return None, True
    # Only a hash becomes a path: a caller-supplied ID cannot escape this folder.
    digest = hashlib.sha256((str(run_id) + "\0" + event).encode("utf-8")).hexdigest()
    folder = Path(root).resolve() / ".ilaria-notifications" / "sent"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / (digest + ".json")
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"event": event, "requested_at": time.time()}, stream)
        return marker, True
    except FileExistsError:
        return marker, False


def notify_training_event(root, model_name, event="completed", detail="",
                          desktop=True, sound=True, run_id=None):
    """Request a hidden, nonblocking alert; exceptions never escape the caller.

    Returns {requested, status, _process?}. Only the private process handle is
    non-serializable; UI callbacks should display status, not return this dict.
    Call once from the monitoring parent, not once per distributed GPU worker.
    """
    if event == "stopped":
        return {"requested": False, "status": "Training was stopped and saved; no completion/error alert was requested."}
    if event not in ("completed", "error", "test"):
        return {"requested": False, "status": "Unknown training notification event."}
    desktop, sound = desktop is True, sound is True
    if not desktop and not sound:
        return {"requested": False, "status": "Training notifications are disabled. Enable desktop alerts or sound to test them."}
    if os.name != "nt":
        return {"requested": False, "status": "Local desktop training notifications are supported on Windows only."}
    marker = None
    try:
        marker, first = _claim(root, run_id, event)
        if not first:
            return {"requested": False, "status": "This training alert has already been requested."}
        title, body = _message(model_name, event, detail)
        payload = {"title": title, "body": body, "event": event,
                   "desktop": desktop, "sound": sound, "icon_path": _asset_icon_path(root)}
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--deliver", json.dumps(payload, ensure_ascii=True)],
            cwd=str(Path(root).resolve()), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"requested": True, "_process": child,
                "status": "Notification request sent. Windows notification settings, Do not disturb and system volume can hide or silence it."}
    except Exception as error:
        if marker:
            try:
                marker.unlink()
            except OSError:
                pass
        LOGGER.warning("Training notification could not be requested: %s", error)
        return {"requested": False, "status": "The training alert could not be requested. Training results are unaffected."}


def test_notifications(root, desktop=True, sound=True):
    return notify_training_event(root, "", event="test", desktop=desktop, sound=sound)


def _play_sound(event):
    import winsound
    if event in ("completed", "test"):
        custom = Path(__file__).resolve().parents[1] / "assets" / "training-completed.wav"
        if custom.is_file():
            try:
                winsound.PlaySound(str(custom), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                return
            except (OSError, RuntimeError):
                pass
    winsound.MessageBeep(winsound.MB_ICONHAND if event == "error" else winsound.MB_ICONASTERISK)


def _show_legacy_desktop(title, body, event, duration=WORKER_SECONDS, icon_path=None):
    """Transient Shell_NotifyIcon balloon with a bounded hidden message loop.

    Microsoft docs: /windows/win32/api/shellapi/ns-shellapi-notifyicondataw
    and /windows/win32/shell/notification-area. A successful API call only
    acknowledges submission: quiet time and OS policy may suppress the alert.
    """
    from ctypes import wintypes as wt

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    class NOTIFYICONDATAW(ctypes.Structure):
        _fields_ = [("cbSize", wt.DWORD), ("hWnd", wt.HWND), ("uID", wt.UINT),
                    ("uFlags", wt.UINT), ("uCallbackMessage", wt.UINT),
                    ("hIcon", wt.HICON), ("szTip", wt.WCHAR * 128),
                    ("dwState", wt.DWORD), ("dwStateMask", wt.DWORD),
                    ("szInfo", wt.WCHAR * 256), ("uVersion", wt.UINT),
                    ("szInfoTitle", wt.WCHAR * 64), ("dwInfoFlags", wt.DWORD),
                    ("guidItem", GUID), ("hBalloonIcon", wt.HICON)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
    user32.CreateWindowExW.restype = wt.HWND
    user32.DestroyWindow.argtypes = [wt.HWND]
    user32.DestroyWindow.restype = wt.BOOL
    user32.DestroyIcon.argtypes = [wt.HICON]
    user32.DestroyIcon.restype = wt.BOOL
    user32.LoadIconW.argtypes = [wt.HINSTANCE, ctypes.c_void_p]
    user32.LoadIconW.restype = wt.HICON
    user32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
    user32.PeekMessageW.restype = wt.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
    user32.TranslateMessage.restype = wt.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wt.HMODULE
    shell32.Shell_NotifyIconW.argtypes = [wt.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
    shell32.Shell_NotifyIconW.restype = wt.BOOL
    instance = kernel32.GetModuleHandleW(None)
    # Built-in STATIC class avoids registering a permanent window class. The
    # window never uses WS_VISIBLE and does not steal foreground focus.
    window = user32.CreateWindowExW(0, "STATIC", "Ilaria RVC notifications", 0,
                                  0, 0, 0, 0, None, None, instance, None)
    if not window:
        raise ctypes.WinError(ctypes.get_last_error())
    data = NOTIFYICONDATAW()
    data.cbSize, data.hWnd, data.uID = ctypes.sizeof(data), window, 1
    data.uCallbackMessage = 0x8001
    data.hIcon = user32.LoadIconW(None, ctypes.c_void_p(32513 if event == "error" else 32516))
    data.szTip = "Ilaria RVC"
    data.uFlags = 0x01 | 0x02 | 0x04  # NIF_MESSAGE | NIF_ICON | NIF_TIP
    added = False
    custom_icon = None
    try:
        custom_icon = _create_asset_icon(user32, icon_path)
        if custom_icon:
            data.hIcon = custom_icon
        if not shell32.Shell_NotifyIconW(0, ctypes.byref(data)):  # NIM_ADD
            raise OSError("Windows did not accept the notification icon.")
        added = True
        data.uVersion = 4
        shell32.Shell_NotifyIconW(4, ctypes.byref(data))  # NIM_SETVERSION
        data.uFlags = 0x10  # NIF_INFO
        data.szInfoTitle, data.szInfo = _clean(title, 48), _clean(body, 199)
        data.dwInfoFlags = (0x03 if event == "error" else 0x01) | 0x10 | 0x80
        if custom_icon:
            data.hBalloonIcon = custom_icon
            data.dwInfoFlags = 0x04 | 0x20 | 0x10 | 0x80  # NIIF_USER | LARGE_ICON | NOSOUND | QUIET
        # NIIF_NOSOUND keeps desktop/sound preferences independent. Quiet time
        # is respected. Display duration is ultimately controlled by Windows.
        if not shell32.Shell_NotifyIconW(1, ctypes.byref(data)):  # NIM_MODIFY
            raise OSError("Windows did not accept the desktop notification.")
        deadline = time.monotonic() + min(WORKER_SECONDS, max(0.0, float(duration)))
        message = wt.MSG()
        dismissed = False
        while time.monotonic() < deadline and not dismissed:
            while user32.PeekMessageW(ctypes.byref(message), window, 0, 0, 1):
                if message.message == data.uCallbackMessage and (message.lParam & 0xffff) in (0x403, 0x404, 0x405):
                    dismissed = True
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            if not dismissed:
                time.sleep(0.05)
        return True
    finally:
        try:
            if added:
                shell32.Shell_NotifyIconW(2, ctypes.byref(data))  # NIM_DELETE
        finally:
            try:
                user32.DestroyWindow(window)
            finally:
                if custom_icon:
                    user32.DestroyIcon(custom_icon)
        # LoadIcon returns shared stock resources, which must not be destroyed.



APP_ID = "IlariaRVC.Training"


def _register_identity(icon_path):
    import winreg
    import shutil
    folder = Path(os.environ["LOCALAPPDATA"]) / "IlariaRVC" / "Notifications"
    folder.mkdir(parents=True, exist_ok=True)
    icon = folder / "icon.png"
    if not icon_path or not Path(icon_path).is_file():
        raise OSError("Ilaria notification artwork is missing.")
    shutil.copyfile(str(icon_path), str(icon))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Software\\Classes\\AppUserModelId\\" + APP_ID) as key:
        winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Ilaria RVC")
        winreg.SetValueEx(key, "IconUri", 0, winreg.REG_SZ, str(icon))
        winreg.SetValueEx(key, "IconBackgroundColor", 0, winreg.REG_SZ, "#53355C")
    return icon


def _show_desktop(title, body, event, duration=WORKER_SECONDS, icon_path=None):
    """Native Windows toast with a dedicated per-user name and header icon."""
    import base64
    import xml.etree.ElementTree as ET
    icon = _register_identity(icon_path)
    toast = ET.Element("toast", {"duration": "short"})
    binding = ET.SubElement(ET.SubElement(toast, "visual"), "binding", {"template": "ToastGeneric"})
    ET.SubElement(binding, "text").text = _clean(title, 48)
    ET.SubElement(binding, "text").text = _clean(body, 199)
    ET.SubElement(binding, "image", {"placement": "appLogoOverride", "src": icon.as_uri()})
    ET.SubElement(toast, "audio", {"silent": "true"})
    xml = base64.b64encode(ET.tostring(toast, encoding="utf-8")).decode("ascii")
    script = """$ErrorActionPreference='Stop'
[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime] | Out-Null
$doc=New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('XML_PAYLOAD')))
$toast=[Windows.UI.Notifications.ToastNotification]::new($doc)
$toast.ExpirationTime=[DateTimeOffset]::Now.AddMinutes(5)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('IlariaRVC.Training').Show($toast)
""".replace("XML_PAYLOAD", xml)
    command = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    completed = subprocess.run([str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", command],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode:
        raise OSError("Windows could not submit the Ilaria RVC toast notification.")
    return True


def deliver(payload):
    """Worker entry point: each channel fails independently and cleans up."""
    if os.name != "nt":
        return False
    event = payload.get("event", "test")
    if event not in ("completed", "error", "test"):
        return False
    sound_ok = desktop_ok = False
    if payload.get("sound") is True:
        try:
            _play_sound(event)
            sound_ok = True
        except Exception:
            pass
    if payload.get("desktop") is True:
        try:
            desktop_ok = bool(_show_desktop(payload.get("title", "Ilaria RVC"), payload.get("body", ""), event,
                                             icon_path=payload.get("icon_path")))
        except Exception:
            pass
    return sound_ok or desktop_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliver", required=True)
    arguments = parser.parse_args()
    try:
        value = json.loads(arguments.deliver)
        raise SystemExit(0 if isinstance(value, dict) and deliver(value) else 1)
    except (ValueError, TypeError):
        raise SystemExit(1)
