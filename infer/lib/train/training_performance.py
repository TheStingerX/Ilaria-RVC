"""Windows performance policy scoped to RVC training processes only."""
import atexit
import os


class _WindowsAPI:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        class PowerState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        self.PowerState = PowerState
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.process = self.kernel.GetCurrentProcess()
        self.kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
        self.kernel.GetPriorityClass.restype = wintypes.DWORD
        self.kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.SetPriorityClass.restype = wintypes.BOOL
        for name in ("GetProcessInformation", "SetProcessInformation"):
            fn = getattr(self.kernel, name, None)
            if fn is not None:
                fn.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
                fn.restype = wintypes.BOOL

    def _check(self, result):
        if not result:
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def get_priority(self):
        value = self.kernel.GetPriorityClass(self.process)
        self._check(value)
        return value

    def set_priority(self, value):
        self._check(self.kernel.SetPriorityClass(self.process, value))

    def get_power(self):
        value = self.PowerState(1, 0, 0)
        self._check(self.kernel.GetProcessInformation(
            self.process, 4, self.ctypes.byref(value), self.ctypes.sizeof(value)))
        return value.Version, value.ControlMask, value.StateMask

    def set_power(self, state):
        value = self.PowerState(*state)
        self._check(self.kernel.SetProcessInformation(
            self.process, 4, self.ctypes.byref(value), self.ctypes.sizeof(value)))


class TrainingProcessPolicy:
    """Request HighQoS; leave CPU scheduling priority unchanged by default."""
    def __init__(self, api, boost_priority=False):
        self.api = api
        self.boost_priority = boost_priority
        self.previous_power = None
        self.previous_priority = None
        self.power_changed = False
        self.priority_changed = False

    def apply(self):
        report = {"power_throttling_disabled": False, "priority": "unchanged", "notes": []}
        try:
            self.previous_power = self.api.get_power()
        except Exception as error:
            # Some Windows versions support setting this policy but not querying it.
            report["notes"].append("Power policy query unavailable: " + str(error))
        try:
            old = self.previous_power or (1, 0, 0)
            self.api.set_power((1, old[1] | 1, old[2] & ~1))
            self.power_changed = True
            report["power_throttling_disabled"] = True
        except Exception as error:
            report["notes"].append("Power policy unchanged: " + str(error))
        try:
            priority = self.api.get_priority()
            if self.boost_priority and priority in (0x20, 0x40, 0x4000):  # Normal, Idle, Below Normal
                self.api.set_priority(0x8000)  # Above Normal, never High/Realtime
                self.previous_priority = priority
                self.priority_changed = True
                report["priority"] = "above normal"
            else:
                report["priority"] = "existing priority retained"
        except Exception as error:
            report["notes"].append("CPU priority unchanged: " + str(error))
        return report

    def restore(self):
        if self.power_changed and self.previous_power is not None:
            try:
                self.api.set_power(self.previous_power)
                self.power_changed = False
            except Exception:
                pass
        if self.priority_changed:
            try:
                self.api.set_priority(self.previous_priority)
                self.priority_changed = False
            except Exception:
                pass


_policies = {}


def configure_training_process(worker_id=None):
    """Apply once per PID; also a picklable DataLoader worker_init_fn."""
    if os.name != "nt":
        return {"supported": False}
    if os.getenv("ILARIA_TRAINING_QOS", "1").strip().lower() in ("0", "off", "false"):
        return {"supported": False, "notes": ["Background QoS disabled for this run"]}
    pid = os.getpid()
    if pid in _policies:
        return _policies[pid][1]
    try:
        policy = TrainingProcessPolicy(
            _WindowsAPI(),
            boost_priority=os.getenv("ILARIA_TRAINING_PRIORITY", "normal").strip().lower() == "above_normal",
        )
        report = policy.apply()
        _policies[pid] = (policy, report)
        atexit.register(policy.restore)
        return report
    except Exception as error:
        # Failure of an optional OS optimization must not stop a training job.
        return {"supported": False, "notes": [str(error)]}
