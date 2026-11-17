"""Record the measurement environment of the latency pass.

Writes results/rev_env.json with what can be read without administrator
rights: CPU affinity of the process, priority class, active Windows power
scheme, AC/battery status, clock used for timing, CPU clock speed, library
versions. Nothing is changed on the machine.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import rev_common as R


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip()
    except Exception as exc:
        return "unavailable (%s)" % type(exc).__name__


def windows_info() -> dict:
    import ctypes
    from ctypes import wintypes
    info = {}
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    h = k32.GetCurrentProcess()
    try:
        proc, syst = ctypes.c_size_t(0), ctypes.c_size_t(0)
        k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]
        if k32.GetProcessAffinityMask(h, ctypes.byref(proc), ctypes.byref(syst)):
            info["affinity_cores_allowed"] = bin(proc.value).count("1")
            info["affinity_cores_system"] = bin(syst.value).count("1")
            info["affinity_pinned"] = proc.value != syst.value
    except Exception as exc:
        info["affinity_error"] = repr(exc)
    try:
        k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k32.GetPriorityClass.restype = wintypes.DWORD
        names = {0x40: "idle", 0x4000: "below normal", 0x20: "normal", 0x8000: "above normal",
                 0x80: "high", 0x100: "realtime"}
        pc = k32.GetPriorityClass(h)
        info["priority_class"] = names.get(pc, hex(pc))
    except Exception as exc:
        info["priority_error"] = repr(exc)
    try:
        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]
        sps = SPS()
        if k32.GetSystemPowerStatus(ctypes.byref(sps)):
            info["on_ac_power"] = {0: False, 1: True}.get(int(sps.ACLineStatus))
    except Exception as exc:
        info["power_status_error"] = repr(exc)
    scheme = _run(["powercfg", "/getactivescheme"])
    info["power_scheme_raw"] = scheme
    if "(" in scheme and scheme.rstrip().endswith(")"):
        info["power_scheme"] = scheme[scheme.rfind("(") + 1: -1].strip()
    info["cpu_clock"] = _run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Processor | "
                              "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
                              "CurrentClockSpeed,MaxClockSpeed | Format-List"])
    return info


def main():
    import numpy, sklearn, torch
    ci = time.get_clock_info("perf_counter")
    env = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "timer": {"function": "time.perf_counter", "implementation": ci.implementation,
                  "resolution_s": ci.resolution, "monotonic": ci.monotonic},
        "torch_threads_for_latency": 1,
        "thermal_state": "not recorded (no unprivileged sensor access)",
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if os.name == "nt":
        try:
            env.update(windows_info())
        except Exception as exc:  # never block the experiments on this record
            env["windows_info_error"] = repr(exc)
    else:
        try:
            aff = os.sched_getaffinity(0)
            env["affinity_cores_allowed"] = len(aff)
            env["affinity_cores_system"] = os.cpu_count()
            env["affinity_pinned"] = len(aff) != os.cpu_count()
        except Exception:
            pass
        env["priority_class"] = "nice %d" % os.nice(0)
    print("saved", R.dump(env, "rev_env.json"))
    for k, v in env.items():
        print("  %-26s %s" % (k, v))


if __name__ == "__main__":
    main()
