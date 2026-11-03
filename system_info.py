import os
import platform
import shutil
import subprocess

def _run(cmd, timeout=5):
    try:
        if isinstance(cmd, str):
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
        else:
            return subprocess.check_output(cmd, shell=False, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except Exception:
        return ""

def _first_line(s: str) -> str:
    s = (s or "").strip()
    return s.splitlines()[0].strip() if s else ""

def _which(name: str) -> str:
    return shutil.which(name) or ""

def retrieve_system_info():
    sysname = platform.system()
    machine = platform.machine() or ""
    release = platform.release() or ""
    
    brand = ""
    if sysname == "Windows":
        brand = _run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"')
        if not brand:
            brand = _run("wmic cpu get Name /value").replace("Name=", "").strip()
    elif sysname == "Linux":
        brand = _run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
    elif sysname == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])

    cores_logical = os.cpu_count() or 0
    cores_physical = cores_logical // 2 if cores_logical > 1 else 1

    return {
        "os": {"system": sysname, "arch": machine, "release": release},
        "cpu": {"brand": brand.strip(), "cores_logical": cores_logical, "cores_physical": cores_physical},
        "compilers": {
            "gcc": _first_line(_run(["gcc", "--version"])),
            "g++": _first_line(_run(["g++", "--version"])),
            "rustc": _first_line(_run(["rustc", "--version"]))
        }
    }
