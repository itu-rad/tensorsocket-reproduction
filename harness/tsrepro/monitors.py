"""System monitors sampled once per second while a workload runs.

Every monitor appends rows ``timestamp,entity,key,value`` to its own CSV file,
where ``timestamp`` is in milliseconds (same clock as the tracker) and
``entity`` is the GPU index for GPU metrics and ``-1`` for host metrics.

Metric keys follow the radT listener names used in the paper's analysis:

* ``TOP - CPU Utilization``: summed %CPU of every process visible to the
  monitor (inside the container: all workload processes and their data-loading
  workers). 100 equals one fully used core, as reported by ``top``.
* ``DCGMI - *``: DCGM profiling fields (SM activity, PCIe/NVLink bytes/s).
* ``SMI - *``: ``nvidia-smi`` utilisation and memory.
* ``IO - MB read/s``: host-wide block-device reads.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time

import psutil


def _now_ms() -> int:
    return int(time.time() * 1000)


class _Monitor:
    name = "monitor"

    def __init__(self, out_dir: str, interval: float = 1.0):
        self.path = os.path.join(out_dir, f"monitor_{self.name}.csv")
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._fh = open(self.path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timestamp", "entity", "key", "value"])

    def write(self, ts: int, entity: int, key: str, value: float) -> None:
        self._writer.writerow([ts, entity, key, value])

    def start(self) -> "_Monitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._fh.close()

    def _loop(self) -> None:
        raise NotImplementedError


class CpuMonitor(_Monitor):
    name = "cpu"

    def _loop(self) -> None:
        # psutil needs the same Process objects across samples to compute
        # cpu_percent deltas; short-lived processes simply drop out.
        procs: dict[int, psutil.Process] = {}
        while not self._stop.is_set():
            seen = set()
            total = 0.0
            for p in psutil.process_iter():
                seen.add(p.pid)
                proc = procs.setdefault(p.pid, p)
                try:
                    total += proc.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            for pid in list(procs):
                if pid not in seen:
                    procs.pop(pid)
            vm = psutil.virtual_memory()
            ts = _now_ms()
            self.write(ts, -1, "TOP - CPU Utilization", round(total, 2))
            self.write(ts, -1, "TOP - Memory Used GB", round(vm.used / 1024**3, 3))
            self._stop.wait(self.interval)


def _whole_block_devices() -> set[str]:
    # /sys/block lists whole devices only, so partitions are not double counted.
    try:
        names = os.listdir("/sys/block")
    except FileNotFoundError:
        return set()
    return {n for n in names if not n.startswith(("loop", "ram", "dm-", "zram"))}


class DiskMonitor(_Monitor):
    name = "io"

    def _loop(self) -> None:
        devices = _whole_block_devices()

        def read_bytes() -> int:
            counters = psutil.disk_io_counters(perdisk=True) or {}
            return sum(c.read_bytes for d, c in counters.items() if not devices or d in devices)

        last, last_t = read_bytes(), time.time()
        while not self._stop.wait(self.interval):
            cur, cur_t = read_bytes(), time.time()
            rate = (cur - last) / max(cur_t - last_t, 1e-6) / 1e6
            self.write(_now_ms(), -1, "IO - MB read/s", round(rate, 3))
            last, last_t = cur, cur_t


class _ProcessMonitor(_Monitor):
    """Runs a long-lived command and parses each stdout line."""

    command: list[str] = []

    def __init__(self, out_dir: str, interval: float = 1.0):
        super().__init__(out_dir, interval)
        self._proc: subprocess.Popen | None = None

    def _loop(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                self.parse(_now_ms(), line)
            except (ValueError, IndexError):
                continue

    def parse(self, ts: int, line: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        super().stop()


class SmiMonitor(_ProcessMonitor):
    name = "smi"
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        "-lms",
        "1000",
    ]

    def parse(self, ts: int, line: str) -> None:
        idx, util, mem = [x.strip() for x in line.split(",")]
        self.write(ts, int(idx), "SMI - GPU Util", float(util))
        self.write(ts, int(idx), "SMI - Mem Used", float(mem))


# DCGM field id -> radT metric key. PCIe/NVLink fields are bytes per second.
DCGM_FIELDS = {
    1002: "DCGMI - SM Active",
    203: "DCGMI - GPU Utilization",
    1009: "DCGMI - PCIE TX Bytes",
    1010: "DCGMI - PCIE RX Bytes",
    1011: "DCGMI - NVlink TX Bytes",
    1012: "DCGMI - NVlink RX Bytes",
    252: "DCGMI - FB Used",
}


class DcgmMonitor(_ProcessMonitor):
    name = "dcgm"
    command = [
        "dcgmi",
        "dmon",
        "-e",
        ",".join(str(f) for f in DCGM_FIELDS),
        "-d",
        "1000",
    ]

    def parse(self, ts: int, line: str) -> None:
        parts = line.split()
        if not parts or parts[0] != "GPU":
            return
        gpu = int(parts[1])
        for key, raw in zip(DCGM_FIELDS.values(), parts[2:]):
            if raw != "N/A":
                self.write(ts, gpu, key, float(raw))


def dcgm_available() -> bool:
    if shutil.which("dcgmi") is None:
        return False
    try:
        out = subprocess.run(
            ["dcgmi", "dmon", "-e", "1002", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0 and any(l.split()[:1] == ["GPU"] for l in out.stdout.splitlines())


def start_monitors(out_dir: str, gpu: bool = True) -> list[_Monitor]:
    monitors: list[_Monitor] = [CpuMonitor(out_dir), DiskMonitor(out_dir)]
    if gpu and shutil.which("nvidia-smi"):
        monitors.append(SmiMonitor(out_dir))
        if dcgm_available():
            monitors.append(DcgmMonitor(out_dir))
        else:
            print(
                "[tsrepro] WARNING: dcgmi unavailable; SM activity, PCIe and NVLink "
                "metrics will be missing (see README: DCGM).",
                flush=True,
            )
    return [m.start() for m in monitors]


def stop_monitors(monitors: list[_Monitor]) -> None:
    for m in monitors:
        m.stop()
