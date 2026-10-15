"""Run the workloads of one experiment and record metrics.

An experiment is a CSV file with one row per process::

    workload,role,devices,collocation,file,params

Rows sharing a ``workload`` id run concurrently; workloads run one after the
other. ``role`` is ``service`` or ``producer`` (started first and terminated once
all trainers are done) or ``train`` (the workload is finished when all of them
exit). ``devices`` are GPU indices (``--gpus`` remaps them to physical ids).
``collocation`` is ``mps`` to run the workload under NVIDIA MPS, anything else
for default time-slicing. ``file`` is relative to the repository root.

``params`` may contain ``{expr}`` templates, evaluated with the experiment
variables: dataset paths from the environment (``IMAGENET``, ``LIBRISPEECH``,
...), ``ncpu`` (CPUs available to the container) and the defaults declared in
``#! name=value`` header lines (overridable with ``--var name=value``).

Usage::

    python -m tsrepro.run experiments/fig08_a100_4way.csv [--workloads 5,51]
        [--repeats 2] [--tag a100] [--var epochs=1] [--fast] [--dry-run]

``--fast`` stops every workload after the experiment's ``fast_duration``
seconds and writes to ``results_fast/``: minutes per experiment instead of
hours, measured on the steady part of the shortened run (tsrepro.plot --fast).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import monitors as mon
from .results import hand_back

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TSREPRO_RESULTS", REPO / "results"))
RESULTS_FAST = Path(os.environ.get("TSREPRO_RESULTS_FAST", REPO / "results_fast"))
FAST_DURATION_DEFAULT = 180
MPS_DIR = Path("/tmp/tsrepro-mps")


@dataclass
class Proc:
    role: str
    devices: str
    file: str
    params: str


@dataclass
class Workload:
    id: str
    collocation: str
    procs: list[Proc] = field(default_factory=list)

    @property
    def mps(self) -> bool:
        return self.collocation.strip().lower() == "mps"


def ncpu() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_experiment(path: Path) -> tuple[dict[str, str], list[Workload]]:
    defaults: dict[str, str] = {}
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            k, v = stripped[2:].strip().split("=", 1)
            defaults[k.strip()] = v.strip()
        elif stripped and not stripped.startswith("#"):
            lines.append(line)

    workloads: dict[str, Workload] = {}
    for row in csv.DictReader(lines):
        wid = row["workload"].strip()
        w = workloads.setdefault(wid, Workload(wid, row.get("collocation", "") or ""))
        w.procs.append(
            Proc(
                role=row["role"].strip(),
                devices=row["devices"].strip(),
                file=row["file"].strip(),
                params=(row.get("params") or "").strip(),
            )
        )
    return defaults, list(workloads.values())


_TEMPLATE = re.compile(r"\{([^{}]+)\}")


def render(params: str, variables: dict) -> str:
    def sub(m: re.Match) -> str:
        return str(eval(m.group(1), {"__builtins__": {}}, variables))

    return _TEMPLATE.sub(sub, params)


def build_variables(defaults: dict[str, str], overrides: list[str]) -> dict:
    variables: dict = {"ncpu": ncpu()}
    for k, v in os.environ.items():
        if k.isupper():
            variables[k] = v
    for k, v in defaults.items():
        variables[k] = _coerce(v)
    for item in overrides:
        k, v = item.split("=", 1)
        variables[k.strip()] = _coerce(v.strip())
    return variables


def _coerce(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def map_devices(devices: str, gpu_map: list[str] | None) -> str:
    ids = [d for d in re.split(r"[;,\s]+", devices) if d]
    if gpu_map:
        ids = [gpu_map[int(d)] for d in ids]
    return ",".join(ids)


def mps_env() -> dict[str, str]:
    return {
        "CUDA_MPS_PIPE_DIRECTORY": str(MPS_DIR / "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": str(MPS_DIR / "log"),
    }


def start_mps(devices: str) -> None:
    for sub in ("pipe", "log"):
        (MPS_DIR / sub).mkdir(parents=True, exist_ok=True)
    env = os.environ | mps_env() | {"CUDA_VISIBLE_DEVICES": devices}
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    time.sleep(2)


def stop_mps() -> None:
    env = os.environ | mps_env()
    subprocess.run(
        "echo quit | nvidia-cuda-mps-control",
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def evict_page_cache(root: str) -> None:
    """Drop ``root``'s files from the page cache so reads hit the storage device.

    Uses posix_fadvise(DONTNEED), which needs no privileges, instead of
    /proc/sys/vm/drop_caches.
    """
    t0 = time.time()
    count = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
            finally:
                os.close(fd)
    print(f"[tsrepro] evicted {count} files under {root} from page cache in {time.time() - t0:.0f}s", flush=True)


def terminate(procs: list[subprocess.Popen], timeout: float = 30) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + timeout
    for p in procs:
        try:
            p.wait(timeout=max(deadline - time.time(), 0.1))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def _reports_throughput(path: Path) -> bool:
    try:
        text = path.read_text()
    except OSError:
        return False
    return any(k in text for k in ("Samples/s", "Iterations/s", "Tokens/s"))


def gpu_inventory() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run_workload(
    exp: str,
    w: Workload,
    variables: dict,
    out: Path,
    args: argparse.Namespace,
    root: Path,
) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    gpu_map = args.gpus.split(",") if args.gpus else None
    all_devices = sorted({d for p in w.procs for d in map_devices(p.devices, gpu_map).split(",") if d})

    commands = []
    for i, p in enumerate(w.procs):
        params = render(p.params, variables)
        script = REPO / p.file
        cmd = [sys.executable, str(script), *shlex.split(params)] if script.suffix == ".py" else [str(script), *shlex.split(params)]
        commands.append((i, p, map_devices(p.devices, gpu_map), cmd))

    meta = {
        "experiment": exp,
        "workload": w.id,
        "collocation": w.collocation,
        "mode": "fast" if args.fast else "full",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "ncpu": ncpu(),
        "gpus": gpu_inventory(),
        "variables": {k: v for k, v in variables.items() if not k.isupper() or k in args.record_env},
        "processes": [
            {"index": i, "role": p.role, "devices": dev, "command": shlex.join(cmd)} for i, p, dev, cmd in commands
        ],
    }

    print(f"\n[tsrepro] === {exp} workload {w.id} -> {out}", flush=True)
    for i, p, dev, cmd in commands:
        print(f"[tsrepro]   [{p.role:8s}] gpu={dev:7s} {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return True

    for path in args.evict:
        evict_page_cache(render(path, variables))

    if w.mps:
        stop_mps()
        start_mps(",".join(all_devices))

    monitors = mon.start_monitors(str(out), gpu=bool(all_devices))
    meta["start_ms"] = int(time.time() * 1000)

    background: list[subprocess.Popen] = []
    trainers: list[tuple[int, subprocess.Popen]] = []
    logs = []
    ok = True
    stopped = False
    try:
        for phase in ("service", "producer", "train"):
            phase_cmds = [c for c in commands if c[1].role == phase]
            if phase == "train" and background:
                time.sleep(args.startup_delay)
            for i, p, dev, cmd in phase_cmds:
                env = os.environ | {
                    "CUDA_VISIBLE_DEVICES": dev,
                    "TSREPRO_METRICS_FILE": str(out / f"proc{i}_{p.role}.metrics.csv"),
                    "TSREPRO_RUN_ID": f"{w.id}-{i}",
                    "PYTHONUNBUFFERED": "1",
                }
                if w.mps:
                    env |= mps_env()
                log = open(out / f"proc{i}_{p.role}.log", "w")
                logs.append(log)
                proc = subprocess.Popen(
                    cmd,
                    cwd=(REPO / p.file).parent,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if phase == "train":
                    trainers.append((i, proc))
                else:
                    background.append(proc)

        started = time.time()
        deadline = started + args.timeout if args.timeout else None
        measuring_since = None
        while any(p.poll() is None for _, p in trainers):
            # The duration counts from the moment every trainer reports throughput,
            # so dataset scans and model set-up do not eat into short runs.
            if args.duration and measuring_since is None:
                running = [i for i, p in trainers if p.poll() is None]
                if all(_reports_throughput(out / f"proc{i}_train.metrics.csv") for i in running) or time.time() - started > 900:
                    measuring_since = time.time()
            if args.duration and measuring_since and time.time() - measuring_since > args.duration:
                print(f"[tsrepro] workload {w.id}: stopping trainers after the {args.duration:.0f}s measurement duration", flush=True)
                stopped = True
                break
            if deadline and time.time() > deadline:
                print(f"[tsrepro] workload {w.id} timed out after {args.timeout}s", flush=True)
                ok = False
                break
            crashed = [b for b in background if b.poll() not in (None, 0)]
            if crashed:
                print(f"[tsrepro] a producer/service exited with code {crashed[0].returncode}", flush=True)
                ok = False
                break
            time.sleep(2)
    except KeyboardInterrupt:
        ok = False
        raise
    finally:
        terminate([p for _, p in trainers])
        # Producers loop forever waiting for new consumers once all trainers are done.
        terminate(background)
        meta["end_ms"] = int(time.time() * 1000)
        mon.stop_monitors(monitors)
        if w.mps:
            stop_mps()
        for log in logs:
            log.close()

    codes = {i: p.returncode for i, p in trainers}
    meta["return_codes"] = codes
    meta["stopped_after_duration"] = stopped
    # Trainers stopped by the duration bound exit through SIGTERM.
    ok = ok and all(c == 0 or (stopped and c == -signal.SIGTERM) for c in codes.values())
    meta["status"] = "ok" if ok else "failed"
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    hand_back(root / exp)
    print(f"[tsrepro] workload {w.id}: {meta['status']} in {(meta['end_ms'] - meta['start_ms']) / 1000:.0f}s", flush=True)
    if not ok:
        for i, _ in trainers:
            tail = (out / f"proc{i}_train.log").read_text().splitlines()[-15:]
            print(f"[tsrepro] --- tail of proc{i}_train.log ---\n" + "\n".join(tail), flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path, help="experiment CSV file")
    ap.add_argument("--workloads", help="comma-separated workload ids to run (default: all)")
    ap.add_argument("--repeats", type=int, default=1, help="repetitions per workload (paper: 2)")
    ap.add_argument("--tag", default="default", help="results sub-directory, e.g. the machine/instance name")
    ap.add_argument("--var", action="append", default=[], help="override a template variable, e.g. epochs=1")
    ap.add_argument("--gpus", help="physical GPU ids for logical devices 0,1,2,... (e.g. 0,1,2,4)")
    ap.add_argument("--evict", action="append", default=[], help="directory to evict from the page cache before each workload")
    ap.add_argument(
        "--startup-delay",
        type=float,
        default=None,
        help="seconds between starting producers/services and trainers (default: the experiment's startup_delay, else 15)",
    )
    ap.add_argument("--timeout", type=float, default=0, help="per-workload timeout in seconds; the run counts as failed (0: none)")
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop trainers after this many seconds and count the run as complete "
        "(default: the experiment's duration, else unbounded)",
    )
    ap.add_argument("--fast", action="store_true", help="short runs (fast_duration per workload) into results_fast/")
    ap.add_argument("--force", action="store_true", help="re-run workloads that already completed")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--record-env", nargs="*", default=["IMAGENET", "LIBRISPEECH", "CC3M"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    exp = args.experiment.stem
    defaults, workloads = parse_experiment(args.experiment)
    variables = build_variables(defaults, args.var)
    if args.startup_delay is None:
        args.startup_delay = float(variables.get("startup_delay", 15))
    if args.duration is None:
        key, default = ("fast_duration", FAST_DURATION_DEFAULT) if args.fast else ("duration", 0)
        args.duration = float(variables.get(key, default))
    root = RESULTS_FAST if args.fast else RESULTS
    if args.workloads:
        wanted = [x.strip() for x in args.workloads.split(",")]
        workloads = [w for w in workloads if w.id in wanted]
        missing = set(wanted) - {w.id for w in workloads}
        if missing:
            ap.error(f"unknown workload ids: {sorted(missing)}")

    failures = []
    for rep in range(args.repeats):
        for w in workloads:
            out = root / exp / args.tag / f"w{w.id}" / f"rep{rep}"
            meta = out / "meta.json"
            if meta.exists() and not args.force and not args.dry_run:
                if json.loads(meta.read_text()).get("status") == "ok":
                    print(f"[tsrepro] skipping {exp}/{args.tag}/w{w.id}/rep{rep} (already done; --force to re-run)")
                    continue
            if not run_workload(exp, w, variables, out, args, root):
                failures.append(f"w{w.id}/rep{rep}")

    if failures:
        print(f"\n[tsrepro] {exp}: {len(failures)} failed workload run(s): {', '.join(failures)}", flush=True)
        return 1
    print(f"\n[tsrepro] {exp}: all workloads finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
