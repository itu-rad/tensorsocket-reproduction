"""Load the results written by ``tsrepro.run`` and compute the paper's metrics.

Layout: ``results/<experiment>/<tag>/w<workload>/rep<k>/`` containing
``meta.json``, ``proc<i>_<role>.metrics.csv`` (tracker output) and
``monitor_<name>.csv`` (system monitors).

Aggregation follows the paper's analysis: throughput is the mean of
``ML - Iterations/s`` logged during one steady-state epoch (the third epoch,
index 2, for the image-classification runs), and hardware metrics are averaged
over the same time window.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TSREPRO_RESULTS", REPO / "results"))
RESULTS_FAST = Path(os.environ.get("TSREPRO_RESULTS_FAST", REPO / "results_fast"))
PAPER = REPO / "paper_results"

THROUGHPUT = "ML - Iterations/s"
WINDOW_THROUGHPUT = "ML - Window Samples/s"

# Fast mode (runs stopped after a few minutes): metrics are taken over the last
# two thirds of each run instead of a steady-state epoch.
FAST = False


def set_fast(fast: bool) -> None:
    global FAST, RESULTS
    FAST = fast
    RESULTS = RESULTS_FAST if fast else Path(os.environ.get("TSREPRO_RESULTS", REPO / "results"))


@dataclass
class Proc:
    index: int
    role: str
    devices: str
    path: Path

    @cached_property
    def metrics(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["run_id", "key", "value", "step", "timestamp"])
        return pd.read_csv(self.path)

    def series(self, key: str) -> pd.DataFrame:
        m = self.metrics
        return m[m.key == key]

    def epoch_window(self, epoch: int, key: str = THROUGHPUT) -> tuple[int, int] | None:
        """[first, last] timestamp of ``key`` rows logged during ``epoch``."""
        s = self.series(key)
        s = s[s.step == epoch]
        if s.empty:
            return None
        return int(s.timestamp.min()), int(s.timestamp.max())

    def throughput(self, epoch: int | None, key: str = THROUGHPUT) -> float:
        """Mean of ``key`` during ``epoch`` (all values when ``epoch`` is None)."""
        s = self.series(key)
        if epoch is not None:
            s = s[s.step == epoch]
        return float(s.value.mean()) if len(s) else float("nan")

    def median(self, key: str = THROUGHPUT, skip: int = 0) -> float:
        s = self.series(key).sort_values("timestamp").value.iloc[skip:]
        return float(s.median()) if len(s) else float("nan")


@dataclass
class Run:
    path: Path
    meta: dict
    procs: list[Proc] = field(default_factory=list)

    @property
    def workload(self) -> str:
        return self.meta["workload"]

    @property
    def ok(self) -> bool:
        return self.meta.get("status") == "ok"

    @property
    def ncpu(self) -> int:
        return int(self.meta.get("ncpu", os.cpu_count() or 1))

    @property
    def trainers(self) -> list[Proc]:
        return [p for p in self.procs if p.role == "train"]

    @property
    def producers(self) -> list[Proc]:
        return [p for p in self.procs if p.role == "producer"]

    @cached_property
    def monitors(self) -> pd.DataFrame:
        frames = [pd.read_csv(p) for p in sorted(self.path.glob("monitor_*.csv"))]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["timestamp", "entity", "key", "value"])
        return pd.concat(frames, ignore_index=True)

    def monitor(self, key: str, window: tuple[int, int] | None = None, entity: int | None = None) -> pd.DataFrame:
        m = self.monitors
        m = m[m.key == key]
        if entity is not None:
            m = m[m.entity == entity]
        if window is not None:
            m = m[(m.timestamp >= window[0]) & (m.timestamp <= window[1])]
        return m

    def monitor_mean(self, key: str, window=None, entity: int | None = None) -> float:
        m = self.monitor(key, window, entity)
        return float(m.value.mean()) if len(m) else float("nan")

    def fast_window(self) -> tuple[int, int]:
        """Last two thirds of the time trainers were logging; the first third is warm-up."""
        starts = [int(p.metrics.timestamp.min()) for p in self.trainers if len(p.metrics)]
        end = int(self.meta["end_ms"])
        t0 = min(starts) if starts else int(self.meta["start_ms"])
        return t0 + (end - t0) // 3, end

    def window(self, epoch: int | None) -> tuple[int, int]:
        """Steady-state window: ``epoch`` of the first trainer, else the whole run."""
        if FAST:
            return self.fast_window()
        if epoch is not None:
            for p in self.trainers:
                w = p.epoch_window(epoch)
                if w:
                    return w
        return int(self.meta["start_ms"]), int(self.meta["end_ms"])

    def cpu_util_pct(self, window) -> float:
        """Mean summed %CPU as a share of the CPUs available to the container."""
        return self.monitor_mean("TOP - CPU Utilization", window) / self.ncpu

    def gpu_ids(self) -> list[int]:
        ids = set()
        for p in self.procs:
            ids.update(int(d) for d in str(p.devices).split(",") if d.strip().isdigit())
        return sorted(ids)

    def proc_rate(self, p: Proc, epoch: int | None) -> float:
        if not FAST:
            return p.throughput(epoch)
        lo, hi = self.fast_window()
        for key in (WINDOW_THROUGHPUT, THROUGHPUT):
            s = p.series(key)
            s = s[(s.timestamp >= lo) & (s.timestamp <= hi)]
            if len(s):
                return float(s.value.mean())
        return float("nan")

    def proc_median(self, p: Proc, key: str | None = None, skip: int = 0) -> float:
        if not FAST:
            return p.median(key or THROUGHPUT, skip)
        lo, hi = self.fast_window()
        for k in [key] if key else [WINDOW_THROUGHPUT, THROUGHPUT]:
            s = p.series(k)
            s = s[(s.timestamp >= lo) & (s.timestamp <= hi)]
            if len(s):
                return float(s.value.median())
        return float("nan")

    def mean_throughput(self, epoch: int | None) -> float:
        vals = [self.proc_rate(p, epoch) for p in self.trainers]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")


def hand_back(path: Path) -> None:
    """chown files written as root in a container to the host user (HOST_UID/HOST_GID)."""
    uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if not uid or os.geteuid() != 0 or not path.exists():
        return
    for p in [path, *path.rglob("*")]:
        try:
            os.chown(p, int(uid), int(gid or uid))
        except OSError:
            pass


def load_runs(experiment: str, tag: str | None = None, workload: str | None = None) -> list[Run]:
    """All completed runs of an experiment (optionally one tag / workload)."""
    root = RESULTS / experiment
    if not root.exists():
        return []
    runs = []
    for meta_path in sorted(root.glob("*/w*/rep*/meta.json")):
        rep_dir = meta_path.parent
        if tag is not None and rep_dir.parents[1].name != tag:
            continue
        meta = json.loads(meta_path.read_text())
        if workload is not None and str(meta["workload"]) != str(workload):
            continue
        run = Run(rep_dir, meta | {"tag": rep_dir.parents[1].name})
        for p in meta["processes"]:
            run.procs.append(
                Proc(p["index"], p["role"], p["devices"], rep_dir / f"proc{p['index']}_{p['role']}.metrics.csv")
            )
        if run.ok:
            runs.append(run)
        else:
            print(f"[tsrepro] ignoring failed run {rep_dir}")
    return runs


def by_workload(runs: list[Run]) -> dict[str, list[Run]]:
    out: dict[str, list[Run]] = {}
    for r in runs:
        out.setdefault(str(r.workload), []).append(r)
    return out


def tags(experiment: str) -> list[str]:
    root = RESULTS / experiment
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def mean_over_reps(values: list[float]) -> float:
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def paper_values(figure: str) -> pd.DataFrame:
    df = pd.read_csv(PAPER / "paper_values.csv", keep_default_na=False)
    df = df[df.figure == figure].copy()
    df["value"] = df["value"].astype(float)
    return df
