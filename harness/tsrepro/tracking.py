"""Minimal metric tracker used by the workload scripts.

The paper's runs logged through radT (an MLflow extension). The analysis only
needs the metric stream, so this module writes the same long format that an
MLflow metric export has (``run_id,key,value,step,timestamp``) to the CSV file
named by ``TSREPRO_METRICS_FILE``. Metric keys are kept identical to the radT
keys (e.g. ``ML - Iterations/s``, ``ML - Epoch``), so the original analysis
notebooks and ``tsrepro.results`` read both sources alike.

When a script runs under radT itself (``RADT_PRESENT`` is set by radT), the
tracker defers to radT so the same workload code can be used with either
harness.
"""

from __future__ import annotations

import csv
import os
import threading
import time
import uuid


class CsvRun:
    def __init__(self, path: str | None):
        self.run_id = os.environ.get("TSREPRO_RUN_ID", uuid.uuid4().hex)
        self._lock = threading.Lock()
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            new = not os.path.exists(path)
            self._fh = open(path, "a", newline="", buffering=1)
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(["run_id", "key", "value", "step", "timestamp"])

    def log_metric(self, key: str, value, step: int = 0) -> None:
        if self._fh is None:
            return
        ts = int(time.time() * 1000)
        with self._lock:
            self._writer.writerow([self.run_id, key, float(value), int(step), ts])

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def close(self) -> None:
        if self._fh is not None:
            with self._lock:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def run():
    """Context manager yielding an object with ``log_metric``/``log_metrics``."""
    if os.environ.get("RADT_PRESENT"):
        import radt

        return radt.run.RADTBenchmark()
    return CsvRun(os.environ.get("TSREPRO_METRICS_FILE"))
