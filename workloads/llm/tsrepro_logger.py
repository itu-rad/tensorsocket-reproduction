from torchtune.training.metric_logging import MetricLoggerInterface

from tsrepro import tracking

KEYS = {
    "tokens_per_second_per_gpu": "ML - Tokens/s",
    "loss": "ML - Loss",
    "peak_memory_alloc": "PT - Peak Mem Allocated GB",
    "peak_memory_reserved": "PT - Peak Mem Reserved GB",
}


class TsreproLogger(MetricLoggerInterface):
    """torchtune metric logger that writes to the tsrepro tracker (step = optimizer step)."""

    def __init__(self, **kwargs):
        self._run = tracking.run()

    def log(self, name, data, step):
        if name in KEYS:
            self._run.log_metric(KEYS[name], float(data), step)

    def log_dict(self, payload, step):
        for name, value in payload.items():
            self.log(name, value, step)

    def log_config(self, config):
        pass

    def close(self):
        self._run.close()
