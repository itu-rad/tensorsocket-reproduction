"""Register an ImageNet LMDB (see convert_imagenet.py) with a running Joader server."""

import sys
from pathlib import Path

import grpc
import lmdb

CLIENT = Path(__file__).resolve().parents[1] / "timm" / "joader"
sys.path[:0] = [str(CLIENT), str(CLIENT / "proto")]

import dataset.dataset as ds  # noqa: E402  (Joader's client, imported from CLIENT)


def main(path: str, ip: str = "127.0.0.1:4321") -> None:
    env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    n = env.stat()["entries"]
    env.close()

    channel = grpc.insecure_channel(ip, options=(("grpc.enable_http_proxy", 0),))
    d = ds.Dataset(name="ImageNet", location=path, ty=ds.DatasetType.LMDB)
    for i in range(n):
        d.add_item([str(i)])
    d.create(channel)
    print(f"registered ImageNet ({n} items) from {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
