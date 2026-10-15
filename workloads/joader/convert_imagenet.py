"""Convert an ImageNet train folder (train/<wnid>/*.JPEG) into Joader's LMDB format.

Joader's LMDB reader expects a single-file LMDB whose keys are ``str(i)`` and
whose values are msgpack ``[{b"data": <JPEG bytes>}, <label>]`` with binary map
keys. Labels are the indices of the sorted class directories, the same mapping
as torchvision/timm ImageFolder.

    python convert_imagenet.py /data/imagenet/train /data/imagenet_joader/ILSVRC-train.lmdb
"""

import os
import sys

import lmdb
import msgpack


def main(train_dir: str, out: str) -> None:
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    samples = [
        (os.path.join(train_dir, c, f), label)
        for label, c in enumerate(classes)
        for f in sorted(os.listdir(os.path.join(train_dir, c)))
    ]
    total_bytes = sum(os.path.getsize(p) for p, _ in samples)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    env = lmdb.open(out, subdir=False, map_size=int(total_bytes * 1.2) + (1 << 30), writemap=True)

    txn = env.begin(write=True)
    for i, (path, label) in enumerate(samples):
        with open(path, "rb") as f:
            value = msgpack.packb([{b"data": f.read()}, label], use_bin_type=True)
        txn.put(str(i).encode(), value)
        if i % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"{i}/{len(samples)}", flush=True)
    txn.commit()
    env.sync()
    env.close()
    print(f"wrote {len(samples)} images ({len(classes)} classes) to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
