"""Generate a small ImageNet-shaped dataset of random JPEGs.

Only used by the smoke test (``./reproduce.sh smoke``), which checks the
toolchain end to end without the real dataset. Its throughput numbers are not
comparable to the paper.

    python datasets/make_synthetic_imagenet.py /data/synthetic_imagenet [--classes 10 --per-class 64]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for c in range(args.classes):
        d = args.out / "train" / f"n{c:08d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            p = d / f"n{c:08d}_{i}.JPEG"
            if p.exists():
                continue
            img = rng.integers(0, 255, (args.size, args.size, 3), dtype=np.uint8)
            Image.fromarray(img).save(p, quality=90)
    print(f"wrote {args.classes * args.per_class} images to {args.out / 'train'}")


if __name__ == "__main__":
    main()
