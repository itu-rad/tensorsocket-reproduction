"""Create the 10% ImageNet subset used on the AWS g5 instances (Figure 13).

The subset is the fixed list of 128,115 training images in
``imagenet_10pct.txt`` (the list used for the paper). Files are hard-linked
when source and destination share a file system, and copied otherwise.

    python datasets/make_imagenet_subset.py /data/imagenet /data/imagenet_10pct
"""

import os
import shutil
import sys
from pathlib import Path


def main(src: Path, dst: Path, listing: Path) -> None:
    names = [l.strip() for l in listing.read_text().splitlines() if l.strip()]
    for i, name in enumerate(names):
        wnid = name.split("_")[0]
        target = dst / "train" / wnid / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = src / "train" / wnid / name
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        if i % 10000 == 0:
            print(f"{i}/{len(names)}", flush=True)
    print(f"done: {len(names)} images in {dst / 'train'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(__file__).with_name("imagenet_10pct.txt"))
