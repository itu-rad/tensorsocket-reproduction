"""Download LibriSpeech train-clean-100 and test-clean as CLMR reads them.

CLMR's librispeech dataset wraps torchaudio.datasets.LIBRISPEECH with
subsets train -> train-clean-100 and test -> test-clean.

    python datasets/prepare_librispeech.py /data/librispeech
"""

import sys
from pathlib import Path

from torchaudio.datasets import LIBRISPEECH


def main(root: str) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    for url in ("train-clean-100", "test-clean"):
        ds = LIBRISPEECH(root=root, url=url, download=True)
        print(f"{url}: {len(ds)} utterances")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
