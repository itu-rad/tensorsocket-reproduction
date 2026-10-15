import torch.nn.functional as F
from torchtune.data import padded_collate_sft
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

# Must match tokenizer.max_seq_len in config.yaml (the tokenizer truncates to it).
MAX_SEQ_LEN = 512


def padded_collate_fixed(batch, padding_idx: int = 0, ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX):
    """``padded_collate_sft`` padded to ``MAX_SEQ_LEN`` so every batch has the same shape."""
    out = padded_collate_sft(batch, padding_idx=padding_idx, ignore_idx=ignore_idx)
    pad = MAX_SEQ_LEN - out["tokens"].shape[1]
    if pad > 0:
        out["tokens"] = F.pad(out["tokens"], (0, pad), value=padding_idx)
        out["labels"] = F.pad(out["labels"], (0, pad), value=ignore_idx)
    return out
