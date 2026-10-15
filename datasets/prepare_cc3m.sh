#!/usr/bin/env bash
# Download the first part of Conceptual Captions (CC3M) as webdataset shards.
#
# The training-split TSV must be downloaded manually (Google requires accepting
# the terms): https://ai.google.com/research/ConceptualCaptions/download
# ("Training split", Train_GCC-training.tsv), placed at $DATA_DIR/cc3m_raw/.
#
# The prior is trained on 100k samples (90k train split), so only the first
# CC3M_ROWS rows are fetched; ~30-40% of the original URLs no longer resolve.
#   prepare_cc3m.sh <Train_GCC-training.tsv> <output dir>
set -euo pipefail

TSV="${1:?usage: prepare_cc3m.sh <Train_GCC-training.tsv> <out dir>}"
OUT="${2:?usage: prepare_cc3m.sh <Train_GCC-training.tsv> <out dir>}"
ROWS="${CC3M_ROWS:-170000}"

[ -f "$TSV" ] || { echo "missing $TSV (see the header of this script)" >&2; exit 1; }
mkdir -p "$OUT"
LIST="$OUT/../cc3m_first_${ROWS}.tsv"
{ printf 'caption\turl\n'; head -n "$ROWS" "$TSV"; } > "$LIST"

/opt/img2dataset/bin/img2dataset \
    --url_list "$LIST" --input_format tsv \
    --url_col url --caption_col caption \
    --output_format webdataset --output_folder "$OUT" \
    --number_sample_per_shard 1000 \
    --image_size 256 --resize_mode keep_ratio \
    --processes_count "$(nproc)" --thread_count 64 \
    --enable_wandb False

python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
stats = sorted(Path(sys.argv[1]).glob("*_stats.json"))
n = sum(json.loads(p.read_text())["successes"] for p in stats)
print(f"{n} images in {len(stats)} shards (the experiment needs 100000)")
EOF

# Fetch the CLIP ViT-B/32 weights once into the shared cache (~/.cache/clip is
# mounted from $DATA_DIR/cache), so collocated processes do not race to download them.
python3 -c "import clip; clip.load('ViT-B/32', device='cpu'); print('CLIP ViT-B/32 cached')"
