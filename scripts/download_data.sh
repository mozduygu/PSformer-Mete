#!/usr/bin/env bash
# Downloads the standard long-term time-series forecasting datasets used by
# PSformer. These are the canonical Autoformer/Informer/PatchTST datasets.
#
# Files expected in $1 (default: ./datasets/) :
#   ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
#   weather.csv
#   electricity.csv
#   exchange_rate.csv
#   traffic.csv
#
# Two reliable sources:
#  (1) Tsinghua's timeseries-library (recommended): a Tsinghua cloud bundle
#      and a Google Drive bundle, both link from the README at
#      https://github.com/thuml/Time-Series-Library
#  (2) ETT only: https://github.com/zhouhaoyi/ETDataset (CSVs in /ETT-small)
#
# This script tries (2) first for ETT (small, simple), and prints clear
# instructions for (1) for the rest.

set -e
DEST="${1:-./datasets}"
mkdir -p "$DEST"
cd "$DEST"

echo ">> Downloading ETT files from zhouhaoyi/ETDataset..."
BASE="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
for f in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
  if [ ! -f "$f" ]; then
    echo "  fetching $f"
    curl -fsSL "$BASE/$f" -o "$f"
  else
    echo "  $f already present, skipping"
  fi
done

echo
echo ">> ETT files downloaded."
echo
echo "Now you need: weather.csv, electricity.csv, exchange_rate.csv, traffic.csv"
echo "These are bundled by the Autoformer authors. Easiest options:"
echo
echo "  Option A (recommended) -- Tsinghua Time-Series-Library README links:"
echo "    https://github.com/thuml/Time-Series-Library"
echo "    Look for 'all_six_datasets.zip' or the Google Drive link."
echo "    Unzip into $DEST/ . Each dataset is in its own subfolder; flatten them"
echo "    so the CSVs sit directly in $DEST/ (matching dataset.py's filename map)."
echo
echo "  Option B -- HuggingFace mirror (no auth needed):"
echo "    huggingface-cli download thuml/time-series-data --repo-type dataset"
echo "    (then move the relevant .csv files into $DEST/)"
echo
echo ">> After all eight CSVs are in $DEST/, you're ready to train."
ls -la "$DEST"
