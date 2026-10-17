#!/usr/bin/env bash
# Spectral statistics, data/spectral_data.h5 and data/spectral_data_extra.h5
# (hours to days), then data/spectral_extrapolation.h5.
#
#   nohup bash run/spectral.sh &
#
# Runs from the repository root. WORKERS processes (default: the physical
# cores), each with one BLAS thread, as for the published files. Writes to
# OUT_DIR (default data/<today>/) and skips files that already exist there.
# extrapolate_spectral_data.py reduces the files in data/ and writes to
# data/<today>/; copy the new spectral and class files into data/ first to
# extrapolate them.
set -euo pipefail
cd "$(dirname "$0")/.."

physical_cores() {
    if [[ "$(uname)" == Darwin ]]; then sysctl -n hw.physicalcpu
    else lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l; fi
}
WORKERS="${WORKERS:-$(physical_cores)}"
OUT_DIR="${OUT_DIR:-data/$(date +%F)}"
export NUM_WORKERS="$WORKERS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
mkdir -p "$OUT_DIR" logs

if [[ ! -e "$OUT_DIR/spectral_data.h5" ]]; then
    python3 scripts/generate_spectral_data.py --workers "$WORKERS" \
        --out-dir "$OUT_DIR" > logs/spectral_data.log 2>&1
fi
if [[ ! -e "$OUT_DIR/spectral_data_extra.h5" ]]; then
    python3 scripts/generate_spectral_data.py --extra 200 \
        --workers "$WORKERS" --out-dir "$OUT_DIR" \
        > logs/spectral_data_extra.log 2>&1
fi
python3 scripts/extrapolate_spectral_data.py \
    > logs/spectral_extrapolation.log 2>&1
