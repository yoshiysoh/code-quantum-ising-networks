#!/usr/bin/env bash
# Class fractions, data/classes_N*_{ferro,pmJ}.h5 (hours to days).
#
#   nohup bash run/classes.sh &
#
# Runs from the repository root. WORKERS processes (default: the physical
# cores), each with one BLAS thread, as for the published files. Writes to
# OUT_DIR (default data/<today>/) and skips files that already exist there.
# The class rule reads data/revival_null.h5.
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

for n in 8 9 10 11 12 13 14 15; do
    nets=200
    if (( n >= 14 )); then nets=100; fi
    for model in ferro pmJ; do
        out="$OUT_DIR/classes_N${n}_${model}.h5"
        if [[ -e "$out" ]]; then echo "skip $out"; continue; fi
        python3 scripts/generate_class_dataset.py --N "$n" --model "$model" \
            --nets "$nets" --out-dir "$OUT_DIR" \
            > "logs/classes_N${n}_${model}.log" 2>&1
    done
done
