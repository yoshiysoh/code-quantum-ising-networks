#!/usr/bin/env bash
# Relaxation-time diagnostic, data/relax_diag_N12_{ferro,pmJ}.h5 (hours).
#
#   nohup bash run/relax_diag.sh &
#
# Runs from the repository root. WORKERS processes (default: the physical
# cores), each with one BLAS thread, as for the published files. Writes to
# OUT_DIR (default data/<today>/) and skips the run if both files exist.
set -euo pipefail
cd "$(dirname "$0")/.."

physical_cores() {
    if [[ "$(uname)" == Darwin ]]; then sysctl -n hw.physicalcpu
    else lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l; fi
}
WORKERS="${WORKERS:-$(physical_cores)}"
OUT_DIR="${OUT_DIR:-data/$(date +%F)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT_DIR" logs

if [[ -e "$OUT_DIR/relax_diag_N12_ferro.h5" &&
      -e "$OUT_DIR/relax_diag_N12_pmJ.h5" ]]; then
    echo "skip: $OUT_DIR/relax_diag_N12_*.h5 exist"
    exit 0
fi
python3 scripts/diagnose_relax_time.py --nets 32 --workers "$WORKERS" \
    --out-dir "$OUT_DIR" > logs/relax_diag.log 2>&1
