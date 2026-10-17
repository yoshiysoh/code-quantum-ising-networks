#!/usr/bin/env bash
# Check of the approximated Hamiltonian, data/hamapprox_N12_*.h5
# (18 files, hours each).
#
#   nohup bash run/hamapprox.sh &
#
# Runs from the repository root. WORKERS processes (default: the physical
# cores), each with one BLAS thread, as for the published files. Writes to
# OUT_DIR (default data/<today>/) and skips files that already exist there.
# The seeds are those of the published files (README).
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

gammas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seeds_ferro=(40000 41000 42000 43000 44000 45000 46000 47000 48000)
seeds_pmJ=(50000 51000 52000 53000 41500 42500 43500 44500 45500)

for model in ferro pmJ; do
    for i in "${!gammas[@]}"; do
        g="${gammas[$i]}"
        if [[ "$model" == ferro ]]; then seed="${seeds_ferro[$i]}"
        else seed="${seeds_pmJ[$i]}"; fi
        out="$OUT_DIR/hamapprox_N12_${model}_gamma$(printf '%.2f' "$g").h5"
        if [[ -e "$out" ]]; then echo "skip $out"; continue; fi
        python3 scripts/verify_hamiltonian_approximation.py --N 12 \
            --model "$model" --gamma "$g" --nsamples 10000 \
            --seed0 "$seed" --skip-self-test --workers "$WORKERS" \
            --out-dir "$OUT_DIR" > "logs/hamapprox_${model}_${g}.log" 2>&1
    done
done
