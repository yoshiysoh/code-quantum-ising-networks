#!/usr/bin/env bash
# Scans that pick the example networks of Fig. 3 (ferromagnetic, networks
# 0-719) and Extended Data Fig. 3 (+-J, networks 0-59); hours on one core.
#
#   nohup bash run/select_examples.sh &
#
# Runs from the repository root. The networks are split into blocks run as
# WORKERS parallel processes (default: the physical cores), each with one
# BLAS thread. Each block logs its candidates to logs/select_<rank>_<a>.log;
# the README states how the examples are read off. A finished block ends its
# log with "# done" and is skipped on a rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

physical_cores() {
    if [[ "$(uname)" == Darwin ]]; then sysctl -n hw.physicalcpu
    else lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l; fi
}
WORKERS="${WORKERS:-$(physical_cores)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs

scan() {  # rank model stop block
    local rank=$1 model=$2 stop=$3 block=$4 a
    for (( a = 0; a < stop; a += block )); do
        local log="logs/select_${rank}_${a}.log"
        if [[ -e "$log" ]] && tail -n 1 "$log" | grep -qx "# done"; then
            continue
        fi
        echo "$rank $model $a $(( a + block < stop ? a + block : stop )) $log"
    done
}
# shellcheck disable=SC2016  # $0..$4 expand in the inner bash
{ scan pxp ferro 720 4; scan purity pmJ 60 2; } |
    xargs -P "$WORKERS" -L 1 bash -c \
        'python3 scripts/select_example.py --rank "$0" --model "$1" \
             --start "$2" --stop "$3" --magnetization 0 \
             --recurrence-spins 2 > "$4" 2>&1 && echo "# done" >> "$4"'
