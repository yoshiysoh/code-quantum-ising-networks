# Nonequilibrium quantum states emerging from flippable spins in random networks

[![arXiv](https://img.shields.io/badge/arXiv-2505.11466-b31b1b.svg)](https://arxiv.org/abs/2505.11466)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23414366.svg)](https://doi.org/10.5281/zenodo.23414366)
[![CI](https://github.com/yoshiysoh/code-quantum-ising-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/yoshiysoh/code-quantum-ising-networks/actions/workflows/ci.yml)

Code and data for

> Yoshiaki Horiike and Yuki Kawaguchi,
> *Nonequilibrium quantum states emerging from flippable spins in random networks*,
> [arXiv:2505.11466](https://arxiv.org/abs/2505.11466) (2026).

## Installation

```bash
mamba env create -f env.yml
mamba activate paper-quantum-ising-networks
pip install -e . --no-deps
```

`env.yml` pins the package versions the figures were made with. `pytest`
checks that every module and script imports, that each file in `data/`
matches its SHA-256 hash in `data/SHA256SUMS`, and that each data script
runs at the smallest size (about a minute in all). GitHub Actions runs
`pytest`, `ruff check` and `ruff format --check` (ruff 0.17.1, 79
columns) on every push.

## Figures

Run a notebook to redraw its figure from the data in `data/`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/fig-02.ipynb
```

Each notebook writes its figure, without panel letters, to
`figures/<fig-id>/` (PDF, SVG, PNG); `<name>.labels.json` gives where each
letter goes.

| Figure | Notebook |
|---|---|
| Fig. 1 | `fig-01` |
| Fig. 2 | `fig-02` |
| Fig. 3 | `fig-03a` (panel a), `fig-03` |
| Fig. 4 | `fig-04` |
| Extended Data Figs. 1–8 | `fig-s01` … `fig-s08` |
| Supplementary Movies 1, 2 | `python3 scripts/render_movie.py --fig fig-03` (`fig-s03`) `--start 0 --end 1800 --out figures/movie-01.mp4` (`movie-02.mp4`) |
| Supplementary Movie 3 | `mov-03` |

## Data

`data/` holds every file the notebooks read. Each was made by the command
below, run from the repository root (*parallel* scripts use all cores, or
`NUM_WORKERS`; the large system sizes take hours to days on a cluster node).
A script writes into `data/<date of the run>/` (or `--out-dir`), so a new
run never overwrites the published files; the notebooks read `data/`, so
copy a new file there to use it.

Run them in this order: the class rule reads `revival_null.h5`, and
`spectral_extrapolation.h5` reduces the class and spectral data.

The multi-hour products have wrappers in `run/` that issue the commands
below with the published arguments and seeds: `classes.sh`, `spectral.sh`
(with the extrapolation), `hamapprox.sh`, `relax_diag.sh`, and
`select_examples.sh` (the scans that pick the examples). Start one with,
for example, `nohup bash run/classes.sh &`. Each runs `WORKERS` processes
(default: the physical cores) with one BLAS thread each, as for the
published files, writes to `OUT_DIR` (default `data/<date>/`), logs to
`logs/`, and skips outputs that already exist.

Every file a script writes has a `provenance` group whose attributes record
the code's git commit, the command line, the Python and package versions,
the CPU, the BLAS library and its threads, and the UTC time;
`h5io.load_arrays` skips it. The files in `data/` were made before this was
added and record their parameters and seeds only.

| File | Command |
|---|---|
| `revival_null.h5` | `python3 scripts/build_revival_null.py --N 13 --model ferro --nets 12 --reps 200` |
| `revival_null_early_dip.h5` | the same with `--rule early-dip` (the earlier revival rule, which Extended Data Fig. 8 compares with) |
| `classes_N*.h5` (*parallel*) | `python3 scripts/generate_class_dataset.py --N 8 9 10 11 12 13 --nets 200`, and `--N 14 15 --nets 100` |
| `spectral_data.h5` (*parallel*) | `python3 scripts/generate_spectral_data.py` |
| `spectral_data_extra.h5` (*parallel*) | `python3 scripts/generate_spectral_data.py --extra 200` |
| `spectral_extrapolation.h5` | `python3 scripts/extrapolate_spectral_data.py` |
| `examples_N12_ferro_phi1.00_M0.h5` | `python3 scripts/generate_examples.py --N 12 --phi 1.0 --model ferro --fixed-net 641 --magnetization 0 --recurrence-spins 2 --scar-pxp` |
| `examples_N12_pmJ_phi1.00_M0.h5` | `python3 scripts/generate_examples.py --N 12 --phi 1.0 --model pmJ --fixed-net 24 --magnetization 0 --recurrence-spins 2` |
| `ring_N12_ferro_h2.0.h5` | `python3 scripts/generate_ring_example.py --N 12 --h 2.0` |
| `ring_N12_pmJ_h2.0_seed0.h5` | `python3 scripts/generate_ring_example.py --N 12 --h 2.0 --model pmJ --seed 0` |
| `relax_diag_N12_*.h5` (*parallel*) | `python3 scripts/diagnose_relax_time.py --nets 32` |
| `hamapprox_N12_*.h5` (*parallel*) | `python3 scripts/verify_hamiltonian_approximation.py --N 12 --model <model> --gamma <Gamma> --nsamples 10000 --seed0 <seed0> --skip-self-test`, with the seeds below |

`--seed0` of the `hamapprox` files:

| Gamma | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| ferro | 40000 | 41000 | 42000 | 43000 | 44000 | 45000 | 46000 | 47000 | 48000 |
| pmJ | 50000 | 51000 | 52000 | 53000 | 41500 | 42500 | 43500 | 44500 | 45500 |

The examples are network 641 (ferromagnetic) and network 24 (±J), chosen
from the scans
`python3 scripts/select_example.py --rank pxp --model ferro --start 0 --stop 720 --magnetization 0 --recurrence-spins 2`
and
`python3 scripts/select_example.py --rank purity --model pmJ --start 0 --stop 60 --magnetization 0 --recurrence-spins 2`:
among the networks with a plateau-scar state, the scar with no minor
revival peaks that spends the largest fraction of the window below
F = 0.01, and the scar of highest spectral purity.

Measured run times: each `examples_*` and `ring_*` file takes about a
minute on one core of an AMD Ryzen 9 7950X; the ferromagnetic selection
scan took about 10 minutes on 165 cores of Intel Xeon nodes and the ±J
scan 2 minutes on 30 cores; Supplementary Movies 1 and 2 take about
11 minutes each on one Ryzen core. The class, spectral, `hamapprox` and
`relax_diag` files take hours to days on 16–40-core cluster nodes; the
null files were not timed. Each notebook runs in under 2 minutes on an
Apple M4 laptop (`mov-03` renders Supplementary Movie 3 in about 1.5
minutes).

Every random draw is seeded, and choices between equal candidates follow
fixed rules: equally large sectors go to the lower classical energy, equal
scores are compared at 9 decimals with ties to the lower Fock index, and
eigenvectors of degenerate eigenvalues, whose basis depends on the LAPACK
build, are not used. We checked the published files with conda-forge
OpenBLAS 0.3.34 and one BLAS thread per process, on Linux x86-64 (AMD
Zen 4 and Intel Xeon) and macOS arm64. With the pinned versions the
scripts reproduce `data/` to rounding (10⁻¹⁰ or less), apart from one
quantity that affects no figure or quoted number: in a few states of
`classes_N*.h5`, `n_eff` and `n_inv1` differ in the last digits of their
float32 storage or by one spin at the threshold.

## License

Code: BSD 3-Clause (`LICENSE`). Data, figures and movies: CC BY 4.0
(`LICENSE-data`). If you use them, please cite the paper above
(`CITATION.cff`).
