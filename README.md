# Nonequilibrium quantum states emerging from flippable spins in random networks

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
checks that every module and script imports and that each file in `data/`
matches its SHA-256 hash in `data/SHA256SUMS`.

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

| File | Command |
|---|---|
| `classes_N*.h5` (*parallel*) | `python3 scripts/generate_class_dataset.py --N 8 9 10 11 12 13 --nets 200`, and `--N 14 15 --nets 100` |
| `spectral_data.npz` (*parallel*) | `python3 scripts/generate_spectral_data.py` |
| `spectral_data_extra.npz` (*parallel*) | `python3 scripts/generate_spectral_data.py --extra 200` |
| `spectral_extrapolation.npz` | `python3 scripts/extrapolate_spectral_data.py` |
| `revival_null.npz` | `python3 scripts/build_revival_null.py --N 13 --model ferro --nets 12 --reps 200` |
| `examples_N12_ferro_phi1.00_M0.h5` | `python3 scripts/generate_examples.py --N 12 --phi 1.0 --model ferro --fixed-net 641 --magnetization 0 --recurrence-spins 2 --scar-pxp` |
| `examples_N12_pmJ_phi1.00_M0.h5` | `python3 scripts/generate_examples.py --N 12 --phi 1.0 --model pmJ --fixed-net 41 --magnetization 0 --recurrence-spins 2` |
| `ring_N12_ferro_h2.0.h5` | `python3 scripts/generate_ring_example.py --N 12 --h 2.0` |
| `ring_N12_pmJ_h2.0_seed0.h5` | `python3 scripts/generate_ring_example.py --N 12 --h 2.0 --model pmJ --seed 0` |
| `relax_diag_N12_*.npz` (*parallel*) | `python3 scripts/diagnose_relax_time.py --nets 32` |
| `hamapprox_N12_*.h5` (*parallel*) | `python3 scripts/verify_hamiltonian_approximation.py --N 12 --model <ferro or pmJ> --gamma <Gamma> --nsamples 10000 --seed0 <seed> --skip-self-test`, one run per model and Gamma; the seed is stored in each file |

`revival_null_early_dip.npz` is the null with the revival measured after the
deepest dip in Jt ≤ 50, which Extended Data Fig. 8 compares with.
The example networks 641 and 41 were chosen with
`scripts/select_example.py --rank pxp` and `--rank purity`.

Every random draw is seeded (the seeds are stored in the files), so with
the pinned versions the scripts reproduce `data/`, except that a few choices
are decided by ties: between equally large sectors (numpy's
sort order differs between CPUs) and between symmetry-related states
(scores equal to rounding). Regenerated files can therefore differ from
those in `data/` in a few realisations, by an equivalent choice.

## License

Code: BSD 3-Clause (`LICENSE`). Data, figures and movies: CC BY 4.0
(`LICENSE-data`). If you use them, please cite the paper above
(`CITATION.cff`).
