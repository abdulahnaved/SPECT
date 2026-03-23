# SPECT Collimator Transport Simulation

Monte Carlo simulation of gamma photon transport through a parallel-hole lead collimator, built with [OpenGATE](https://opengate-python.readthedocs.io/) (Geant4 Python wrapper).

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (once)
git clone https://github.com/abdulahnaved/SPECT.git
cd SPECT
uv sync                                             # creates venv + installs everything
```

## Run Simulation

```bash
uv run python -m collimator_transport.run --total 10000000 --batches 10 --workers 4
```

This shoots 10M photons split across 10 batches (4 running in parallel). Each batch writes ROOT files to `output/batch_XXXX/phsp/`.

## Post-process

```bash
uv run python postprocess.py        # merge batches → postprocessed_data.npy
uv run python inspect_data.py       # print summary stats
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--total` | Total primary photons | 1,000,000 |
| `--batches` | Number of batches | 4 |
| `--workers` | Parallel processes | 4 |
| `--base-seed` | Base random seed (batch *i* gets seed + *i*) | 42 |
| `--output-dir` | Output directory | `output/` |
