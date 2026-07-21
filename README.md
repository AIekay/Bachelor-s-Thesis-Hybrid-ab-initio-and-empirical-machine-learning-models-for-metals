# Cu DeePMD-JAX: ab initio + experimentally-informed interatomic potentials

Code skeleton for the bachelor thesis **"Hybrid ab initio and empirical machine learning models
for the potential energy surface of simple metals"** — machine-learned interatomic potentials for
copper trained with [DeePMD-JAX](https://github.com/deepmodeling/deepmd-jax). Real model
checkpoints and training logs, plus genericized scripts (cluster paths/usernames replaced with
placeholders).

## Structure

- [`datasets/`](datasets/README.md) — ab initio dataset, experimental density data, descriptor/t-SNE analysis.
- [`models/`](models/README.md) — 4 released checkpoints + training logs.
- [`training_scripts/`](training_scripts/README.md) — ab initio + hybrid training/test scripts.
- [`molecular_dynamics/`](molecular_dynamics/README.md) — solid/liquid MD, trajectory→dataset conversion, coexistence melting-point MD.

## Getting started

```bash
conda env create -f environment.yml
conda activate cu-deepmd-jax
```

PLUMED (needed only for `molecular_dynamics/coexistence/`) is a separate system dependency — see that folder's README.

### Placeholders

| Placeholder | Meaning |
|---|---|
| `<DATA_ROOT>` | Root directory containing your datasets |
| `<YOUR_SLURM_ACCOUNT>` | SLURM `--account` |
| `<YOUR_SLURM_QOS>` | SLURM `--qos` |
| `<PATH_TO_YOUR_CONDA_ENV>` | Path to your `deepmd-jax` conda env |

## Citation

```
Alexander Jon Barrena Garay. "Hybrid ab initio and empirical machine learning models for the potential energy
surface of simple metals." Bachelor's Thesis, EHU, 2026.
```

## License

MIT (see [`LICENSE`](LICENSE)). `molecular_dynamics/coexistence/EnvironmentSimilarity.cpp` is
third-party PLUMED code under a separate license — see [`NOTICE.md`](NOTICE.md).
