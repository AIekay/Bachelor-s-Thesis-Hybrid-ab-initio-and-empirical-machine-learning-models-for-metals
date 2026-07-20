# Models

```text
models/
├── abinit/{model/model_abinit.pkl, logs/log_abinit.txt}
├── hyb1/{model/model_hyb1.pkl,     logs/log_hyb1.txt}
├── hyb2/{model/model_hyb2.pkl,     logs/log_hyb2.txt}
└── hyb3/{model/model_hyb3.pkl,     logs/log_hyb3.txt}
```

| Model | Trained on | Steps | Wall-clock |
|---|---|---|---|
| `abinit` | DFT energies + forces | 10,000,000 | ~4h52m |
| `hyb1` | `abinit` + density loss (MD from `abinit`) | 5,000,000 | ~18h20m |
| `hyb2` | `hyb1` + density loss (MD from `hyb1`) | 5,000,000 | ~18h25m |
| `hyb3` | `hyb2` + density loss (MD from `hyb2`) | 5,000,000 | ~18h27m |

Scripts: [`../training_scripts/`](../training_scripts/README.md). MD→training-data lineage:
[`../molecular_dynamics/trajectory_to_dataset/`](../molecular_dynamics/trajectory_to_dataset/README.md).

## Architecture (shared by all 4)

`rcut=6.0` Å, embed widths `[32,32,64]`, fit widths `[128,128,128]`, axis `12`, 135,137 params, compressed.

## Loading a model

```python
from deepmd_jax.md import DPJaxCalculator
calc = DPJaxCalculator(model_path="models/hyb3/model/model_hyb3.pkl", type_idx=type_idx)
atoms.calc = calc
```

## Log format

Real, unmodified logs (only `<DATA_ROOT>` path substituted). One line per iteration:

```text
Iter <n> L <total_loss> LE <energy_loss> LF <force_loss> [LOBS<i> <val> ESS<i> <val> OBS_REW_<i>_0 <val> OBS_<i>_0 <val> ...] Time <seconds>s
```

`LOBS<i>`/`ESS<i>`/`OBS_REW_<i>_0`/`OBS_<i>_0` (hybrid models only) are per-temperature-index
observable loss/effective sample size/reweighted mean/unweighted mean — see
[`../training_scripts/hybrid/README.md`](../training_scripts/hybrid/README.md) for the temperature mapping.
