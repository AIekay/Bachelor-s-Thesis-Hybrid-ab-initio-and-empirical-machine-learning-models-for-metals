# Ab initio training script

DFT energy+force training only, no observable loss, no hybrid mode.

| Parameter | Value |
|---|---|
| `rcut` | 6.0 Å |
| `l_pref_e` | 1 |
| `s_pref_e` | 1 |

Released model: 10,000,000 steps, ~4h52m on 1 GPU (`models/abinit/model/model_abinit.pkl`).

## Files

| File | Description |
|---|---|
| `train_abinit.py` | Training script |
| `test.py` | RMSE evaluation vs. ab initio data |

## Data

`<DATA_ROOT>/Cu_DATA/pbc/<System>/` in DeePMD-JAX raw/set format — see
[`../../datasets/ab_initio/README.md`](../../datasets/ab_initio/README.md).

## How to run

```bash
python train_abinit.py --steps=500   # local smoke test
python test.py

python train_abinit.py --steps=10000000   # full run
```

`test.py` expects the model at `<DATA_ROOT>/model_abinit.pkl` (where `train_abinit.py` saves it —
not the current directory, unlike `training_scripts/hybrid/test.py`).
