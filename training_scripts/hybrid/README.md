# Hybrid training script

Ab initio (energy+force) loss + experimental density observable loss, on top of a previous
generation's weights. Representative of `hyb1`/`hyb2`/`hyb3` (identical prefactors, only data
paths differ — each generation's experimental data comes from MD with the *previous*
generation's weights, see
[`../../molecular_dynamics/trajectory_to_dataset/`](../../molecular_dynamics/trajectory_to_dataset/README.md)).

## Prefactors

| Parameter | Value |
|---|---|
| `l_pref_e` | 0.1 |
| `s_pref_e` | 0.1 |
| `obs_l_pref` | 100 |
| `obs_s_pref` | 1.0 |
| `obs_step_every` | 10 |

Effective obs weight/step ≈ `obs_l_pref / obs_step_every` = 10. Released models: 5,000,000 steps
each, ~18.3–18.5h on 1 GPU.

## Experimental data

`<DATA_ROOT>/DATA_EXP/DATA_EXP_<T>K` (800–2000K, 1300K excluded — the source model's melting
point was below 1300K, so that trajectory equilibrated liquid instead of solid).

## Files

| File | Description |
|---|---|
| `train_hybrid.py` | Training script |
| `test.py` | RMSE evaluation vs. ab initio data only |
| `plot_train_evo.py` | Loss/RMSE/observable plots |
| `job_mn.sh` | Reference SLURM job showing the real train→plot→test invocation sequence |

`job_mn.sh` is kept here as a reference example only — job scripts are otherwise excluded from
`training_scripts/` (see [`../README.md`](../README.md)).

No `log.txt`/`.pkl` shipped here — see the real ones at [`../../models/hyb3/`](../../models/README.md).

## How to run

```bash
python train_hybrid.py --steps=500   # local smoke test
python test.py
python plot_train_evo.py

python train_hybrid.py --steps=5000000   # full run
# or, on a SLURM cluster, after filling in placeholders:
sbatch job_mn.sh
```

Fill in `<DATA_ROOT>` first. Unzip the matching
[`../../datasets/experimental/DATA_EXP/`](../../datasets/experimental/README.md) archive there
(e.g. `DATA_EXP_HYB1.zip` to train `hyb2`), or produce your own via
[`../../molecular_dynamics/trajectory_to_dataset/`](../../molecular_dynamics/trajectory_to_dataset/README.md).
