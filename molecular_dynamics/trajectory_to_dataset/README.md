# Trajectory → training dataset

Converts a raw MD trajectory (`.xyz`, from [`../solid/`](../solid/README.md) or
[`../liquid/`](../liquid/README.md)) into DeePMD-kit raw/set format + a density `observable.raw`,
at `<DATA_ROOT>/DATA_EXP/DATA_EXP_<T>K/` (expected by
[`../../training_scripts/hybrid/train_hybrid.py`](../../training_scripts/hybrid/train_hybrid.py)).

## How to run

```bash
mkdir DATA_EXP
cp md_800K.xyz md_900K.xyz ... DATA_EXP/       # your md_<T>K.xyz files
cp xyz_to_raw_obs.py raw_to_set.sh master.sh DATA_EXP/
cd DATA_EXP
bash master.sh
```

`master.sh` loops over `md_*.xyz`, creating one `DATA_EXP_<T>K/` per file (runs
`xyz_to_raw_obs.py` then `raw_to_set.sh` in each).

Manual (one temperature):

```bash
mkdir DATA_EXP_800K && cd DATA_EXP_800K
cp ../md_800K.xyz ../xyz_to_raw_obs.py ../raw_to_set.sh .
python xyz_to_raw_obs.py --input_traj md_800K.xyz
bash raw_to_set.sh
```

`xyz_to_raw_obs.py` writes `coord/energy/force/box.raw` (real values), `type.raw`+`type_map.raw`,
and `observable.raw` (density) — the filename `observable.raw` is what marks a dataset as
experimental/observable to the training scripts. `raw_to_set.sh` is the standard DeePMD-kit
raw→set splitter.
