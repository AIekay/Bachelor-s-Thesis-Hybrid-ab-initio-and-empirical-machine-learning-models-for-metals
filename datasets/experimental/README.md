# Experimental density data

`Cu_densities.py` — self-contained, computes the target densities used as `obs_target` in
[`../../training_scripts/hybrid/train_hybrid.py`](../../training_scripts/hybrid/train_hybrid.py):
solid density via integrated thermal-expansion coefficient (below 1357K), liquid density via
linear fit to the reference below (above 1357K). Saves `Cu_densities_plot.png`.

## Liquid density reference

```bibtex
@article{Cu_density_liquid_brillo2003,
  author    = {Brillo, J. and Egry, I.},
  title     = {Density Determination of Liquid Copper, Nickel, and Their Alloys},
  journal   = {International Journal of Thermophysics},
  volume    = {24},
  number    = {4},
  pages     = {1155--1170},
  year      = {2003},
  publisher = {Plenum Publishing Corporation},
  doi       = {10.1023/A:1025021521945}
}
```

## How to run

```bash
python Cu_densities.py
```

## Per-generation training data (`DATA_EXP/`)

The experimental/observable training data (`obs_train_data_path` in
[`../../training_scripts/hybrid/train_hybrid.py`](../../training_scripts/hybrid/train_hybrid.py)),
generated from each generation's own MD via
[`../../molecular_dynamics/trajectory_to_dataset/`](../../molecular_dynamics/trajectory_to_dataset/README.md):

| Zip | Trains | Source MD |
|---|---|---|
| `DATA_EXP/DATA_EXP_ABINIT.zip` | `hyb1` | `abinit` |
| `DATA_EXP/DATA_EXP_HYB1.zip` | `hyb2` | `hyb1` |
| `DATA_EXP/DATA_EXP_HYB2.zip` | `hyb3` | `hyb2` |

Each unzips to a `DATA_EXP/` folder of `DATA_EXP_<T>K/` subfolders (DeePMD-kit `type.raw` +
`set.000/*.npy`). Unzip the one matching the generation you want to train and point
`<DATA_ROOT>/DATA_EXP` at it.

Trimmed from the original per-run data: raw MD trajectories (`.xyz`) and intermediate `.raw` text
files were dropped, keeping only the DeePMD-consumable format.
