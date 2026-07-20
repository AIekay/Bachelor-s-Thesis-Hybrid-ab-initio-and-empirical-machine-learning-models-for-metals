# Descriptor / t-SNE dataset analysis

Per-atom DeepPot-SE descriptors from the ab initio dataset, reduced to 2D via t-SNE.

```text
descriptors/
├── 01_compute_descriptors.py   JAX model evaluation over sampled frames
├── 02_reduce_and_save.py       Standardize + t-SNE (independent of step 1)
├── 03_plot_check.py            Sanity-check scatter plots
└── assets/one_config.png
```

`01_compute_descriptors.py`: samples frames per system under `../ab_initio/Cu_DATA/pbc/`,
evaluates descriptors against `../../models/abinit/model/model_abinit.pkl`, saves
`runs/<RUN_NAME>/descriptors_raw.npz`. `batch_size` is hardcoded to `1` (confirmed
`evaluate_descriptors` bug at `batch_size > 1`).

`02_reduce_and_save.py`: t-SNE (`perplexity=30`, `init='pca'`, `random_state=42`) →
`descriptors_reduced.npy` + `INFO.txt`.

`03_plot_check.py`: loads only `descriptors_reduced.npy`.

## Requirements

Step 1 needs `deepmd_jax` (JAX) + the ab initio dataset + `abinit` model. Steps 2–3 need only
`numpy`/`scikit-learn`/`matplotlib`.

## How to run

```bash
python 01_compute_descriptors.py
python 02_reduce_and_save.py
python 03_plot_check.py
```
