# Datasets

```text
datasets/
├── ab_initio/       DFT dataset (see below for how to get it)
├── experimental/     Literature density data + calculation script
└── descriptors/      DeepPot-SE descriptor + t-SNE analysis pipeline
```

## Data availability

- **Ab initio master dataset** (`Cu_FHIaims-PBE-dataset.json`): not shipped — download from
  Zenodo, see [`ab_initio/README.md`](ab_initio/README.md).
- **Derived DeePMD-JAX training data**: `ab_initio/Cu_DATA.zip` (ab initio) and
  `experimental/DATA_EXP/*.zip` (per-generation observable data) are both included — see their
  respective READMEs.

## Dataset format (DeePMD-JAX)

```text
<System>/
├── type.raw          One integer type index per atom
├── type_map.raw       Element symbol per line, indexed by type.raw
└── set.NNN/
    ├── box.npy         (n_frames, 9)
    ├── coord.npy       (n_frames, 3*n_atoms)
    ├── energy.npy      (n_frames,)
    └── force.npy       (n_frames, 3*n_atoms)
```
