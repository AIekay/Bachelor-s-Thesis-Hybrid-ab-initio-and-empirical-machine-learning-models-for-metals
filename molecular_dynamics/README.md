# Molecular dynamics

Jobs included here (unlike `training_scripts/`).

```text
molecular_dynamics/
├── solid/                  Bulk solid Cu NPT MD (example: 800K)
├── liquid/                 Bulk liquid Cu NPT MD (example: 1800K)
├── trajectory_to_dataset/   MD trajectory → DeePMD-kit training data
└── coexistence/             PLUMED interface-pinning melting-point MD (1200/1300/1400K)
```

## Shared conventions

- `solid/md_solid.py`, `liquid/md_liquid.py`, `coexistence/run-ase.py` default `model_path` to
  `../../models/hyb3/model/model_hyb3.pkl` — edit to use a different model.
- Placeholders: `<YOUR_SLURM_ACCOUNT>` / `<YOUR_SLURM_QOS>` / `<PATH_TO_YOUR_CONDA_ENV>` — see
  root [`README.md`](../README.md).

PLUMED (system dependency, see root [`environment.yml`](../environment.yml)) is needed only for `coexistence/`.
