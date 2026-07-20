# Training scripts

Job/SLURM scripts are generally **not** included here (unlike `molecular_dynamics/`) — step
counts below are the surviving record of them, except one reference example kept in `hybrid/`.

```text
training_scripts/
├── abinit/
│   ├── train_abinit.py       Ab initio (DFT energy + force) training
│   └── test.py               RMSE evaluation vs. ab initio data
└── hybrid/
    ├── train_hybrid.py       Hybrid (ab initio + experimental density) training
    ├── test.py               RMSE evaluation vs. ab initio data
    ├── plot_train_evo.py     Loss/RMSE/observable plots
    └── job_mn.sh             Reference SLURM job (train -> plot -> test)
```

`hybrid/` is representative of **all three** hybrid generations (hyb1/hyb2/hyb3) — identical
prefactors, only data paths differ. See [`hybrid/README.md`](hybrid/README.md).

Both `test.py` scripts test against ab initio data only — testing a hybrid model against its own
training-time observable dataset would only measure self-consistency, not accuracy.

## Step counts

| Script | Model(s) | Steps | Wall-clock |
|---|---|---|---|
| `abinit/train_abinit.py` | `abinit` | 10,000,000 | ~4h52m |
| `hybrid/train_hybrid.py` | `hyb1`/`hyb2`/`hyb3` | 5,000,000 | ~18.3–18.5h |

Both scripts use `<DATA_ROOT>` — see root [`README.md`](../README.md).
