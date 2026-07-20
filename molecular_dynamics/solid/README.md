# Solid-phase MD

`md_solid.py` — 3×3×3 FCC Cu supercell (108 atoms), 1ns NPT at a given temperature, DeePMD-JAX
model as ASE calculator. Writes `RESULTS/SOLID/xyz/solid_<T>K.xyz`,
`RESULTS/SOLID/logs/solid_<T>K.log`, and a final `.lammps-data` snapshot.

`model_path` defaults to the released `hyb3` model — edit to use a different one.

## How to run

```bash
python md_solid.py --temperature 800

# On a SLURM cluster (fill in placeholders first):
sbatch job_solid.sh 800
```
