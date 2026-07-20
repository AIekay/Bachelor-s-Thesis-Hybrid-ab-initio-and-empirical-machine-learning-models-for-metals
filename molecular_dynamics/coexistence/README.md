# Solid-liquid coexistence (interface-pinning) melting-point MD

Solid-liquid interface-pinning method, PLUMED-biased (via ASE's `Plumed` calculator) DeePMD-JAX
MD. `run-ase.py` runs biased NPT MD per temperature (1200/1300/1400K default) from
`cu_interface.lammps-data` (1728-atom two-phase config); `calc_melting.py` fits `Δμ(T)` from the
resulting `COLVAR` files and solves for `Δμ=0`.

Needs ≥2 (in practice 3) temperatures to fit a line — doesn't reduce to one example the way
`../solid/`/`../liquid/` do.

## Files

| File | Role |
|---|---|
| `run-ase.py` | MD driver (ASE + PLUMED + DeePMD-JAX) |
| `plumed.dat` | PLUMED input (CV + restraint + output) |
| `EnvironmentSimilarity.cpp` | Third-party PLUMED CV source — see below |
| `cu_interface.lammps-data` | Initial two-phase config, 1728 atoms |
| `calc_melting.py` | Δμ(T) fit → melting point + plot + report |
| `job_melting.sh` | SLURM job, one temperature (reconstructed, see file) |
| `job_calc_melting.sh` | SLURM job, post-processing |
| `submit_melting.sh` | Orchestrates the above |

`plumed.dat`'s `SPECIES=1-1728` is specific to the shipped interface config — update for a
different system size.

## Third-party code: `EnvironmentSimilarity.cpp`

PLUMED's own `ENVIRONMENTSIMILARITY` CV, copied verbatim. Copyright (c) 2020-2023 The PLUMED team
(Pablo Piaggi, Princeton), LGPLv3 — see [`../../NOTICE.md`](../../NOTICE.md). JIT-compiled by
PLUMED at runtime; no `.o`/`.so` shipped.

## Requirements

PLUMED (system dependency, see root [`environment.yml`](../../environment.yml)) + `deepmd-jax` + ASE.

## How to run

```bash
bash submit_melting.sh   # fill in placeholders in the job scripts first

# or locally:
python run-ase.py 1200
python run-ase.py 1300
python run-ase.py 1400
python calc_melting.py --temperatures 1200 1300 1400
```
