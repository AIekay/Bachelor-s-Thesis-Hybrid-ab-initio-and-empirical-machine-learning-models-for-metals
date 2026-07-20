# Liquid-phase MD

`md_liquid.py` starts from `og_config.xyz` (a pre-equilibrated liquid-like configuration),
unlike `../solid/md_solid.py` which builds a fresh FCC cell in code. Runs 1ns of NPT at a given
temperature, with looser thermostat/barostat coupling than the solid script (`ttime=100fs`,
`ptime=1000fs` vs. `10fs`/`100fs`). Writes to `RESULTS/LIQUID/`.

`model_path` defaults to the released `hyb3` model — edit to use a different one.

## How to run

```bash
python md_liquid.py --temperature 1800

# On a SLURM cluster (fill in placeholders first):
sbatch job_liquid.sh 1800
```
