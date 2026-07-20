"""
Molecular dynamics for liquid Cu starting from og_config.xyz (a pre-equilibrated liquid-like
configuration, unlike md_solid.py which builds a fresh FCC cell in code).
Runs NPT ensemble; results saved under RESULTS/LIQUID/.
"""

import argparse
import os
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md import MDLogger
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from deepmd_jax.md import DPJaxCalculator

parser = argparse.ArgumentParser(description="Liquid Cu NPT MD simulation")
parser.add_argument("--temperature", type=int, required=True, help="Temperature in K")
args = parser.parse_args()
T = args.temperature

time_step_fs = 1.0
total_time_ns = 1.0
n_steps = int((total_time_ns * 1e6) / time_step_fs)

base_dir = os.path.dirname(os.path.abspath(__file__))
results_log = os.path.join(base_dir, "RESULTS", "LIQUID", "logs")
results_xyz = os.path.join(base_dir, "RESULTS", "LIQUID", "xyz")
os.makedirs(results_log, exist_ok=True)
os.makedirs(results_xyz, exist_ok=True)

# hyb3 by default (final generation, fine-tuned on density) -- any released model works;
# point at your own model.pkl to use a different one.
model_path   = os.path.join(base_dir, "..", "..", "models", "hyb3", "model", "model_hyb3.pkl")
og_config    = os.path.join(base_dir, "og_config.xyz")
xyz_out      = os.path.join(results_xyz, f"liquid_{T}K.xyz")
log_out      = os.path.join(results_log, f"liquid_{T}K.log")
final_out    = os.path.join(results_log, f"liquid_{T}K_final.lammps-data")

atoms = read(og_config, index=0)

symbols  = atoms.get_chemical_symbols()
type_map = {0: "Cu"}
type_idx = np.array([{v: k for k, v in type_map.items()}[s] for s in symbols])

atoms = Atoms(symbols=symbols, positions=atoms.get_positions(), cell=atoms.get_cell(), pbc=True)
atoms.calc = DPJaxCalculator(model_path=model_path, type_idx=type_idx)

MaxwellBoltzmannDistribution(atoms, temperature_K=T)

timestep = time_step_fs * units.fs
# Looser thermostat/barostat coupling than md_solid.py (10fs/100fs) -- deliberately gentler
# for the liquid phase.
ttime    = 100  * units.fs
ptime    = 1000 * units.fs
pfactor  = (ptime ** 2) * 10.0 * units.GPa

def write_frame():
    dyn.atoms.write(xyz_out, append=True)

def remove_com():
    Stationary(atoms)

print(f"Starting liquid MD at {T} K")
dyn = NPT(
    atoms,
    timestep,
    temperature_K=T,
    externalstress=0.0,
    ttime=ttime,
    pfactor=pfactor,
    mask=np.eye(3, dtype=int),
)
dyn.set_fraction_traceless(0)

dyn.attach(write_frame, interval=1000)
dyn.attach(MDLogger(dyn, atoms, log_out, stress=True, peratom=False, mode="a"), interval=100)
dyn.attach(remove_com, interval=1000)

dyn.run(n_steps)
print("Liquid MD done.")

write(final_out, atoms)
