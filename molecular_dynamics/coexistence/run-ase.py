import numpy as np
import os
import argparse
from deepmd_jax.md import DPJaxCalculator
from ase import Atoms
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from ase.io import read, write
from ase.md import MDLogger
from ase.calculators.plumed import Plumed

parser = argparse.ArgumentParser(description="Melting-point interface MD with Plumed")
parser.add_argument("temperature", type=float, help="Simulation temperature in K")
args = parser.parse_args()

T_init = args.temperature
temp_str = f"{T_init:g}"

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir  = os.path.join(base_dir, "RESULTS", "MELTING", temp_str)
os.makedirs(out_dir, exist_ok=True)

atoms = read(os.path.join(base_dir, "cu_interface.lammps-data"))

symbols  = atoms.get_chemical_symbols()
type_map = {"Cu": 0}
type_idx = np.array([type_map[s] for s in symbols])

# hyb3 by default (final generation, fine-tuned on the melting-point observable) -- any
# released model works; point at your own model.pkl to use a different one.
model_path = os.path.join(base_dir, "..", "..", "models", "hyb3", "model", "model_hyb3.pkl")
calc_dp    = DPJaxCalculator(model_path=model_path, type_idx=type_idx)

with open(os.path.join(base_dir, "plumed.dat")) as f:
    setup = []
    for line in f:
        stripped = line.strip()
        if "FILE=COLVAR2" in stripped:
            stripped = stripped.replace("FILE=COLVAR2", f"FILE={out_dir}/COLVAR2_{temp_str}")
        elif "FILE=COLVAR" in stripped:
            stripped = stripped.replace("FILE=COLVAR", f"FILE={out_dir}/COLVAR_{temp_str}")
        setup.append(stripped)

kB           = 0.00008617333262
timestep     = 1.0 * units.fs
ttime        = 100 * units.fs
ptime        = 1000 * units.fs
bulk_modulus = 140.0  # Cu ~140 GPa
pfactor      = (ptime ** 2) * bulk_modulus * units.GPa

MaxwellBoltzmannDistribution(atoms, temperature_K=T_init)

atoms.calc = Plumed(calc=calc_dp, input=setup, timestep=timestep, atoms=atoms, kT=kB * T_init)

xyz_file        = os.path.join(out_dir, f"md_{temp_str}.xyz")
log_file        = os.path.join(out_dir, f"md_{temp_str}.log")
final_data_file = os.path.join(out_dir, f"cu_final_{temp_str}.lammps-data")

def write_frame():
    dyn.atoms.write(xyz_file, append=True)

def remove_com():
    Stationary(atoms)

print(f"Starting melting MD at {T_init} K")

dyn = NPT(
    atoms,
    timestep,
    temperature_K=T_init,
    externalstress=0.0,
    ttime=ttime,
    pfactor=pfactor,
    mask=np.eye(3, dtype=int),
)

dyn.attach(write_frame, interval=1000)
dyn.attach(MDLogger(dyn, atoms, log_file, stress=True, peratom=False, mode="a"), interval=100)
dyn.attach(remove_com, interval=10000)

n_steps = 1_000_000
dyn.run(n_steps)

print("Melting MD done.")
write(final_data_file, atoms)
