"""Convert an ASE-written MD trajectory (e.g. from ../solid/md_solid.py or ../liquid/md_liquid.py)
into DeePMD-kit raw training files, including a density observable.raw, for the experimental
observable loss used by training_scripts/hybrid/train_hybrid.py."""

import numpy as np
from ase.io import read
from ase.units import mol
import argparse

parser = argparse.ArgumentParser(description='Convert ASE trajectory to raw DeepMD-kit data files.')
parser.add_argument('--input_traj', type=str, required=True,
                    help='Path to the input ASE trajectory file (e.g., md_800K.xyz).')
args = parser.parse_args()

traj = read(args.input_traj, index=':')

# Output files are written to the current working directory.
with open('coord.raw', 'w') as coord_file, \
     open('energy.raw', 'w') as energy_file, \
     open('force.raw', 'w') as force_file, \
     open('box.raw', 'w') as box_file, \
     open('type.raw', 'w') as type_file, \
     open('observable.raw', 'w') as density_file:
        # observable.raw's filename is what marks a dataset as experimental to the training scripts

    for i, atoms in enumerate(traj):
        coords = atoms.get_positions().flatten()
        coord_file.write(' '.join(map(str, coords)) + '\n')

        energy = atoms.get_potential_energy()
        energy_file.write(str(energy) + '\n')

        forces = atoms.get_forces().flatten()
        force_file.write(' '.join(map(str, forces)) + '\n')

        box = atoms.get_cell().flatten()
        box_file.write(' '.join(map(str, box)) + '\n')

        n_atoms = len(atoms)
        M_Cu = 63.546  # g/mol
        mass = n_atoms * M_Cu / mol  # ASE mol = Avogadro number
        volume_cm3 = atoms.get_volume() * 1e-24  # Å^3 -> cm^3
        density = mass / volume_cm3  # g/cm^3
        density_file.write(str(density) + '\n')

        if i == 0:
            symbols = atoms.get_chemical_symbols()
            type_map = {0: "Cu"}
            symbol_to_type = {v: k for k, v in type_map.items()}
            types = np.array([symbol_to_type[s] for s in symbols])
            type_file.write(' '.join(map(str, types)) + '\n')

            # type_map.raw wasn't written by the original script -- added for consistency with
            # the ab initio dataset format (../../datasets/ab_initio/).
            with open('type_map.raw', 'w') as type_map_file:
                type_map_file.write('\n'.join(type_map[k] for k in sorted(type_map)) + '\n')
