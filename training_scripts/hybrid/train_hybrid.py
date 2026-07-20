import numpy as np
from deepmd_jax.train2 import train
import argparse
import os

root_path = '<DATA_ROOT>'  # root directory containing Cu_DATA/ and DATA_EXP/ (see below)

root_path_abinit = root_path
# DATA_EXP_<T>K folders from molecular_dynamics/trajectory_to_dataset/, using the PREVIOUS
# generation's MD (see ../../models/README.md for the lineage).
root_path_exp = os.path.join(root_path, 'DATA_EXP')

raw_data_paths_abinit = [
    'Cu_DATA/pbc/Cu1_expanded',
    'Cu_DATA/pbc/Cu2_expanded',
    'Cu_DATA/pbc/Cu3',
    'Cu_DATA/pbc/Cu4',
    'Cu_DATA/pbc/Cu5',
    'Cu_DATA/pbc/Cu6',
    'Cu_DATA/pbc/Cu7',
    'Cu_DATA/pbc/Cu8',
    'Cu_DATA/pbc/Cu9',
    'Cu_DATA/pbc/Cu10',
    'Cu_DATA/pbc/Cu12',
    'Cu_DATA/pbc/Cu14',
    'Cu_DATA/pbc/Cu15',
    'Cu_DATA/pbc/Cu16',
    'Cu_DATA/pbc/Cu21',
    'Cu_DATA/pbc/Cu24',
    'Cu_DATA/pbc/Cu30',
    'Cu_DATA/pbc/Cu31',
    'Cu_DATA/pbc/Cu32',
    'Cu_DATA/pbc/Cu37',
    'Cu_DATA/pbc/Cu51',
    'Cu_DATA/pbc/Cu52',
    'Cu_DATA/pbc/Cu53',
    'Cu_DATA/pbc/Cu54',
    'Cu_DATA/pbc/Cu55',
    'Cu_DATA/pbc/Cu56',
    'Cu_DATA/pbc/Cu58',
    'Cu_DATA/pbc/Cu62',
    'Cu_DATA/pbc/Cu63',
    'Cu_DATA/pbc/Cu64',
    'Cu_DATA/pbc/Cu96',
    'Cu_DATA/pbc/Cu105',
    'Cu_DATA/pbc/Cu106',
    'Cu_DATA/pbc/Cu107',
    'Cu_DATA/pbc/Cu108',
    'Cu_DATA/pbc/Cu125',
    'Cu_DATA/pbc/Cu128',
]

# 1300K excluded: source model's melting point was below 1300K, so that trajectory
# equilibrated liquid instead of solid, corrupting the observable gradient.
raw_data_paths_exp = [
    'DATA_EXP_800K',
    'DATA_EXP_900K',
    'DATA_EXP_1000K',
    'DATA_EXP_1100K',
    'DATA_EXP_1200K',
    'DATA_EXP_1400K',
    'DATA_EXP_1500K',
    'DATA_EXP_1600K',
    'DATA_EXP_1700K',
    'DATA_EXP_1800K',
    'DATA_EXP_1900K',
    'DATA_EXP_2000K',
]

parser = argparse.ArgumentParser(
    description='Train a hybrid energy-force-density Cu model '
                '(representative of hyb1/hyb2/hyb3 -- see ../README.md).')
parser.add_argument('--steps', type=int, default=500,
                    help='Number of training steps (default: 500). '
                         'The released hyb1/hyb2/hyb3 models were each trained with '
                         '--steps=5000000 (~18.3-18.5h on 1 GPU) -- the default here is a '
                         'fast local smoke test.')
args = parser.parse_args()

data_paths_abinit = [os.path.join(root_path_abinit, p) for p in raw_data_paths_abinit]
data_paths_exp = [os.path.join(root_path_exp, p) for p in raw_data_paths_exp]

model_name = 'model_hybrid.pkl'

obs_temperatures = [800, 900, 1000, 1100, 1200, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

obs_densities = [
    8.68881782, 8.63603972, 8.58153597, 8.52520065, 8.46695053,  # 800K-1200K (Solid)
    7.86872158, 7.79231586, 7.71591014, 7.63950443, 7.56309871, 7.48669299, 7.41028728,  # 1400K-2000K (Liquid)
]

# Same prefactors used to train hyb1/hyb2/hyb3 -- only the data paths differ per generation.
train(
    model_type='energy',
    hybrid=True,
    rcut=6.0,
    save_path=model_name,
    l_pref_e=0.1,
    s_pref_e=0.1,
    train_data_path=data_paths_abinit,
    obs_train_data_path=data_paths_exp,
    step=args.steps,
    print_loss_smoothing=20,
    print_every=1000,
    obs_temperature=obs_temperatures,
    obs_target=obs_densities,
    obs_batch_size=1000,
    obs_s_pref=1.0,
    obs_l_pref=100,
    obs_step_every=10,
    compress=True,
)
