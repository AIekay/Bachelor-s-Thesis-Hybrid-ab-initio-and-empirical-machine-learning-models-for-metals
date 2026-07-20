import numpy as np
from deepmd_jax.train import train
import argparse

root_path = '<DATA_ROOT>'  # root directory containing Cu_DATA/ (see datasets/ab_initio/README.md)


raw_data_path = [   '/Cu_DATA/pbc/Cu1_expanded',
                    '/Cu_DATA/pbc/Cu2_expanded',
                    '/Cu_DATA/pbc/Cu3',
                    '/Cu_DATA/pbc/Cu3',
                    '/Cu_DATA/pbc/Cu4',
                    '/Cu_DATA/pbc/Cu5',
                    '/Cu_DATA/pbc/Cu6',
                    '/Cu_DATA/pbc/Cu7',
                    '/Cu_DATA/pbc/Cu8',
                    '/Cu_DATA/pbc/Cu9',
                    '/Cu_DATA/pbc/Cu10',
                    '/Cu_DATA/pbc/Cu12',

                    '/Cu_DATA/pbc/Cu14',
                    '/Cu_DATA/pbc/Cu15',
                    '/Cu_DATA/pbc/Cu16',
                    '/Cu_DATA/pbc/Cu21',
                    '/Cu_DATA/pbc/Cu24',
                    '/Cu_DATA/pbc/Cu30',
                    '/Cu_DATA/pbc/Cu31',
                    '/Cu_DATA/pbc/Cu32',
                    '/Cu_DATA/pbc/Cu37',
                    '/Cu_DATA/pbc/Cu51',
                    '/Cu_DATA/pbc/Cu52',
                    '/Cu_DATA/pbc/Cu53',
                    '/Cu_DATA/pbc/Cu54',
                    '/Cu_DATA/pbc/Cu55',
                    '/Cu_DATA/pbc/Cu56',
                    '/Cu_DATA/pbc/Cu58',
                    '/Cu_DATA/pbc/Cu62',
                    '/Cu_DATA/pbc/Cu63',
                    '/Cu_DATA/pbc/Cu64',
                    '/Cu_DATA/pbc/Cu96',
                    '/Cu_DATA/pbc/Cu105',
                    '/Cu_DATA/pbc/Cu106',
                    '/Cu_DATA/pbc/Cu107',
                    '/Cu_DATA/pbc/Cu108',
                    '/Cu_DATA/pbc/Cu125',
                    '/Cu_DATA/pbc/Cu128',]

data_paths = [root_path + path for path in raw_data_path]

# NOTE: released as 'model_abinit.pkl'; an earlier version of this script computed
# 'model_abinitio.pkl' here — fixed to match.
model_name = 'model_abinit.pkl'
model_path = root_path + '/' + model_name

parser = argparse.ArgumentParser(description='Train an ab initio (DFT energy + force) Cu model.')
parser.add_argument('--steps', type=int, default=10000,
                    help='Number of training steps (default: 10000). '
                         'The released abinit model was trained with --steps=10000000 '
                         '(~4h52m on 1 GPU) — the default here is a fast local smoke test.')
args = parser.parse_args()

train(
        model_type='energy',
        rcut=6.0,
        l_pref_e = 1,
        s_pref_e = 1,   # NOTE: original script's comment said 0.02 here; 1 is what shipped
        save_path=model_path,
        #print_distribution=True,
        train_data_path=data_paths,
        step=args.steps,
        )
