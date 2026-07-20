import numpy as np
import matplotlib.pyplot as plt
import os
from deepmd_jax.train import test as deepmd_test

root_path = '<DATA_ROOT>'  # root directory containing Cu_DATA/ -- must match train_abinit.py

raw_data_paths_abinit = [
    'Cu_DATA/pbc/Cu1_expanded',
    'Cu_DATA/pbc/Cu2_expanded',
    'Cu_DATA/pbc/Cu3',
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

# train_abinit.py saves under root_path, not the current directory (unlike hybrid/test.py).
model_path = os.path.join(root_path, 'model_abinit.pkl')
it_label = os.path.basename(os.path.abspath('.'))  # label from directory name


def evaluate(data_path):
    dataset_name = os.path.basename(data_path)
    print(f"Testing {dataset_name}...")
    _, _, _, predictions, ground_truth = deepmd_test(model_path, data_path)
    Natoms = predictions['force'].shape[1]
    e_rmse = np.sqrt(np.mean((predictions['energy'] - ground_truth['energy']) ** 2)) * 1000 / Natoms
    f_rmse = np.sqrt(np.mean((predictions['force'].flatten() - ground_truth['force'].flatten()) ** 2)) * 1000
    return {'name': dataset_name, 'energy_rmse': e_rmse, 'force_rmse': f_rmse}


def main():
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found — aborting.")
        return

    data_paths_abinit = [os.path.join(root_path, p) for p in raw_data_paths_abinit]

    results = []
    for data_path in data_paths_abinit:
        try:
            results.append(evaluate(data_path))
        except Exception as exc:
            print(f"WARNING: failed on {os.path.basename(data_path)}: {exc}")

    if not results:
        print("No results to report.")
        return

    os.makedirs('RESULTS', exist_ok=True)

    e_rmses = [r['energy_rmse'] for r in results]
    f_rmses = [r['force_rmse'] for r in results]
    max_e_r = results[int(np.argmax(e_rmses))]
    max_f_r = results[int(np.argmax(f_rmses))]

    # --- Text report ---
    report_path = os.path.join('RESULTS', 'test_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"        {it_label} — MODEL TEST REPORT (ab initio only)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n\n")

        f.write("--- Individual Dataset Results ---\n")
        for r in results:
            msg = (f"{r['name']:<20}: "
                   f"E_RMSE = {r['energy_rmse']:>7.3f} meV/atom, "
                   f"F_RMSE = {r['force_rmse']:>7.3f} meV/Å")
            print(msg)
            f.write(msg + "\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("--- Summary ---\n")
        f.write(f"Total datasets tested : {len(results)}\n")
        f.write(f"Overall Mean E RMSE   : {np.mean(e_rmses):.3f} meV/atom\n")
        f.write(f"Overall Mean F RMSE   : {np.mean(f_rmses):.3f} meV/Å\n")
        f.write(f"\nMax E RMSE: {max_e_r['energy_rmse']:.3f} meV/atom  ({max_e_r['name']})\n")
        f.write(f"Max F RMSE: {max_f_r['force_rmse']:.3f} meV/Å  ({max_f_r['name']})\n")
        f.write("=" * 60 + "\n")

    # --- Bar chart ---
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(18, 7))
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('Energy RMSE (meV/atom)', color='tab:blue', fontweight='bold')
    ax1.bar(x - width / 2, e_rmses, width, color='tab:blue', alpha=0.8, label='Energy RMSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=7)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Force RMSE (meV/Å)', color='tab:red', fontweight='bold')
    ax2.bar(x + width / 2, f_rmses, width, color='tab:red', alpha=0.6, label='Force RMSE')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    max_e_idx = int(np.argmax(e_rmses))
    max_f_idx = int(np.argmax(f_rmses))
    ax1.bar(max_e_idx - width / 2, e_rmses[max_e_idx], width,
            color='navy', edgecolor='black', linewidth=1.5)
    ax2.bar(max_f_idx + width / 2, f_rmses[max_f_idx], width,
            color='darkred', edgecolor='black', linewidth=1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.title(f'{it_label} — RMSE by Dataset (ab initio only)', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join('RESULTS', 'test_rmse_by_dataset.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nReport saved to: {report_path}")
    print(f"Plot saved to:   {plot_path}")


if __name__ == "__main__":
    main()
