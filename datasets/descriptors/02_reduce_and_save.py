"""Step 2/3: reduce descriptors_raw.npz (from 01_compute_descriptors.py) to 2D via t-SNE.
Cheap (no JAX) -- re-run freely with different t-SNE settings."""

import os
import re
import datetime

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ============================== CONFIG ======================================
n_components = 2
perplexity = 30           # best embedding found among {3, 30, 200} at ratio=0.1
learning_rate = "auto"
init = "pca"              # more stable/reproducible than sklearn's default 'random'
random_state = 42
standardize_before_tsne = True

RUN_NAME = "ratio_0.10"        # this run's output subdir, under runs/
INPUT_RUN_NAME = "ratio_0.10"  # which run's descriptors_raw.npz to reduce (set to a
                                # different existing run to retry t-SNE settings without recomputing)

# ============================== PATHS =======================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
INPUT_PATH = os.path.join(SCRIPT_DIR, "runs", INPUT_RUN_NAME, "descriptors_raw.npz")
OUTPUT_NPY_PATH = os.path.join(RUN_DIR, "descriptors_reduced.npy")
OUTPUT_INFO_PATH = os.path.join(RUN_DIR, "INFO.txt")


def parse_nominal_system_size(system_name):
    """Nominal system size from a folder name, e.g. 'Cu30' -> 30. Differs from the real
    per-frame atom count for '_expanded' systems (supercell-replicated for the cutoff radius)."""
    match = re.match(r"Cu(\d+)", str(system_name))
    return int(match.group(1))


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    raw = np.load(INPUT_PATH, allow_pickle=True)
    descriptors = raw["descriptors"]
    n_rows = descriptors.shape[0]
    print(f"Loaded {n_rows} descriptor rows (dim={descriptors.shape[1]}) from {INPUT_PATH}")

    X = descriptors
    if standardize_before_tsne:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Standardized descriptors before t-SNE.")

    print(
        f"Running t-SNE: n_components={n_components}, perplexity={perplexity}, "
        f"learning_rate={learning_rate}, init={init}, random_state={random_state}"
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
    )
    tsne_embedding = tsne.fit_transform(X).astype(np.float32)
    print("t-SNE done.")

    system_names = raw["system_names"]
    system_id = raw["system_id"]

    nominal_size_per_system = np.array(
        [parse_nominal_system_size(name) for name in system_names], dtype=np.int32
    )
    nominal_system_size = nominal_size_per_system[system_id]

    # Per-system row-count breakdown, for INFO.txt
    per_system_counts = {}
    for sid, name in enumerate(system_names):
        per_system_counts[str(name)] = int((system_id == sid).sum())

    config = {
        "run_name": RUN_NAME,
        "input_run_name": INPUT_RUN_NAME,
        "frame_sample_ratio": float(raw["frame_sample_ratio"]),
        "min_frames_per_system": int(raw["min_frames_per_system"]),
        "tsne_n_components": n_components,
        "tsne_perplexity": perplexity,
        "tsne_learning_rate": learning_rate,
        "tsne_init": init,
        "tsne_random_state": random_state,
        "standardize_before_tsne": standardize_before_tsne,
        "model_path": os.path.join(SCRIPT_DIR, "..", "..", "models", "abinit", "model", "model_abinit.pkl"),
        "dataset_path": os.path.join(SCRIPT_DIR, "..", "ab_initio", "Cu_DATA", "pbc"),
        "generated_at": datetime.datetime.now().isoformat(),
        "evaluate_descriptors_batch_size": 1,
    }

    data = {
        "tsne_embedding": tsne_embedding,
        "descriptors": descriptors,
        "energy_total": raw["energy_total"],
        "energy_per_atom": raw["energy_per_atom"],
        "force": raw["force"],
        "force_magnitude": raw["force_magnitude"],
        "n_atoms": raw["n_atoms"],
        "nominal_system_size": nominal_system_size,
        "system_id": system_id,
        "system_names": system_names,
        "atom_type": raw["atom_type"],
        "atom_index_in_frame": raw["atom_index_in_frame"],
        "global_frame_id": raw["global_frame_id"],
        "orig_frame_index": raw["orig_frame_index"],
        "config": config,
    }
    np.save(OUTPUT_NPY_PATH, data, allow_pickle=True)
    print(f"Saved {OUTPUT_NPY_PATH}")

    write_info_txt(OUTPUT_INFO_PATH, data, n_rows, per_system_counts)
    print(f"Saved {OUTPUT_INFO_PATH}")


def write_info_txt(path, data, n_rows, per_system_counts):
    """Human-readable companion to descriptors_reduced.npy: array shapes + config."""
    cfg = data["config"]

    lines = []
    lines.append("descriptors_reduced.npy -- contents description")
    lines.append("=" * 60)
    lines.append(f"Generated: {cfg['generated_at']}")
    lines.append(f"Model: {cfg['model_path']}")
    lines.append(f"Dataset: {cfg['dataset_path']}")
    lines.append("One row per atom per sampled frame (not per frame).")
    lines.append("")
    lines.append("ARRAY REFERENCE "
                  f"(row-aligned, n_rows={n_rows}, except system_names)")
    lines.append("-" * 60)
    lines.append(f"tsne_embedding       ({n_rows}, {cfg['tsne_n_components']}) float32 -- t-SNE coords")
    lines.append(f"descriptors          ({n_rows}, 768) float32 -- pre-t-SNE 12x64 DeepPot-SE vectors")
    lines.append(f"energy_total         ({n_rows},) float64, eV -- frame's total DFT energy (repeated)")
    lines.append(f"energy_per_atom      ({n_rows},) float64, eV -- energy_total / n_atoms")
    lines.append(f"force                ({n_rows}, 3) float64, eV/A -- this atom's own force")
    lines.append(f"force_magnitude      ({n_rows},) float64, eV/A")
    lines.append(f"n_atoms              ({n_rows},) int32 -- real per-frame atom count")
    lines.append(f"nominal_system_size  ({n_rows},) int32 -- parsed from folder name, not n_atoms")
    lines.append(f"system_id            ({n_rows},) int32 -- index into system_names")
    lines.append(f"system_names         ({len(data['system_names'])},) string array, index by system_id")
    lines.append(f"atom_type            ({n_rows},) int32 -- type.raw index")
    lines.append(f"atom_index_in_frame  ({n_rows},) int")
    lines.append(f"global_frame_id      ({n_rows},) int -- unique per sampled (system, frame)")
    lines.append(f"orig_frame_index     ({n_rows},) int -- index in the pre-sampling frame ordering")
    lines.append("config               dict (see below)")
    lines.append("")
    lines.append("PER-SYSTEM ROW COUNTS")
    lines.append("-" * 60)
    for name, count in per_system_counts.items():
        lines.append(f"  {name}: {count}")
    lines.append("")
    lines.append("HOW TO LOAD")
    lines.append("-" * 60)
    lines.append("  data = np.load('descriptors_reduced.npy', allow_pickle=True).item()")
    lines.append("")
    lines.append("CONFIG")
    lines.append("-" * 60)
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
