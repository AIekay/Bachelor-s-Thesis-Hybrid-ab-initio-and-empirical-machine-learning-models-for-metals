"""Step 1/3: compute per-atom descriptors for a sampled subset of frames from every
system in Cu_DATA/pbc. Expensive (JAX model evaluation) -- output consumed by
02_reduce_and_save.py, which can be re-run independently with different t-SNE settings."""

import os
import glob
import shutil
import tempfile
import time

import numpy as np
from deepmd_jax.train import evaluate_descriptors

# ============================== CONFIG ======================================
# Fraction of each system's frames to sample, applied per system folder (not globally).
# Rows fed to t-SNE = sampled_frames x atoms_per_frame, summed over systems -- Cu2_expanded
# (128 atoms/frame) dominates unless this is kept small.
frame_sample_ratio = 0.1

# Floor so small systems (e.g. Cu96/Cu125/Cu128, 1-2 frames total) stay represented.
min_frames_per_system = 1

# Output subdirectory under runs/ -- MUST match RUN_NAME in 02_reduce_and_save.py / 03_plot_check.py.
RUN_NAME = "ratio_0.10"

# ============================== PATHS =======================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "..", "ab_initio", "Cu_DATA", "pbc")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "abinit", "model", "model_abinit.pkl")
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
OUTPUT_PATH = os.path.join(RUN_DIR, "descriptors_raw.npz")

# Hardcoded, not a config knob: evaluate_descriptors has a confirmed bug where
# batch_size > 1 silently drops all but the first frame of each batch.
EVAL_BATCH_SIZE = 1


def load_system_frames(system_dir):
    """Load and concatenate all set.* subdirectories of a system folder.

    Returns dict with keys: type_idx (n_atoms,), type_map (list[str]),
    box (n_frames,9), coord (n_frames,3*n_atoms), energy (n_frames,),
    force (n_frames,3*n_atoms).
    """
    type_idx = np.loadtxt(os.path.join(system_dir, "type.raw"), dtype=np.int32, ndmin=1)
    with open(os.path.join(system_dir, "type_map.raw")) as f:
        type_map = f.read().split()

    set_dirs = sorted(glob.glob(os.path.join(system_dir, "set.*")))
    boxes, coords, energies, forces = [], [], [], []
    for set_dir in set_dirs:
        boxes.append(np.load(os.path.join(set_dir, "box.npy")))
        coords.append(np.load(os.path.join(set_dir, "coord.npy")))
        energies.append(np.load(os.path.join(set_dir, "energy.npy")))
        forces.append(np.load(os.path.join(set_dir, "force.npy")))

    return {
        "type_idx": type_idx,
        "type_map": type_map,
        "box": np.concatenate(boxes, axis=0),
        "coord": np.concatenate(coords, axis=0),
        "energy": np.concatenate(energies, axis=0),
        "force": np.concatenate(forces, axis=0),
    }


def select_sample_indices(n_frames, ratio, min_frames):
    n_sample = max(min_frames, round(n_frames * ratio))
    n_sample = min(n_sample, n_frames)
    return np.linspace(0, n_frames - 1, n_sample).astype(int)


def build_temp_dataset(frames, sample_idx, tmp_root):
    """Write a DeePMD-format dataset containing only the sampled frames."""
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    np.savetxt(os.path.join(tmp_dir, "type.raw"), frames["type_idx"], fmt="%d")
    with open(os.path.join(tmp_dir, "type_map.raw"), "w") as f:
        f.write("\n".join(frames["type_map"]) + "\n")

    set_dir = os.path.join(tmp_dir, "set.000")
    os.makedirs(set_dir)
    np.save(os.path.join(set_dir, "box.npy"), frames["box"][sample_idx])
    np.save(os.path.join(set_dir, "coord.npy"), frames["coord"][sample_idx])
    np.save(os.path.join(set_dir, "energy.npy"), frames["energy"][sample_idx])
    np.save(os.path.join(set_dir, "force.npy"), frames["force"][sample_idx])
    return tmp_dir


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"Run directory: {RUN_DIR}")

    system_dirs = sorted(
        d for d in glob.glob(os.path.join(DATASET_ROOT, "*")) if os.path.isdir(d)
    )
    print(f"Found {len(system_dirs)} system folders under {DATASET_ROOT}")

    tmp_root = tempfile.mkdtemp(prefix="descr_sampling_")

    all_descriptors = []
    all_energy_total = []
    all_energy_per_atom = []
    all_force = []
    all_n_atoms = []
    all_system_id = []
    all_atom_type = []
    all_atom_index_in_frame = []
    all_global_frame_id = []
    all_orig_frame_index = []
    system_names = []
    per_system_row_counts = {}

    global_frame_counter = 0
    t_start = time.time()

    try:
        for system_id, system_dir in enumerate(system_dirs):
            name = os.path.basename(system_dir)
            system_names.append(name)

            frames = load_system_frames(system_dir)
            n_frames_total = frames["coord"].shape[0]
            n_atoms = frames["type_idx"].shape[0]

            sample_idx = select_sample_indices(
                n_frames_total, frame_sample_ratio, min_frames_per_system
            )
            n_sample = len(sample_idx)

            print(
                f"[{system_id + 1}/{len(system_dirs)}] {name}: "
                f"{n_atoms} atoms, {n_frames_total} frames total, "
                f"sampling {n_sample} frames"
            )

            temp_dir = build_temp_dataset(frames, sample_idx, tmp_root)
            try:
                desc_list = evaluate_descriptors(
                    MODEL_PATH, temp_dir, batch_size=EVAL_BATCH_SIZE
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # desc_list: length n_sample, each entry (n_atoms, axis=12, embed_width=64)
            desc_arr = np.asarray(desc_list)  # (n_sample, n_atoms, 12, 64)
            desc_arr = desc_arr.reshape(n_sample, n_atoms, -1)  # (n_sample, n_atoms, 768)

            energy_sampled = frames["energy"][sample_idx]  # (n_sample,)
            force_sampled = frames["force"][sample_idx].reshape(n_sample, n_atoms, 3)

            # Flatten (n_sample, n_atoms, ...) -> (n_sample * n_atoms, ...)
            all_descriptors.append(desc_arr.reshape(n_sample * n_atoms, -1))
            all_force.append(force_sampled.reshape(n_sample * n_atoms, 3))

            energy_total_row = np.repeat(energy_sampled, n_atoms)
            all_energy_total.append(energy_total_row)
            all_energy_per_atom.append(energy_total_row / n_atoms)
            all_n_atoms.append(np.full(n_sample * n_atoms, n_atoms, dtype=np.int32))
            all_system_id.append(np.full(n_sample * n_atoms, system_id, dtype=np.int32))
            all_atom_type.append(np.tile(frames["type_idx"], n_sample))
            all_atom_index_in_frame.append(np.tile(np.arange(n_atoms), n_sample))

            frame_ids = global_frame_counter + np.arange(n_sample)
            all_global_frame_id.append(np.repeat(frame_ids, n_atoms))
            all_orig_frame_index.append(np.repeat(sample_idx, n_atoms))
            global_frame_counter += n_sample

            per_system_row_counts[name] = n_sample * n_atoms

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s.")

    descriptors = np.concatenate(all_descriptors, axis=0).astype(np.float32)
    energy_total = np.concatenate(all_energy_total, axis=0).astype(np.float64)
    energy_per_atom = np.concatenate(all_energy_per_atom, axis=0).astype(np.float64)
    force = np.concatenate(all_force, axis=0).astype(np.float64)
    force_magnitude = np.linalg.norm(force, axis=1)
    n_atoms_arr = np.concatenate(all_n_atoms, axis=0)
    system_id_arr = np.concatenate(all_system_id, axis=0)
    atom_type_arr = np.concatenate(all_atom_type, axis=0)
    atom_index_in_frame = np.concatenate(all_atom_index_in_frame, axis=0)
    global_frame_id = np.concatenate(all_global_frame_id, axis=0)
    orig_frame_index = np.concatenate(all_orig_frame_index, axis=0)

    print(f"Total rows (atoms x sampled frames, all systems): {descriptors.shape[0]}")
    print("Per-system row counts:")
    for name, count in per_system_row_counts.items():
        print(f"  {name}: {count}")

    np.savez(
        OUTPUT_PATH,
        descriptors=descriptors,
        energy_total=energy_total,
        energy_per_atom=energy_per_atom,
        force=force,
        force_magnitude=force_magnitude,
        n_atoms=n_atoms_arr,
        system_id=system_id_arr,
        system_names=np.array(system_names),
        atom_type=atom_type_arr,
        atom_index_in_frame=atom_index_in_frame,
        global_frame_id=global_frame_id,
        orig_frame_index=orig_frame_index,
        frame_sample_ratio=frame_sample_ratio,
        min_frames_per_system=min_frames_per_system,
    )
    print(f"Saved raw descriptors + metadata to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
