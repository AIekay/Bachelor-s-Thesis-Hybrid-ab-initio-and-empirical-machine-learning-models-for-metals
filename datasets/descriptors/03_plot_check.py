"""Step 3/3: sanity-check scatter plots from descriptors_reduced.npy alone (no JAX needed)."""

import os

import numpy as np
import matplotlib.pyplot as plt

RUN_NAME = "ratio_0.10"  # must match 01_compute_descriptors.py / 02_reduce_and_save.py

# Force magnitude has a long tail -- clip the color scale so outliers don't flatten it.
FORCE_COLOR_CLIP_PERCENTILE = 95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)
DATA_PATH = os.path.join(RUN_DIR, "descriptors_reduced.npy")
OUT_PATH = os.path.join(RUN_DIR, "tsne_sanity_check.png")

data = np.load(DATA_PATH, allow_pickle=True).item()
xy = data["tsne_embedding"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

sc0 = axes[0].scatter(
    xy[:, 0], xy[:, 1], c=data["energy_per_atom"], cmap="viridis", s=4, linewidths=0
)
axes[0].set_title("colored by energy per atom (eV)")
fig.colorbar(sc0, ax=axes[0])

# nominal_system_size, not n_atoms -- n_atoms would show Cu1_expanded/Cu2_expanded as huge.
sc1 = axes[1].scatter(
    xy[:, 0], xy[:, 1], c=data["nominal_system_size"], cmap="viridis", s=4, linewidths=0
)
axes[1].set_title("colored by nominal system size (pre-supercell-expansion)")
fig.colorbar(sc1, ax=axes[1])

force_vmax = np.percentile(data["force_magnitude"], FORCE_COLOR_CLIP_PERCENTILE)
sc2 = axes[2].scatter(
    xy[:, 0], xy[:, 1], c=data["force_magnitude"], cmap="viridis", s=4, linewidths=0,
    vmin=0, vmax=force_vmax,
)
axes[2].set_title(
    f"colored by |force| (eV/Angstrom), clipped at p{FORCE_COLOR_CLIP_PERCENTILE}={force_vmax:.2f}"
)
fig.colorbar(sc2, ax=axes[2], extend="max")

for ax in axes:
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

fig.suptitle(
    f"t-SNE of atomic-environment descriptors "
    f"(n={xy.shape[0]} atom-rows, ratio={data['config']['frame_sample_ratio']})"
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved {OUT_PATH}")
