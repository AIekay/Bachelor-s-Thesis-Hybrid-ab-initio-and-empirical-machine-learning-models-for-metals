"""Post-training plots for a hybrid training run: parses log.txt (as produced by train_hybrid.py)
and writes total loss, RMSE, per-temperature observable loss/ESS/density-evolution plots to
RESULTS/."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Must match obs_temperatures and obs_densities in train_hybrid.py exactly.
# 1300K is excluded — indices here correspond to the log keys (LOBS0, LOBS1, ...).
obs_temperatures = [800, 900, 1000, 1100, 1200, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

obs_densities = [
    8.68881782, 8.63603972, 8.58153597, 8.52520065, 8.46695053,  # 800K–1200K (Solid)
    7.86872158, 7.79231586, 7.71591014, 7.63950443, 7.56309871, 7.48669299, 7.41028728,  # 1400K–2000K (Liquid)
]

# Phase classification and color gradients:
#   Solid (800-1200K, indices 0-4):  green → blue
#   Liquid (1400-2000K, indices 5-11): red → yellow
N_SOLID  = 5
N_LIQUID = 7
solid_cmap  = LinearSegmentedColormap.from_list('solid',  ['green', 'blue'])
liquid_cmap = LinearSegmentedColormap.from_list('liquid', ['red',   'yellow'])

def temp_color(obs_idx):
    if obs_idx < N_SOLID:
        return solid_cmap(obs_idx / max(N_SOLID - 1, 1))
    else:
        liq_idx = obs_idx - N_SOLID
        return liquid_cmap(liq_idx / max(N_LIQUID - 1, 1))


def parse_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Iter"):
                tokens = line.split()
                for i in range(0, len(tokens), 2):
                    key = tokens[i]
                    if key == "Time":
                        continue
                    value = float(tokens[i + 1])
                    if key not in data:
                        data[key] = []
                    data[key].append(value)
    for key in data:
        data[key] = np.array(data[key])
    return data


def running_avg(x, window_size):
    if window_size <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window_size + 1)
        count = i - start + 1
        out[i] = (cumsum[i + 1] - cumsum[start]) / count
    return out


def main():
    filepath = 'log.txt'
    if not os.path.exists(filepath):
        print(f"Log file '{filepath}' not found — nothing to plot.")
        return

    data = parse_file(filepath)
    if 'Iter' not in data:
        print("No valid iteration data found in log.")
        return

    os.makedirs('RESULTS', exist_ok=True)
    iterations = data['Iter']
    window = 50
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 1. Total Loss
    if 'L' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, data['L'], color=colors[0], alpha=0.3, linewidth=1)
        plt.plot(iterations, running_avg(data['L'], window), color=colors[0], linewidth=2, label='Total Loss (L)')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Total Loss Evolution")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(left=0.1)
        plt.savefig(os.path.join("RESULTS", "total_loss_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. RMSE Energy and Force
    if 'LE' in data and 'LF' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, data['LE'], color=colors[0], alpha=0.3, linewidth=1)
        plt.plot(iterations, data['LF'], color=colors[1], alpha=0.3, linewidth=1)
        plt.plot(iterations, running_avg(data['LE'], window), color=colors[0], linewidth=2, label="RMSE Energy (LE)")
        plt.plot(iterations, running_avg(data['LF'], window), color=colors[1], linewidth=2, label="RMSE Force (LF)")
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.yscale("log")
        plt.title("RMSE Energy and Force Evolution")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(left=0.1)
        plt.savefig(os.path.join("RESULTS", "rmse_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Observable Losses (LOBS)
    lobs_keys = [k for k in data.keys() if k.startswith('LOBS')]
    if lobs_keys:
        sorted_lobs_keys = sorted(lobs_keys, key=lambda k: int(k[4:]))
        plt.figure(figsize=(10, 6))
        for k in sorted_lobs_keys:
            try:
                obs_idx = int(k[4:])
                temp = obs_temperatures[obs_idx]
                label = f"{temp} K"
                color = temp_color(obs_idx)
            except (ValueError, IndexError):
                label = k
                color = 'gray'
            plt.plot(iterations, data[k], color=color, alpha=0.2, linewidth=0.5)
            plt.plot(iterations, running_avg(data[k], window), color=color, linewidth=2, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Observable Loss")
        plt.yscale("log")
        plt.title("Observable Losses Evolution")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(left=0.1)
        plt.savefig(os.path.join("RESULTS", "lobs_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Effective Sample Sizes (ESS)
    ess_keys = [k for k in data.keys() if k.startswith('ESS')]
    if ess_keys:
        sorted_ess_keys = sorted(ess_keys, key=lambda k: int(k[3:]))
        plt.figure(figsize=(10, 6))
        for k in sorted_ess_keys:
            try:
                obs_idx = int(k[3:])
                temp = obs_temperatures[obs_idx]
                label = f"{temp} K"
                color = temp_color(obs_idx)
            except (ValueError, IndexError):
                label = k
                color = 'gray'
            plt.plot(iterations, data[k], color=color, alpha=0.2, linewidth=0.5)
            plt.plot(iterations, running_avg(data[k], window), color=color, linewidth=2, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("ESS")
        plt.title("Effective Sample Size (ESS) Evolution")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, ls="--", alpha=0.5)
        plt.xlim(left=0.0)
        plt.savefig(os.path.join("RESULTS", "ess_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. OBS vs OBS_REW pairs (density evolution per temperature)
    obs_rew_keys = [k for k in data.keys() if k.startswith('OBS_REW_')]
    x_min_limit = iterations.max() * 0.1
    for rew_k in obs_rew_keys:
        base_suffix = rew_k.replace('OBS_REW_', '')
        obs_k = f"OBS_{base_suffix}"
        if obs_k not in data:
            continue

        temp = None
        try:
            obs_idx = int(base_suffix.split('_')[0])
            if obs_idx < len(obs_temperatures):
                temp = obs_temperatures[obs_idx]
        except (ValueError, IndexError):
            pass

        rew_label = f"Reweighted Mean ({temp} K)" if temp is not None else f"Reweighted Mean ({rew_k})"
        obs_label = f"Unweighted Batch Mean ({temp} K)" if temp is not None else f"Unweighted Batch Mean ({obs_k})"

        plt.figure(figsize=(10, 6))
        plt.scatter(iterations, data[rew_k], color=colors[0], alpha=0.25, s=4, zorder=2)
        plt.scatter(iterations, data[obs_k], color=colors[1], alpha=0.25, s=4, zorder=2)
        rew_smooth = running_avg(data[rew_k], window)
        obs_smooth = running_avg(data[obs_k], window)
        plt.plot(iterations, rew_smooth, color=colors[0], linewidth=2, label=rew_label, zorder=3)
        plt.plot(iterations, obs_smooth, color=colors[1], linewidth=2, label=obs_label, zorder=3)

        title = f"Observable {base_suffix} Evolution"
        filename = f"obs_{base_suffix}_evolution.png"
        try:
            obs_idx = int(base_suffix.split('_')[0])
            if obs_idx < len(obs_densities):
                target_val = obs_densities[obs_idx]
                plt.axhline(y=target_val, color='black', linestyle='--', linewidth=1.5,
                            alpha=0.8, label=f"Target ({target_val:.3f})", zorder=1)
                mask = iterations >= x_min_limit
                rew_vis = data[rew_k][mask] if np.any(mask) else data[rew_k]
                obs_vis = data[obs_k][mask] if np.any(mask) else data[obs_k]
                all_vis = np.concatenate([rew_vis, obs_vis])
                y_min = min(np.percentile(all_vis, 2), target_val)
                y_max = max(np.percentile(all_vis, 98), target_val)
                y_buf = max((y_max - y_min) * 0.15, 0.05)
                plt.ylim(y_min - y_buf, y_max + y_buf)
            if obs_idx < len(obs_temperatures):
                temp = obs_temperatures[obs_idx]
                title = f"Density Evolution at {temp} K"
                filename = f"density_{temp}K_evolution.png"
        except (ValueError, IndexError):
            pass

        plt.xlabel("Iteration")
        plt.ylabel("Density (g/cm³)")
        plt.title(title)
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.xlim(left=x_min_limit)
        plt.savefig(os.path.join("RESULTS", filename), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Observable error comparison: running avg(OBS_REW_i) - target_i for all temperatures
    #    Uses the same solid/liquid color convention as the ESS plot.
    obs_rew_keys_sorted = sorted(
        [k for k in data.keys() if k.startswith('OBS_REW_')],
        key=lambda k: int(k.replace('OBS_REW_', '').split('_')[0])
    )
    if obs_rew_keys_sorted:
        # iteration 0 has OBS_REW=0 (spurious spike) -- excluded from y-axis clipping below
        error_series = {}
        skip_init = iterations > 5000
        all_post_init_errors = []

        for rew_k in obs_rew_keys_sorted:
            base_suffix = rew_k.replace('OBS_REW_', '')
            try:
                obs_idx = int(base_suffix.split('_')[0])
                if obs_idx >= len(obs_densities) or obs_idx >= len(obs_temperatures):
                    continue
                target_val = obs_densities[obs_idx]
                temp = obs_temperatures[obs_idx]
                color = temp_color(obs_idx)
            except (ValueError, IndexError):
                continue

            error = running_avg(data[rew_k], window) - target_val
            error_series[rew_k] = (error, temp, color)
            all_post_init_errors.extend(error[skip_init].tolist())

        y_min = np.percentile(all_post_init_errors, 1)
        y_max = np.percentile(all_post_init_errors, 99)
        y_buf = max((y_max - y_min) * 0.15, 0.002)

        plt.figure(figsize=(12, 6))
        for rew_k, (error, temp, color) in error_series.items():
            plt.plot(iterations, error, color=color, linewidth=2, label=f"{temp} K")

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        plt.ylim(y_min - y_buf, y_max + y_buf)
        plt.xlabel("Iteration")
        plt.ylabel("Error (g/cm³)")
        plt.title("Reweighted Observable Error Evolution (Running Avg − Target)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Temperature")
        plt.grid(True, ls="--", alpha=0.5)
        plt.xlim(left=0.0)
        plt.savefig(os.path.join("RESULTS", "obs_error_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print("[PLOT] Training plots saved to RESULTS/")


if __name__ == "__main__":
    main()
