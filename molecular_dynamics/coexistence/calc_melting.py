"""
Estimate Cu melting point from solid-liquid interface MD runs.
Reads COLVAR files from RESULTS/MELTING/<T>/ and fits Delta_mu vs T.

Usage:
    python calc_melting.py --temperatures 1200 1300 1400
    python calc_melting.py                          # uses default list below
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TEMPS = [1200, 1300, 1400]
KAPPA = 0.5   # kJ/mol  (must match KAPPA in plumed.dat)
N0    = 864.0 # half the total number of atoms (solid half at T_melt)

# Label only (this script never loads a model) -- set to whichever model.pkl you ran run-ase.py with.
model_name = "<PATH_TO_YOUR_MODEL>.pkl"

parser = argparse.ArgumentParser(description="Cu melting point estimation")
parser.add_argument(
    "--temperatures", type=int, nargs="+",
    default=DEFAULT_TEMPS,
    help="List of temperatures (K) for which COLVAR files exist",
)
args = parser.parse_args()
temps = sorted(args.temperatures)

base_dir    = os.path.dirname(os.path.abspath(__file__))
melting_dir = os.path.join(base_dir, "RESULTS", "MELTING")
out_dir     = melting_dir
os.makedirs(out_dir, exist_ok=True)

mu_values = []
for T in temps:
    colvar_path = os.path.join(melting_dir, str(T), f"COLVAR_{T}")
    data = np.genfromtxt(colvar_path, skip_header=100)
    mu   = -KAPPA * (np.mean(data[:, 2]) - N0)
    mu_values.append(mu)
    print(f"  T = {T} K  ->  Delta_mu = {mu:.4f} kJ/mol")

temps_arr = np.array(temps)
mu_arr    = np.array(mu_values)

m, b = np.polyfit(temps_arr, mu_arr, 1)
T_melt    = -b / m
T_exp     = 1357.0
rel_error = abs(T_melt - T_exp) / T_exp * 100

print(f"\nEstimated melting point: {T_melt:.2f} K  (exp. {T_exp} K, error {rel_error:.2f}%)")

# --- Plot ---
x_min = min(temps[0], T_melt, T_exp) - 50
x_max = max(temps[-1], T_melt, T_exp) + 50
x_fit = np.linspace(x_min, x_max, 200)

plt.figure(figsize=(8, 6))
plt.scatter(temps_arr, mu_arr, marker="o", s=80, color="blue", label="Simulation data", zorder=3)
plt.plot(x_fit, m * x_fit + b, color="red", label="Linear fit", zorder=2)
plt.scatter([T_melt], [0], color="red", marker="*", s=150, zorder=4,
            label=f"$T_{{melt}}$ = {T_melt:.1f} K")
plt.axvline(T_melt, color="gray", linestyle=":", alpha=0.7)
plt.axvline(T_exp, color="green", linestyle="-.", alpha=0.7,
            label=f"Experimental = {T_exp} K")
plt.scatter([T_exp], [0], color="green", marker="x", s=100, zorder=4)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Temperature (K)")
plt.ylabel(r"$\Delta\mu$ (kJ/mol)")
plt.title("Cu Melting Point Estimation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "melting_point.png"), dpi=300)
plt.close()

# --- Text report ---
report = f"""=============================================
       MELTING POINT ESTIMATION REPORT
=============================================
Model:        {model_name}
Temperatures: {temps} K
Fit:          Delta_mu = {m:.5f} * T + {b:.5f} kJ/mol

Estimated melting point: {T_melt:.2f} K
Experimental value:      {T_exp:.2f} K
Relative error:          {rel_error:.2f} %
=============================================
"""
report_path = os.path.join(out_dir, "melting_point_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Results saved to {out_dir}")
