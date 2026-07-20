from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

desired_temperatures = np.array([800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]) # K

melting_point = 1357 # K

# Brillo, J. and Egry, I., "Density Determination of Liquid Copper, Nickel, and Their Alloys",
# International Journal of Thermophysics, 24(4), 1155-1170 (2003). doi: 10.1023/A:1025021521945
dens_l = np.array([7.886,7.784,7.730,7.670,7.655,7.559]) # g/cm^3
temp_l = np.array([1400, 1490,1564,1652,1711,1798]) # K

# Straumanis, M. E. and Yu, L. S., "Lattice Parameters, Densities, Expansion Coefficients and
# Perfection of Structure of Cu and of Cu-In alpha Phase"
dens_s_0 = 8.9314 # g/cm^3 at 298 K
temp_s_0 = 298 # K

# James, J D, Spittle, J A, Brown, S G R and Evans, R W, "A review of measurement techniques
# for the thermal expansion coefficient of metals and alloys at elevated temperatures"
alpha_values =  np.array([16.85, 17.47, 18.12, 18.83, 19.62, 20.48, 21.40, 22.39, 23.46, 24.59, 25.79]) * 1e-6 # K^-1
alpha_temp = np.array([293, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])

desired_temperatures_s = desired_temperatures[desired_temperatures < melting_point]

dens_s = []
for T in desired_temperatures_s:
    # V(T) = V(T_0) * (1 + 3 * integral[T_0, T] alpha(T') dT')
    T_range = np.linspace(temp_s_0, T, 500)
    alpha_interp = np.interp(T_range, alpha_temp, alpha_values)
    integral_alpha = np.trapezoid(alpha_interp, T_range)
    volumetric_expansion = 1 + 3 * integral_alpha
    dens_s.append(dens_s_0 / volumetric_expansion)

dens_s = np.array(dens_s)

print("Solid Temperatures (K):", desired_temperatures_s)
print("Solid Densities (g/cm^3):", dens_s)

desired_temperatures_l = desired_temperatures[desired_temperatures > melting_point]

liquid_fit = np.polyfit(temp_l, dens_l, 1)
liquid_p = np.poly1d(liquid_fit)
dens_l_pred = liquid_p(desired_temperatures_l)

print("Liquid Temperatures (K):", desired_temperatures_l)
print("Liquid Densities (g/cm^3):", dens_l_pred)

plt.figure(figsize=(10, 6))
plt.plot(desired_temperatures_s, dens_s, 'o-', color='blue', label='Solid Density (Calculated)')
plt.plot(desired_temperatures_l, dens_l_pred, 'o-', color='red', label='Liquid Density (Linear Fit)')
plt.scatter(temp_l, dens_l, color='darkred', marker='x', s=60, label='Experimental Liquid Data')
plt.axvline(x=melting_point, color='black', linestyle='--', label=f'Melting Point ({melting_point} K)')

plt.title('Copper Density vs Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Density (g/cm³)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

save_path = Path(__file__).resolve().parent / 'Cu_densities_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {save_path}")
