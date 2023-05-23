import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quad_func(x, a, b, c):
    return a * x**2 + b * x + c

CO_fps = np.array([15.7, 19.1, 24.9, 24.6, 26.5, 29.1])               # FPS
CO_mota = np.array([70.7, 69.5, 67.8, 65.4, 62.0, 61.1])             # MOTA

SO_fps = np.array([7.9, 9.9, 11.7, 12.0, 12.8, 13.1])               # FPS
SO_mota = np.array([70.2, 69.5, 67.5, 64.8, 61.9, 61.1])            # MOTA

SOAT_fps = np.array([13.0, 20.4, 26.8, 29.4, 30.5, 31.2])               # FPS
SOAT_mota = np.array([69.6, 68.7, 65.7, 63.4, 60.4, 60.0])            # MOTA

SAT_fps = np.array([15.3, 18.5, 23.8, 24.3, 26.5, 28.0])               # FPS
SAT_mota = np.array([70.4, 69.6, 67.7, 64.8, 61.7, 60.9])          # MOTA

# Fit the quadratic functions
full_popt, _ = curve_fit(quad_func, CO_fps, CO_mota)
half_popt, _ = curve_fit(quad_func, SO_fps, SO_mota)
quarter_popt, _ = curve_fit(quad_func, SOAT_fps, SOAT_mota)
DC_popt, _ = curve_fit(quad_func, SAT_fps, SAT_mota)

# Generate x values for the fitted curves
full_x = np.linspace(np.min(CO_fps), np.max(CO_fps), 100)
half_x = np.linspace(np.min(SO_fps), np.max(SO_fps), 100)
quarter_x = np.linspace(np.min(SOAT_fps), np.max(SOAT_fps), 100)
DC_x = np.linspace(np.min(SAT_fps), np.max(SAT_fps), 100)

# Plot the fitted curves
plt.plot(full_x, quad_func(full_x, *full_popt), 'r-')
plt.plot(half_x, quad_func(half_x, *half_popt), 'b-')
plt.plot(quarter_x, quad_func(quarter_x, *quarter_popt), 'c-')
plt.plot(DC_x, quad_func(DC_x, *DC_popt), 'g-')

# Plot the original points
plt.scatter(CO_fps, CO_mota, color='red', marker='o', label='CO')
plt.scatter(SO_fps, SO_mota, color='blue', marker='^', label='SO')
plt.scatter(SOAT_fps, SOAT_mota, color='cyan', marker='s', label='SOAT')
plt.scatter(SAT_fps, SAT_mota, color='green', marker='*', label='SAT')
plt.legend(loc='lower left')
plt.xlabel("FPS")
plt.ylabel("MOTA")
plt.ylim(60, 72)

# Save the plot to a file
plt.savefig('mota_vs_fps_arch.png', dpi=300, bbox_inches='tight')
