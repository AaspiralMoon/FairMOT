import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quad_func(x, a, b, c):
    return a * x**2 + b * x + c

full_fps = np.array([38.6, 41.8, 42.6, 43.2, 43.6])               # FPS
full_mota = np.array([62.9, 61.4, 58.5, 57.2, 54.0])             # MOTA

half_fps = np.array([39.6, 42.7, 43.6, 44.1, 44.5])               # FPS
half_mota = np.array([58.3, 56.8, 52.2, 50.4, 44.5])            # MOTA

quarter_fps = np.array([42.6, 43.9, 45.6, 46.2, 46.5])               # FPS
quarter_mota = np.array([52.6, 51.6, 46.5, 43.3, 37.7])            # MOTA

DC_fps = np.array([40.0, 42.5, 43.1, 43.4, 44.2, 44.7, 45.6, 46.3])               # FPS
DC_mota = np.array([63.7, 63.0, 60.7, 59.8, 57.8, 52.1, 47.4, 45.0])          # MOTA

# Fit the quadratic functions
full_popt, _ = curve_fit(quad_func, full_fps, full_mota)
half_popt, _ = curve_fit(quad_func, half_fps, half_mota)
quarter_popt, _ = curve_fit(quad_func, quarter_fps, quarter_mota)
DC_popt, _ = curve_fit(quad_func, DC_fps, DC_mota)

# Generate x values for the fitted curves
full_x = np.linspace(np.min(full_fps), np.max(full_fps), 100)
half_x = np.linspace(np.min(half_fps), np.max(half_fps), 100)
quarter_x = np.linspace(np.min(quarter_fps), np.max(quarter_fps), 100)
DC_x = np.linspace(np.min(DC_fps), np.max(DC_fps), 100)

# Plot the fitted curves
plt.plot(full_x, quad_func(full_x, *full_popt), 'r-')
plt.plot(half_x, quad_func(half_x, *half_popt), 'b-')
plt.plot(quarter_x, quad_func(quarter_x, *quarter_popt), 'c-')
plt.plot(DC_x, quad_func(DC_x, *DC_popt), 'g-')

# Plot the original points
plt.scatter(full_fps, full_mota, color='red', marker='o', label='Full-Yolo')
plt.scatter(half_fps, half_mota, color='blue', marker='^', label='Half-Yolo')
plt.scatter(quarter_fps, quarter_mota, color='cyan', marker='s', label='Quarter-Yolo')
plt.scatter(DC_fps, DC_mota, color='green', marker='*', label='DeepScale')

plt.legend(loc='lower left')
plt.xlabel("FPS")
plt.ylabel("MOTA")
# resolutions = ['1088', '864', '704', '640', '576']
# for i, txt in enumerate(resolutions):
#     plt.annotate(txt, (full_fps[i], full_mota[i]))
#     plt.annotate(txt, (half_fps[i], half_mota[i]))
#     plt.annotate(txt, (quarter_fps[i], quarter_mota[i]))

# thresh_settings = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
# for i, txt in enumerate(thresh_settings):
#     plt.annotate(txt, (DC_fps[i], DC_mota[i]))

# Save the plot to a file
plt.savefig('mota_vs_fps.png', dpi=300, bbox_inches='tight')
