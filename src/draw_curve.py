import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quad_func(x, a, b, c):
    return a * x**2 + b * x + c

full_fps = np.array([16.7, 23.3, 28.9, 30.3, 32.2])               # FPS
full_mota = np.array([70.7, 68.8, 65.0, 62.9, 58.9])             # MOTA

half_fps = np.array([24.8, 30.0, 34.3, 35.6, 36.6])               # FPS
half_mota = np.array([65.5, 62.6, 56.6, 53.9, 51.3])            # MOTA

quarter_fps = np.array([28.7, 33.3, 36.0, 36.8, 37.6])               # FPS
quarter_mota = np.array([60.2, 58.2, 54.5, 52.2, 47.8])            # MOTA

DC_fps = np.array([18.0, 23.7, 28.4, 30.3, 31.8, 32.2])               # FPS
DC_mota = np.array([70.7, 69.5, 67.8, 65.4, 62.0, 61.1])          # MOTA

# DC_fps = np.array([18.0, 23.7, 28.9, 30.4, 32.4, 31.8, 32.2, 32.4, 32.1, 33.2, 32.9, 34.2, 34.9, 35.5])               # FPS
# DC_mota = np.array([70.7, 69.3, 67.6, 65.4, 62.0, 62.0, 61.1, 60.3, 59.4, 58.4, 57.3, 55.1, 53.9, 53.2])          # MOTA

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
plt.scatter(full_fps, full_mota, color='red', marker='o', label='Full-DLA-34')
plt.scatter(half_fps, half_mota, color='blue', marker='^', label='Half-DLA-34')
plt.scatter(quarter_fps, quarter_mota, color='cyan', marker='s', label='Quarter-DLA-34')
plt.scatter(DC_fps, DC_mota, color='green', marker='*', label='DeepScale')
# plt.scatter(DC_fps_classifier, DC_mota_classifier, color='magenta', marker='X', label='DeepScale_classifier')
plt.legend(loc='lower left')
plt.xlabel("FPS")
plt.ylabel("MOTA")
plt.ylim(45, 72)
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
