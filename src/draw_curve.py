import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

full_fps = np.array([16.7, 23.1, 28.3, 30.3, 31.5])               # FPS
full_mota = np.array([71.8, 69.6, 65.5, 63.3, 59.3])             # MOTA

half_fps = np.array([23.4, 29.5, 33.7, 35.3, 35.6])               # FPS
half_mota = np.array([66.6, 63.9, 57.8, 55.0, 52.3])            # MOTA

quarter_fps = np.array([27.8, 32.7, 35.3, 36.8, 36.7])               # FPS
quarter_mota = np.array([61.5, 59.5, 57.1, 53.1, 48.3])            # MOTA

DC_fps = np.array([30.5, 30.5, 28.2, 26.6, 25.1, 22.8, 21.2, 18.0, 24.2, 28.0, 29.2, 17.0, 17.7])               # FPS
DC_mota = np.array([59.5, 60.7, 62.1, 64.2, 66.5, 68.4, 70.5, 71.9, 69.7, 67.9, 66.2, 71.5, 71.0])          # MOTA

# DC_fps_classifier = np.array([32.7, 30.8, 31.0, 28.7, 26.6, 23.8])               # FPS
# DC_mota_classifier = np.array([58.3, 59.2, 60.0, 62.4, 63.9, 65.9])          # MOTA

# Plot the original points
plt.scatter(full_fps, full_mota, color='red', marker='o', label='Full-DLA-34')
plt.scatter(half_fps, half_mota, color='blue', marker='^', label='Half-DLA-34')
plt.scatter(quarter_fps, quarter_mota, color='cyan', marker='s', label='Quarter-DLA-34')
plt.scatter(DC_fps, DC_mota, color='green', marker='*', label='DeepScale')
# plt.scatter(DC_fps_classifier, DC_mota_classifier, color='magenta', marker='X', label='DeepScale_classifier')
plt.legend(loc='lower left')
plt.xlabel("FPS")
plt.ylabel("MOTA")
resolutions = ['1088', '864', '704', '640', '576']
for i, txt in enumerate(resolutions):
    plt.annotate(txt, (full_fps[i], full_mota[i]))
    plt.annotate(txt, (half_fps[i], half_mota[i]))
    plt.annotate(txt, (quarter_fps[i], quarter_mota[i]))

thresh_settings = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']
for i, txt in enumerate(thresh_settings):
    plt.annotate(txt, (DC_fps[i], DC_mota[i]))

# Save the plot to a file
plt.savefig('mota_vs_fps.png', dpi=300, bbox_inches='tight')
