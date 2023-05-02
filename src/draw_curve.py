import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

full_fps = np.array([16.7, 23.3, 28.9, 30.3, 32.2])               # FPS
full_mota = np.array([70.7, 68.8, 65.0, 62.9, 58.9])             # MOTA

half_fps = np.array([24.8, 30.0, 34.3, 35.6, 36.6])               # FPS
half_mota = np.array([65.5, 62.6, 56.6, 53.9, 51.3])            # MOTA

quarter_fps = np.array([28.7, 33.3, 36.0, 36.8, 37.6])               # FPS
quarter_mota = np.array([60.2, 58.2, 54.5, 52.2, 47.8])            # MOTA

DC_fps = np.array([21.4, 18.2, 23.5, 28.4, 30.6, 18.3, 17.4, 24.6, 28.4, 30.9, 32.8])               # FPS
DC_mota = np.array([69.6, 70.8, 69.6, 67.1, 65.5, 70.3, 70.6, 69.4, 67.0, 65.2, 61.9])          # MOTA

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

thresh_settings = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
for i, txt in enumerate(thresh_settings):
    plt.annotate(txt, (DC_fps[i], DC_mota[i]))

# Save the plot to a file
plt.savefig('mota_vs_fps.png', dpi=300, bbox_inches='tight')
