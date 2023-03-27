# This script is only for comparing our model with the baseline
# Author: Renjie Xu
# Time: 2023/3/26

import numpy as np
import matplotlib.pyplot as plt

# Create random data with 5 discrete points
fps = np.array([0, 15, 30, 45, 60])
mota1 = np.array([0.9, 0.8, 0.75, 0.7, 0.6])
mota2 = np.array([0.85, 0.77, 0.72, 0.65, 0.55])
mota3 = np.array([0.8, 0.73, 0.68, 0.6, 0.5])

# Plot the curves
plt.plot(fps, mota1, marker='o', label='Curve 1', linewidth=2)
plt.plot(fps, mota2, marker='s', label='Curve 2', linewidth=2)
plt.plot(fps, mota3, marker='D', label='Curve 3', linewidth=2)

# Customize the plot
plt.xlabel('FPS')
plt.ylabel('MOTA')
plt.legend()

# Customize axis lines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set tick direction to 'in'
ax.tick_params(direction='in')

# Save the plot to a file
plt.savefig('mota_vs_fps.png', dpi=300, bbox_inches='tight')