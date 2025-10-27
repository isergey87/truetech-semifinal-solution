import matplotlib.pyplot as plt
import numpy as np

from detect_walls import detect_walls
from lidar_scans import SCAN_1, SCAN_2, SCAN_3, SCAN_4

range_points = SCAN_4

angles_deg = range_points[:, 0]
ranges = range_points[:, 1]
angles_rad = np.deg2rad(angles_deg)
x = ranges * np.cos(angles_rad)
y = ranges * np.sin(angles_rad)
points = np.stack((x, y), axis=1)

merged_segments = detect_walls(range_points)

# ================= ВИЗУАЛИЗАЦИЯ =================
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=4, label="Lidar points")
colors = plt.cm.tab10(np.linspace(0, 1, len(merged_segments)))
for segment, c in zip(merged_segments, colors):
    p1 = segment['start_point']
    p2 = segment['end_point']
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, lw=3)
    a = np.degrees(segment['angle'])
    plt.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, f"{a:.1f}°", color=c)
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
