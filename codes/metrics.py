import matplotlib.pyplot as plt
import numpy as np

# Due punti
p1 = np.array([1, 2])
p2 = np.array([4, 6])

fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharex=True, sharey=True)

POINT_SIZE = 60
LINE_WIDTH = 3

# ---------- L2 ----------
ax = axes[0]
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=POINT_SIZE)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
        linestyle='--', linewidth=LINE_WIDTH)
ax.set_title('Distanza L2 (Euclidea)')
ax.set_aspect('equal')
ax.grid(True)

# ---------- L1 ----------
ax = axes[1]
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=POINT_SIZE)
ax.plot([p1[0], p2[0]], [p1[1], p1[1]],
        linestyle=':', linewidth=LINE_WIDTH)
ax.plot([p2[0], p2[0]], [p1[1], p2[1]],
        linestyle=':', linewidth=LINE_WIDTH)
ax.set_title('Distanza L1 (Manhattan)')
ax.set_aspect('equal')
ax.grid(True)

# ---------- L∞ ----------
ax = axes[2]
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=POINT_SIZE)
ax.plot([p1[0], p2[0]], [p1[1], p1[1]],
        linestyle='-.', linewidth=LINE_WIDTH)
ax.set_title('Distanza L∞ (Chebyshev)')
ax.set_aspect('equal')
ax.grid(True)

# Etichette comuni
for ax in axes:
    ax.set_xlabel('x')
axes[0].set_ylabel('y')

plt.tight_layout()
# plt.savefig("metrics_comparison.png")
plt.show()