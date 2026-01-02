import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------------------
# 1. Generazione di un dataset sintetico: altezza e peso
# ------------------------------
rng = np.random.default_rng(42)

n_persone = 80

# Altezza in cm: media ~ 172, deviazione ~ 9 cm
altezza = rng.normal(loc=172, scale=9, size=n_persone)
altezza = np.clip(altezza, 150, 200)  # limiti ragionevoli

# Peso in kg: costruiamo una relazione "peso ≈ -100 + 1.1 * altezza + rumore"
rumore_peso = rng.normal(loc=0, scale=5, size=n_persone)
peso = -100 + 1.1 * altezza + rumore_peso
# Evitiamo pesi assurdi
peso = np.clip(peso, 45, 120)

# Mettiamo i dati in forma 2xN per la KDE bivariata
data = np.vstack([altezza, peso])

# ------------------------------
# 2. Stima densità 2D (KDE)
# ------------------------------
kde = gaussian_kde(data)

# Creiamo una griglia 2D su cui valutare la densità stimata
xmin, xmax = altezza.min() - 5, altezza.max() + 5
ymin, ymax = peso.min() - 5, peso.max() + 5

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
)

zz = kde(np.vstack([xx.ravel(), yy.ravel()]))
zz = zz.reshape(xx.shape)

# ------------------------------
# 3. Plot: densità e curve di livello
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- (A) Densità con sfondo colorato + contour
ax = axes[0]

dens_plot = ax.pcolormesh(
    xx, yy, zz,
    shading='auto',
    cmap='magma'  # cambio palette per differenziarlo dall'altro esempio
)

contours = ax.contour(
    xx, yy, zz,
    colors='white',
    linewidths=1.0
)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.3f")

# scatter dei punti
ax.scatter(
    altezza,
    peso,
    s=35,
    color='white',
    edgecolor='black',
    alpha=0.8
)

ax.set_title("Altezza vs Peso: densità stimata")
ax.set_xlabel("Altezza (cm)")
ax.set_ylabel("Peso (kg)")
ax.grid(alpha=0.2)

# Barra colore per interpretare la densità
cbar = fig.colorbar(dens_plot, ax=ax, shrink=0.8)
cbar.set_label("Densità stimata (KDE)")

# --- (B) Solo curve di livello
ax2 = axes[1]

contours2 = ax2.contour(
    xx, yy, zz,
    cmap='magma',
    linewidths=1.5
)
ax2.clabel(contours2, inline=True, fontsize=8, fmt="%.3f")

ax2.scatter(
    altezza,
    peso,
    s=35,
    color='gray',
    edgecolor='black',
    alpha=0.8
)

ax2.set_title("Altezza vs Peso: curve di livello")
ax2.set_xlabel("Altezza (cm)")
ax2.set_ylabel("Peso (kg)")
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("images/density_contour_altezza_peso.png", dpi=300)
plt.show()