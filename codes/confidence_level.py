import matplotlib
matplotlib.use("Agg")  # usa backend non interattivo (niente Qt richiesto)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------
# Parametri del problema
# -------------------------

alpha = 0.05                # livello di significatività (probabilità di "mancare" μ)
conf_level = 1 - alpha      # livello di confidenza (es. 0.95)
mu = 0                      # media vera della popolazione
sigma = 1                   # deviazione standard della popolazione

# Distribuzione normale standard (per Z = (X̄ - μ) / SE)
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)

# Valore critico z* (quantile della normale standard)
# Per alpha = 0.05 ⇒ z_crit ≈ 1.96
z_crit = norm.ppf(1 - alpha / 2)

# -------------------------
# Plot
# -------------------------

fig, ax = plt.subplots(figsize=(10, 5))

# Curva della densità normale standard
ax.plot(
    x, pdf,
    color='black',
    linewidth=2,
    label='Distribuzione delle medie campionarie'
)

# Banda centrale: intervallo che cattura μ con probabilità (1 - α)
ax.fill_between(
    x, 0, pdf,
    where=(x >= -z_crit) & (x <= z_crit),
    color='tab:green',
    alpha=0.4,
    label=fr'Intervallo di confidenza $(1-\alpha)$ = {conf_level:.0%}'
)

# Code: le situazioni in cui l'intervallo NON contiene μ (probabilità totale α)
ax.fill_between(
    x, 0, pdf,
    where=(x < -z_crit),
    color='tab:red',
    alpha=0.4,
    label=fr'Errore totale $\alpha = {alpha:.2f}$'
)
ax.fill_between(
    x, 0, pdf,
    where=(x > z_crit),
    color='tab:red',
    alpha=0.4
)

# Linee verticali ai bordi dell'intervallo critico ±z*
ax.axvline(-z_crit, color='tab:green', linestyle='--', linewidth=2)
ax.axvline(z_crit, color='tab:green', linestyle='--', linewidth=2)

# Linea verticale sulla media vera μ
ax.axvline(mu, color='black', linestyle=':', linewidth=2)
ax.text(
    mu,
    norm.pdf(mu, mu, sigma) + 0.005,
    r'$\mu$',
    ha='center',
    va='bottom',
    fontsize=12
)

# Testo esplicativo sul corpo centrale (parte verde)
ax.text(
    0,
    0.02,
    fr"L'intervallo stimato $\bar{{X}} \pm {z_crit:.2f} \cdot SE(\bar{{X}})$"
    + f"\n(contiene $\\mu$ con probabilità {conf_level:.0%})",
    color='tab:green',
    ha='center',
    va='bottom',
    fontsize=11,
)

# Annotazioni sulle code (α/2 ciascuna)
ax.text(
    -3,
    0.015,
    r'$\alpha/2$',
    color='tab:red',
    ha='center',
    fontsize=11,
)
ax.text(
    3,
    0.015,
    r'$\alpha/2$',
    color='tab:red',
    ha='center',
    fontsize=11,
)

# Titolo e assi
ax.set_title(
    r"Interpretazione di $\alpha$ e $(1-\alpha)$ nell'intervallo di confidenza",
    fontsize=14
)

ax.set_xlabel(
    r"Possibili valori stimati per la media campionaria $\bar{X}$"
)
ax.set_ylabel("Densità di probabilità")

ax.set_xlim([-4, 4])
ax.set_ylim(bottom=0)

# Legenda
ax.legend(loc='upper left', fontsize=10, frameon=True)

plt.tight_layout()

# Salva la figura su file invece di aprire una finestra grafica interattiva
# plt.savefig("confidence_level_plot.png", dpi=200)

# Se vuoi anche vederla in una sessione interattiva locale, puoi comunque:
plt.show()