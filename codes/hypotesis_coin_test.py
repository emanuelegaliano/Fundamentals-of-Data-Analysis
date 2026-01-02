import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# -----------------------
# Parametri del problema
# -----------------------
n = 10          # numero di lanci
p = 0.5         # ipotesi nulla: moneta equa
alpha = 0.05    # livello di significatività (5%)
obs_heads = 9   # teste osservate

# Asse x: possibili numeri di "testa" ottenuti in 10 lanci
k_vals = np.arange(n + 1)

# PMF binomiale sotto H0 (moneta equa)
pmf = binom.pmf(k_vals, n, p)

# regioni estreme per test bilaterale (<=1 testa oppure >=9 teste)
extreme_mask = (k_vals <= 1) | (k_vals >= 9)

# p-value = probabilità di osservare un risultato almeno così estremo
p_value = pmf[k_vals <= 1].sum() + pmf[k_vals >= 9].sum()

# -----------------------
# Stile grafico globale
# -----------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

fig, ax = plt.subplots(figsize=(11, 5.5))

# -----------------------
# Barre non estreme (blu)
# -----------------------
ax.bar(
    k_vals[~extreme_mask],
    pmf[~extreme_mask],
    color="tab:blue",
    alpha=0.8,
    edgecolor="black",
    width=0.8,
    label=r"Probabilità sotto $H_0$"
)

# -----------------------
# Barre estreme (rosse) = p-value
# -----------------------
ax.bar(
    k_vals[extreme_mask],
    pmf[extreme_mask],
    color="tab:red",
    alpha=0.6,
    edgecolor="black",
    width=0.8,
    label="Regioni estreme (p-value)"
)

# -----------------------
# Linea verticale sul valore osservato
# -----------------------
ax.axvline(
    obs_heads,
    color="black",
    linestyle="--",
    linewidth=1.5
)

# Etichetta del valore osservato con freccia
ax.annotate(
    "Osservato:\n9 teste",
    xy=(obs_heads, pmf[obs_heads]),
    xytext=(obs_heads + 0.5, pmf[obs_heads] + 0.07),
    ha="left",
    va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8),
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        lw=1.2,
        shrinkA=4,
        shrinkB=4
    )
)

# -----------------------
# Box riassuntivo in alto a destra
# -----------------------
box_text = fr"""$H_0$: moneta equa ($p = 0.5$)
$H_a$: moneta truccata ($p \neq 0.5$)

Lancio 10 volte $\rightarrow$ 9 teste

$p$-value $= P(X \leq 1) + P(X \geq 9)$
$= {p_value*100:.2f}\%$

$\alpha = {alpha*100:.0f}\%$

Confronto:
$p$-value $< \alpha$
$\Rightarrow$ Rifiutiamo $H_0$
"""

# uso figure.text invece di ax.text così il box va "fuori" dal grafico, in alto a destra
fig.text(
    0.78, 0.80,           # posizione nella figura (percentuale, da 0 a 1)
    box_text,
    ha="left",
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.0)
)

# -----------------------
# Dettagli estetici assi
# -----------------------
ax.set_title("Test di ipotesi sulla moneta: è equa o è truccata?")
ax.set_xlabel("Numero di teste su 10 lanci")
ax.set_ylabel(r"Probabilità sotto $H_0$ (moneta equa)")

ax.set_xticks(k_vals)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, max(pmf) + 0.1)

# Legenda (blu prima, rosso dopo)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left", frameon=True)

plt.tight_layout()
plt.show()