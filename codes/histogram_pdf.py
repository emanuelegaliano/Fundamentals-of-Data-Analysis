import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # per la PDF normale

# 1. Generiamo un campione di dati (es. altezze in cm)
np.random.seed(42)  # per riproducibilità
dati = np.random.normal(loc=170, scale=10, size=300)

# 2. Calcoliamo media e dev std campionarie
media = np.mean(dati)
std = np.std(dati, ddof=1)  # ddof=1 -> stima campionaria

# 3. Creiamo l'istogramma normalizzato (stima empirica della densità)
plt.figure(figsize=(10, 5))
plt.hist(
    dati,
    bins=15,
    density=True,          # rende l'area totale = 1 → PDF stimata
    edgecolor="black",
    alpha=0.7,
    color="#4a7dad",
    label="Istogramma (densità empirica)"
)

# 4. Sovrapponiamo la PDF normale stimata dal campione
x = np.linspace(min(dati) - 5, max(dati) + 5, 300)
pdf_normale = norm.pdf(x, loc=media, scale=std)

plt.plot(
    x,
    pdf_normale,
    color="darkred",
    linewidth=2,
    label=f"PDF normale stimata\n($\\mu$={media:.1f}, $\\sigma$={std:.1f})"
)

# 5. Linea della media campionaria
plt.axvline(
    media,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Media campionaria = {media:.1f} cm"
)

# 6. Etichette e titolo
plt.title("Distribuzione delle altezze: istogramma e PDF normale stimata")
plt.xlabel("Altezza (cm)")
plt.ylabel("Densità di probabilità")

plt.legend()
plt.tight_layout()

# Mostra o salva
# plt.show()
plt.savefig("istogramma_altezze_pdf_normale.png", dpi=300)
