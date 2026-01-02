import numpy as np
import matplotlib.pyplot as plt

# 1. Generiamo un campione di dati (es. "altezza" in cm)
#    ipotizziamo che le altezze siano circa normali con media 170 e std 10
np.random.seed(42)          # per avere sempre lo stesso risultato
dati = np.random.normal(loc=170, scale=10, size=300)

# 2. Creiamo l'istogramma
plt.figure(figsize=(8, 4))

conteggi, bordi, _ = plt.hist(
    dati,
    bins=15,                # numero di "secchi" dell'istogramma
    density=True,           # normalizza l'area a 1 (stima della densità)
    edgecolor="black",      # bordo delle barre
    alpha=0.7               # trasparenza per renderlo più leggibile
)

# 3. Aggiungiamo la linea della media del campione
media = np.mean(dati)
plt.axvline(media, color="red", linestyle="--", linewidth=2,
            label=f"Media campionaria = {media:.1f} cm") 

# 4. Etichette e titolo
plt.title("Distribuzione delle altezze (istogramma)")
plt.xlabel("Altezza (cm)")
plt.ylabel("Densità stimata")

plt.legend()

plt.tight_layout()
# plt.show()

# Se vuoi salvarlo invece di mostrarlo:
plt.savefig("istogramma_altezze.png", dpi=300)
