import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------
# 1. Definizione classi base (le tue)
# -----------------------

class Studente:
    def __init__(self, ore_studio):
        self.ore_studio = ore_studio
        self.punteggio = None

    def generaPunteggio(self):
        # punteggio ~ normale centrata sulle ore di studio, con rumore
        # poi tagliato in [0,100]
        self.punteggio = min(
            100,
            max(
                0,
                int(np.random.normal(loc=self.ore_studio, scale=5))
            )
        )


class PunteggiRandom:
    def __init__(self, num_studenti, seed=42):
        np.random.seed(seed)
        # ore di studio settimanali casuali tra 0 e 100
        self.studenti = [Studente(np.random.randint(0, 100))
                         for _ in range(num_studenti)]

    def calcolaPunteggi(self):
        for studente in self.studenti:
            studente.generaPunteggio()

    def ottieniDati(self):
        ore_studio = np.array([studente.ore_studio for studente in self.studenti], dtype=float)
        punteggi = np.array([studente.punteggio for studente in self.studenti], dtype=float)
        return ore_studio, punteggi


# -----------------------
# 2. Funzioni per le ALTRE variabili
#    (costruite a partire da ore_studio e punteggio)
# -----------------------

def genera_assenze(ore_studio, rumore_std=2.0):
    """
    Assenze totali (0-15).
    Più studi -> in genere meno assenze.

    ~ 15 - 0.12 * ore_studio + rumore
    poi tagliato [0,15]
    """
    base = 15 - 0.12 * ore_studio
    rumore = np.random.normal(loc=0, scale=rumore_std, size=len(ore_studio))
    ass = base + rumore
    ass = np.clip(ass, 0, 15)
    return ass


def genera_attivita_extra(ore_studio, rumore_std=2.5):
    """
    Ore/settimana di attività extra (0-20).
    Chi studia tantissimo tende ad avere meno tempo libero.

    ~ 20 - 0.15 * ore_studio + rumore
    poi [0,20]
    """
    base = 20 - 0.15 * ore_studio
    rumore = np.random.normal(loc=0, scale=rumore_std, size=len(ore_studio))
    extra = base + rumore
    extra = np.clip(extra, 0, 20)
    return extra


def genera_stress(ore_studio, assenze, rumore_std=1.0):
    """
    Livello di stress percepito (1-10).
    Aumenta se studi tanto e se hai molte assenze (ansia da recupero).

    ~ 2 + 0.06 * ore_studio + 0.15 * assenze + rumore
    poi [1,10]
    """
    base = 2 + 0.06 * ore_studio + 0.15 * assenze
    rumore = np.random.normal(loc=0, scale=rumore_std, size=len(ore_studio))
    s = base + rumore
    s = np.clip(s, 1, 10)
    return s


# -----------------------
# 3. Generazione dati
# -----------------------

p = PunteggiRandom(num_studenti=200, seed=42)
p.calcolaPunteggi()
ore_studio, punteggi = p.ottieniDati()

assenze = genera_assenze(ore_studio)
attivita_extra = genera_attivita_extra(ore_studio)
stress = genera_stress(ore_studio, assenze)

df = pd.DataFrame({
    "ore_studio_sett": ore_studio,
    "punteggio_esame": punteggi,
    "assenze": assenze,
    "attivita_extra": attivita_extra,
    "stress": stress
})


# -----------------------
# 4. Statistiche descrittive per lo scatter singolo
# -----------------------

std_ore = np.std(ore_studio, ddof=1)   # dev std campionaria ore_studio
std_punt = np.std(punteggi, ddof=1)    # dev std campionaria punteggi
cov_op = np.cov(ore_studio, punteggi, ddof=1)[0, 1]  # covarianza campionaria


# -----------------------
# 5. Scatter plot Ore di Studio vs Punteggio
# -----------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=ore_studio,
    y=punteggi,
    color="tab:blue",
    edgecolor="white",
    s=60,
    alpha=0.8
)

plt.title('Grafico di Dispersione: Ore di Studio vs Punteggio')
plt.xlabel('Ore di Studio Settimanali')
plt.ylabel('Punteggio Esame')
plt.grid(True, alpha=0.3)

testo_stats = (
    f"Dev. Std (ore studio) = {std_ore:.2f}\n"
    f"Dev. Std (punteggio)  = {std_punt:.2f}\n"
    f"Covarianza            = {cov_op:.2f}"
)

plt.text(
    0.02, 0.98,
    testo_stats,
    transform=plt.gca().transAxes,
    fontsize=11,
    va='top', ha='left',
    bbox=dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        edgecolor='black',
        alpha=0.8
    )
)

plt.tight_layout()
plt.savefig('images/scatterplot_studio_punteggio.png', dpi=300)
plt.show()


# -----------------------
# 6. Scatter matrix (pairplot)
# -----------------------

sns.set_style("whitegrid")

g = sns.pairplot(
    df,
    diag_kind="kde",   # curva di densità sulla diagonale
    plot_kws=dict(
        s=30,
        alpha=0.7,
        edgecolor="white"
    )
)

g.figure.suptitle(
    "Relazioni tra variabili degli studenti\n"
    "(ore di studio, punteggio, assenze, attività extra, stress)",
    y=1.03,
    fontsize=14
)

plt.tight_layout()
plt.savefig('images/scatter_matrix_studenti.png', dpi=300)
plt.show()


# -----------------------
# 7. Heatmap della matrice di correlazione
# -----------------------

corr = df.corr(numeric_only=True)

# Figura più grande
plt.figure(figsize=(8, 6))

ax = sns.heatmap(
    corr,
    cmap="rocket",          # palette tipo viola→arancio chiaro
    annot=True,             # scrivi i valori nella cella
    fmt=".2f",              # due decimali
    vmin=-1, vmax=1,        # scala fissa [-1, 1]
    square=True,            # celle quadrate
    linewidths=0.8,         # bordi celle più visibili
    linecolor="white",
    cbar_kws={
        "shrink": 0.9,
        "label": "Correlazione"
    },
    annot_kws={"fontsize": 11}  # grandezza testo dentro le celle
)

# Titolo più leggibile
plt.title("Matrice di correlazione tra variabili degli studenti", fontsize=14, pad=12)

# Etichette assi più grandi e leggibili
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)

# Etichetta della barra colore più grande
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Correlazione", fontsize=11)

plt.tight_layout()
plt.savefig("images/heatmap_correlazioni_studenti.png", dpi=300)
plt.show()

# -----------------------
# 5bis. Scatter plot con retta di regressione e intervallo di confidenza
# -----------------------

plt.figure(figsize=(10, 6))

# solo linea di regressione (niente CI di seaborn)
sns.regplot(
    data=df,
    x="ore_studio_sett",
    y="punteggio_esame",
    ci=None,
    scatter_kws=dict(
        color="tab:blue",
        edgecolor="white",
        s=60,
        alpha=0.8
    ),
    line_kws=dict(
        color="tab:orange",
        linewidth=2
    )
)

# stima retta
m, q = np.polyfit(df["ore_studio_sett"], df["punteggio_esame"], 1)
x_vals = np.linspace(df["ore_studio_sett"].min(), df["ore_studio_sett"].max(), 200)
y_pred = m * x_vals + q

# banda "didattica" attorno alla retta (scegli tu lo spessore)
delta = 8  # mezzo-spessore della banda
plt.fill_between(
    x_vals,
    y_pred - delta,
    y_pred + delta,
    color="tab:purple",
    alpha=0.2,
    zorder=0
)

plt.title('Scatter plot con retta di regressione e banda di confidenza illustrativa')
plt.xlabel('Ore di studio settimanali')
plt.ylabel('Punteggio esame')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/scatterplot_ci_studio_punteggio.png', dpi=300)
plt.show()
