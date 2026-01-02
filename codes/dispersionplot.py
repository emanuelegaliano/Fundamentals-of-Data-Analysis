import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Studente:
    def __init__(self, ore_studio):
        self.ore_studio = ore_studio
        self.punteggio = None
        
    def generaPunteggio(self):
        # punteggio ~ normale centrata sulle ore di studio, con rumore
        self.punteggio = min(100, max(0, int(np.random.normal(loc=self.ore_studio, scale=5))))

class PunteggiRandom:
    def __init__(self, num_studenti):
        # ore di studio settimanali casuali tra 0 e 100
        self.studenti = [Studente(np.random.randint(0, 100)) for _ in range(num_studenti)]
        
    def calcolaPunteggi(self):
        for studente in self.studenti:
            studente.generaPunteggio()
            
    def ottieniDati(self):
        ore_studio = np.array([studente.ore_studio for studente in self.studenti])
        punteggi = np.array([studente.punteggio for studente in self.studenti])
        return ore_studio, punteggi

# -----------------------
# Generazione dei dati
# -----------------------
p = PunteggiRandom(200)
p.calcolaPunteggi()
ore_studio, punteggi = p.ottieniDati()

# -----------------------
# Statistiche descrittive
# -----------------------
std_ore = np.std(ore_studio, ddof=1)      # dev std campionaria ore_studio
std_punt = np.std(punteggi, ddof=1)       # dev std campionaria punteggi
cov_op = np.cov(ore_studio, punteggi, ddof=1)[0, 1]  # covarianza campionaria

# -----------------------
# Plot
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

# Box di testo con deviazioni standard e covarianza
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