import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

altezze = np.random.normal(170, 10, 1000)  # Altezza in cm
pesi = np.random.normal(70, 15, 1000)      # Peso

plt.figure(figsize=(8, 6))
hb = plt.hexbin(altezze, pesi, gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(hb, label='Densità di punti')
plt.xlabel('Altezza (cm)')
plt.ylabel('Peso (kg)')
plt.title('Hexbin plot di Altezza vs Peso') 

plt.savefig('images/hexbin_altezza_peso.png', dpi=300)
plt.show()