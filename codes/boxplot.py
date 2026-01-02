import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

student_scores = np.random.normal(loc=19, scale=5, size=100)

plt.figure(figsize=(6, 8))
sns.boxplot(y=student_scores, color='lightblue')
plt.title('Boxplot dei punteggi degli studenti')
plt.ylabel('Punteggi')
plt.grid(axis='x')
plt.savefig('boxplot_punteggi_studenti.png')