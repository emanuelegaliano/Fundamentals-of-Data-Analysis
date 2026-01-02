import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

corsi = ['Matematica', 'Fisica', 'Informatica', 'Biologia', 'Chimica']
studenti = [120, 80, 150, 60, 90]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=corsi, y=studenti, palette='viridis')

ax.set(
    title='Numero di studenti iscritti per corso',
    xlabel='Corsi Universitari',
    ylabel='Numero di Studenti'
)

plt.savefig('barplot_corsi.png')