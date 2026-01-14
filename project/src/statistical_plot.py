import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_t_test_distribution(
    t_statistic: float,
    df: int,
    alpha: float = 0.05,
    transparent_background: bool = True,
    title=None,
    xlabel: str = "t statistic",
    ylabel: str = "Density"
):
    """
    Plot t-distribution with critical region and observed t-statistic.
    One-sided test (greater).
    """

    # Palette obbligatoria
    RED = "#902F1A"
    GREEN = "#564F13"
    DARK_GRAY = "#2B2B2B"
    LIGHT_BG = "#F4EBDC"
    ACCENT = "#D1832F"

    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)

    t_critical = stats.t.ppf(1 - alpha, df)

    fig, ax = plt.subplots(figsize=(8, 4))

    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor(LIGHT_BG)
        ax.set_facecolor(LIGHT_BG)
    

    # Curva t
    ax.plot(x, y, color=DARK_GRAY, linewidth=2)

    # Regione critica (H₁)
    x_crit = np.linspace(t_critical, x.max(), 300)
    ax.fill_between(
        x_crit,
        stats.t.pdf(x_crit, df),
        color=RED,
        alpha=0.25,
        label="Rejection region"
    )

    # Statistica osservata
    ax.axvline(
        t_statistic,
        color=ACCENT,
        linestyle="--",
        linewidth=2,
        label="Observed t"
    )

    # Linea valore critico
    ax.axvline(
        t_critical, # type: ignore
        color=GREEN,
        linestyle=":",
        linewidth=2,
        label="Critical value"
    )

    # Etichette
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    if title not in [None, "", False]:
        ax.set_title(title, fontsize=13, pad=10) # type: ignore

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()